from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

import wandb


import buffer_gap

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Algorithm specific arguments
    env_id: str = "PushText-v1"
    """the id of the environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 2
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 512
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.95
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    wandb_video_freq: int = 200_000
    """log an eval video to wandb every this many environment steps (0 = never)"""
    checkpoint_freq: int = 200_000
    """save a model checkpoint every this many environment steps (0 = only save final)"""
    finite_horizon_gae: bool = False
    use_async_vector_env: bool = False
    """if toggled, use gym.vector.AsyncVectorEnv/SyncVectorEnv over ManiSkill internal vectorization"""
    return_buffer_size: int = 10_000
    """size of the episode-return buffer for sub-optimality gap tracking"""
    top_return_buff_percentage: float = 0.05
    """top-k% of returns to use as the experience-optimal estimate"""
    plot_freq: int = 5
    """log gap metrics every this many PPO iterations"""
    fixed_layout: bool = False
    """if toggled, letter tiles spawn at fixed positions every episode (no randomization)"""
    bc_coef: float = 0.1
    """weight on the behavior cloning loss from top-return trajectories"""
    bc_batch_size: int = 256
    """number of observation-action pairs sampled from the return buffer per optimizer step"""


    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

    def get_value(self, x):
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: Optional[SummaryWriter] = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        if self.writer is not None:
            self.writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup — training/eval envs never render; video capture uses a separate on-demand env
    # During --evaluate mode, rendering is always enabled for trajectory saving
    render_mode = "rgb_array" if (args.capture_video and args.evaluate) else None
    render_backend = "gpu" if (args.capture_video and args.evaluate) else "none"
    env_kwargs: dict[str, object] = dict(obs_mode="state", render_mode=render_mode, render_backend=render_backend, sim_backend="physx_cpu")
    if args.fixed_layout:
        env_kwargs["fixed_layout"] = True
    # Separate kwargs for on-demand video capture (always GPU-rendered)
    video_env_kwargs: dict[str, object] = dict(obs_mode="state", render_mode="rgb_array", render_backend="gpu", sim_backend="physx_cpu")
    if args.fixed_layout:
        video_env_kwargs["fixed_layout"] = True
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
        video_env_kwargs["control_mode"] = args.control_mode

    probe_env = gym.make(args.env_id, **env_kwargs)
    max_episode_steps = gym_utils.find_max_episode_steps_value(probe_env)
    probe_env.close()

    train_num_envs = args.num_envs if not args.evaluate else 1

    eval_output_dir = None
    policy_video_dir = None
    deterministic_eval_video_dir = None
    if args.capture_video:
        if args.evaluate:
            assert args.checkpoint is not None
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        else:
            eval_output_dir = f"runs/{run_name}/videos"
            policy_video_dir = f"runs/{run_name}/videos/policy"
            deterministic_eval_video_dir = f"runs/{run_name}/videos/deterministic_eval"
        print(f"Saving eval videos to {eval_output_dir}")

    def _make_cpu_env(seed: int, reconfiguration_freq: Optional[int], ignore_terminations: bool,
                      video_output_dir: Optional[str] = None):
        def _thunk():
            env = gym.make(args.env_id, reconfiguration_freq=reconfiguration_freq, **env_kwargs)
            if isinstance(env.action_space, gym.spaces.Dict):
                env = FlattenActionSpaceWrapper(env)
            if video_output_dir is not None:
                env = RecordEpisode(env, output_dir=video_output_dir, save_trajectory=args.evaluate,
                                    trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
            env = CPUGymWrapper(env, ignore_terminations=ignore_terminations, record_metrics=True)
            import buffer_gap
            env = buffer_gap.RecordEpisodeStatisticsV2(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return _thunk

    if not args.use_async_vector_env:
        envs = gym.make(args.env_id, num_envs=train_num_envs, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
        eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
        if isinstance(envs.action_space, gym.spaces.Dict):
            envs = FlattenActionSpaceWrapper(envs)
        if isinstance(eval_envs.action_space, gym.spaces.Dict):
            eval_envs = FlattenActionSpaceWrapper(eval_envs)
    else:
        # CPU multiprocessing path — each env is a separate process
        train_vector_cls = gym.vector.SyncVectorEnv if train_num_envs == 1 else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
        eval_vector_cls = gym.vector.SyncVectorEnv if args.num_eval_envs == 1 else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")

        envs = train_vector_cls([
            _make_cpu_env(seed=args.seed + i, reconfiguration_freq=args.reconfiguration_freq, ignore_terminations=not args.partial_reset)
            for i in range(train_num_envs)
        ])
        eval_envs = eval_vector_cls([
            _make_cpu_env(seed=args.seed + 100000 + i, reconfiguration_freq=args.eval_reconfiguration_freq,
                          ignore_terminations=not args.eval_partial_reset,
                          video_output_dir=eval_output_dir if (args.evaluate and args.capture_video and i == 0) else None)
            for i in range(args.num_eval_envs)
        ])


    if args.capture_video and not args.use_async_vector_env:
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        if args.evaluate:
            eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=True, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    if not args.use_async_vector_env:
        envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
        eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    def _make_gpu_video_env(out_dir: str, num_steps: int):
        """Create a temporary GPU-rendered env with RecordEpisode for one episode."""
        os.makedirs(out_dir, exist_ok=True)
        vid_env = gym.make(args.env_id, num_envs=1, **video_env_kwargs)
        if isinstance(vid_env.action_space, gym.spaces.Dict):
            vid_env = FlattenActionSpaceWrapper(vid_env)
        vid_env = RecordEpisode(vid_env, output_dir=out_dir, save_trajectory=False,
                                max_steps_per_video=num_steps, video_fps=30)
        return vid_env

    def _latest_video(out_dir: str) -> Optional[str]:
        videos = sorted(
            [f for f in os.listdir(out_dir) if f.endswith(".mp4")],
            key=lambda f: os.path.getmtime(os.path.join(out_dir, f)),
        )
        return os.path.join(out_dir, videos[-1]) if videos else None

    def record_video_policy(step: int = 0) -> Optional[str]:
        """Record the current deterministic policy for one episode (seed varies by step)."""
        if policy_video_dir is None:
            return None
        try:
            vid_env = _make_gpu_video_env(policy_video_dir, args.num_eval_steps)
            vid_env = ManiSkillVectorEnv(vid_env, 1, ignore_terminations=True, record_metrics=False)
            obs_v, _ = vid_env.reset(seed=args.seed + step)
            obs_v = to_tensor(obs_v)
            agent.eval()
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    obs_v, _, _, _, _ = vid_env.step(agent.get_action(obs_v, deterministic=True))
                    obs_v = to_tensor(obs_v)
            vid_env.close()
        except Exception as e:
            print(f"Warning: policy video capture failed: {e}")
            return None
        return _latest_video(policy_video_dir)

    def record_video_deterministic_eval(step: int = 0) -> Optional[str]:
        """Record eval_deterministic() running on a GPU-rendered env (fixed seed=args.seed)."""
        if deterministic_eval_video_dir is None:
            return None
        try:
            vid_env = _make_gpu_video_env(deterministic_eval_video_dir, args.num_eval_steps)
            agent.eval()
            gap_stats.eval_deterministic(envs=vid_env)
            vid_env.close()
        except Exception as e:
            print(f"Warning: deterministic eval video capture failed: {e}")
            return None
        return _latest_video(deterministic_eval_video_dir)

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.as_tensor(x, device=device)

    def select_by_mask(x, mask: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return x[mask]
        if isinstance(x, list):
            x = np.asarray(x, dtype=object)
        selected = x[mask.cpu().numpy()]
        # gymnasium 0.29 may return object arrays containing numpy arrays; stack them
        if isinstance(selected, np.ndarray) and selected.dtype == object:
            items = [o for o in selected if o is not None]
            if items:
                selected = np.stack(items).astype(np.float32)
        return selected

    def extract_episode_metrics(info_dict, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        # ManiSkillVectorEnv emits dict-of-batched-values, while Gym AsyncVectorEnv can emit
        # a per-environment object array for final_info.
        final_info = info_dict["final_info"]
        if isinstance(final_info, dict) and "episode" in final_info and isinstance(final_info["episode"], dict):
            metrics = {}
            for k, v in final_info["episode"].items():
                vals = to_tensor(select_by_mask(v, mask)).reshape(-1)
                if vals.numel() > 0:
                    metrics[k] = vals
            return metrics

        mask_np = mask.cpu().numpy()
        aggregated = defaultdict(list)
        final_info_arr = final_info
        if isinstance(final_info_arr, list):
            final_info_arr = np.asarray(final_info_arr, dtype=object)
        for i, done in enumerate(mask_np):
            if not done:
                continue
            env_info = final_info_arr[i]
            if env_info is None or "episode" not in env_info:
                continue
            for k, v in env_info["episode"].items():
                aggregated[k].append(v)
        return {k: to_tensor(np.asarray(v)).reshape(-1) for k, v in aggregated.items()}

    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                monitor_gym=True,
                save_code=True,
                group="PPO",
                tags=["ppo", "walltime_efficient"]
            )
        if SummaryWriter is not None:
            writer = SummaryWriter(f"runs/{run_name}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )
        else:
            writer = None
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    gap_eval_env = gym.vector.SyncVectorEnv([
        _make_cpu_env(seed=args.seed + 200000, reconfiguration_freq=None, ignore_terminations=False)
    ])
    gap_stats = buffer_gap.BufferGapV2(
        args.return_buffer_size, args.top_return_buff_percentage,
        policy=agent, device=device, args=args, envs=gap_eval_env,
    )
    _ep_return_buf = torch.zeros(args.num_envs, device=device)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_obs = to_tensor(next_obs)
    eval_obs = to_tensor(eval_obs)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    action_space_low, action_space_high = torch.from_numpy(envs.single_action_space.low).to(device), torch.from_numpy(envs.single_action_space.high).to(device)
    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    last_video_log_step = -args.wandb_video_freq  # ensure first video logs at step 0+freq
    last_ckpt_step = -args.checkpoint_freq  # ensure first checkpoint saves at step 0+freq

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()
            eval_obs = to_tensor(eval_obs)
            eval_metrics = defaultdict(list)
            eval_step_rewards = []
            eval_step_successes = []
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    eval_obs = to_tensor(eval_obs)
                    eval_step_rewards.append(to_tensor(eval_rew).float().mean().item())
                    if "success" in eval_infos:
                        eval_step_successes.append(to_tensor(eval_infos["success"]).float().mean().item())
                    if "final_info" in eval_infos:
                        mask = to_tensor(eval_infos["_final_info"]).to(torch.bool)
                        num_episodes += int(mask.sum().item())
                        for k, vals in extract_episode_metrics(eval_infos, mask).items():
                            eval_metrics[k].append(vals)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            if logger is not None:
                mean_rew = float(np.mean(eval_step_rewards))
                logger.add_scalar("eval/mean_reward", mean_rew, global_step)
                print(f"eval_mean_reward={mean_rew:.4f}")
                if eval_step_successes:
                    mean_succ = float(np.mean(eval_step_successes))
                    logger.add_scalar("eval/success_rate", mean_succ, global_step)
                    print(f"eval_success_rate={mean_succ:.4f}")
            for k, v in eval_metrics.items():
                mean = torch.cat(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")

            # Capture and upload videos to wandb every wandb_video_freq steps
            if (args.track and args.capture_video
                    and args.wandb_video_freq > 0
                    and global_step - last_video_log_step >= args.wandb_video_freq):
                print(f"Capturing eval videos at step {global_step}")
                policy_path = record_video_policy(step=global_step)
                if policy_path:
                    wandb.log({"eval/policy_video": wandb.Video(policy_path, fps=30, format="mp4")}, step=global_step)
                det_eval_path = record_video_deterministic_eval(step=global_step)
                if det_eval_path:
                    wandb.log({"eval/deterministic_eval_video": wandb.Video(det_eval_path, fps=30, format="mp4")}, step=global_step)
                last_video_log_step = global_step

            if args.evaluate:
                break
        if (args.save_model and args.checkpoint_freq > 0
                and global_step - last_ckpt_step >= args.checkpoint_freq):
            model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
            os.makedirs(f"runs/{run_name}", exist_ok=True)
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
            last_ckpt_step = global_step
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_time = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_obs = to_tensor(next_obs)
            reward = to_tensor(reward)
            terminations = to_tensor(terminations)
            truncations = to_tensor(truncations)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            # Accumulate unscaled episode returns for gap metrics
            _ep_return_buf += reward.view(-1)
            done_mask = next_done.bool()
            if done_mask.any():
                for env_idx in done_mask.nonzero(as_tuple=False).flatten().tolist():
                    gap_stats.add({"r": _ep_return_buf[env_idx].item(),
                                   "actions": [], "rewards": [], "observations": [],
                                   "seed": args.seed + env_idx})
                _ep_return_buf[done_mask] = 0.0

            if "final_info" in infos:
                done_mask = to_tensor(infos["_final_info"]).to(torch.bool)
                for k, metric_vals in extract_episode_metrics(infos, done_mask).items():
                    if metric_vals.numel() > 0:
                        logger.add_scalar(f"train/{k}", metric_vals.float().mean(), global_step)
                with torch.no_grad():
                    final_obs = to_tensor(select_by_mask(infos["final_observation"], done_mask))
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(final_obs).view(-1)
        rollout_time = time.time() - rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        bc_loss = torch.zeros((), device=device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                bc_loss = torch.zeros((), device=device)
                demo_batch = gap_stats.get_top_batch(args.bc_batch_size, device=device)
                if demo_batch is not None:
                    bc_pred_actions = agent.get_action(demo_batch["observations"], deterministic=True)
                    bc_loss = F.mse_loss(bc_pred_actions, demo_batch["actions"])

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.bc_coef * bc_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/bc_loss", bc_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        logger.add_scalar("train/mean_reward", rewards.float().mean().item(), global_step)
        if iteration % args.plot_freq == 0 and gap_stats._returns:
            gap_stats.plot_gap(logger, global_step)
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            os.makedirs(f"runs/{run_name}", exist_ok=True)
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    envs.close()
    eval_envs.close()
    gap_eval_env.close()
