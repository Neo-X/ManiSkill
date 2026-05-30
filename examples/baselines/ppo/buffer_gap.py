


## A special buffer to help track the optimality gap between data generated
from typing import Any, Dict, Generator, List, Optional, Union
try:
    from stable_baselines3.common.buffers import ReplayBuffer
except ImportError:
    ReplayBuffer = object  # BufferGap (SB3-based) unused; only BufferGapV2 is used here
from gymnasium import spaces
import numpy as np
import torch as th

class BufferGap(ReplayBuffer):
    """
    A special buffer to help track the optimality gap between data generated
    by the agent and the optimal policy.
    """

    def __init__(self, 
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        *args, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs,
                         optimize_memory_usage, handle_timeout_termination)
        # super().__init__(*args, **kwargs)
        self.gap = 0.0
        self._gap_percentage = 0.05
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)


    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        ## Check is the info dictionary contains the return key "r" if it does add that to the returns vector
        
        self.returns[self.pos] = np.array([info.get("return") for info in infos])
        super().add(obs, next_obs, action, reward, done, infos)

import heapq, collections, torch
import gymnasium as gym

class BufferGapV2():

    def __init__(self, 
        buffer_size: int,
        top_buffer_percet: float = 0.05,
        policy = False, 
        device = "auto", 
        args = None,
        envs = None,
    ):
        
        self._max_return = -100000
    ## max_returns is a list of the top 10 episodic returns
        self._top_k_plans = []
        self._max_return_buff_size = int(buffer_size * (top_buffer_percet))
        self._top_buffer_percet = top_buffer_percet
        ## returns is a deque of the last 100 episodic returns
        self._returns = collections.deque(maxlen=buffer_size)

        self._policy = policy
        self._device = device

        # env setup
        self._envs = envs
        self._args = args
        self._last_eval = 0

        self._best_traj = []
        self._best_traj_r = []


    def add(self, info):
        return_ = info.get("r")
        self._returns.append(return_)
        plan = list(info.get("actions"))
    
        ## Store the best trajectory
        if return_ > self._max_return:
            self._max_return = return_
            self._best_traj = info["actions"]
            self._best_traj_r = info["rewards"]
     
        # The element to be stored in the heap is a tuple: (return, plan)
        new_heap_item = (return_, plan)

        current_heap_size = len(self._top_k_plans)

        if current_heap_size < self._max_return_buff_size:
            # If the buffer is not full, just push the new item onto the heap
            heapq.heappush(self._top_k_plans, new_heap_item)
        # Check if the new return is greater than the smallest return in the heap
        # The smallest return is the 0th element's 0th index: self._top_k_plans[0][0]
        elif return_ > self._top_k_plans[0][0]:
            # The buffer is full and the new return is better than the smallest return
            # Use heapreplace to pop the smallest (return, plan) and push the new one
            heapq.heappushpop(self._top_k_plans, new_heap_item)


    def plot_gap(self, writer, step: int):
        """
        Plot the gap between the current return and the maximum return
        """
        returns_ = list(self._returns)
        heapq.heapify(returns_)
        _max_returns = [ret for ret, plan in self._top_k_plans] ## Extract the returns from the (return, plan) tuples
        writer.add_scalar("charts/best_trajectory_return", self._max_return, step)
        writer.add_scalar("charts/avg_top_returns_global", np.mean(list(_max_returns)), step)
        writer.add_scalar("charts/avg_top_returns_local", np.mean(heapq.nlargest(max(int(self._top_buffer_percet * len(returns_)), 1), returns_)), step)
        writer.add_scalar("charts/global_optimality_gap", np.mean(list(_max_returns)) - np.mean(returns_), step)
        writer.add_scalar("charts/local_optimality_gap", np.mean(heapq.nlargest(max(int(self._top_buffer_percet * len(returns_)), 1), returns_)) - np.mean(returns_), step)
        
        ## Get performance for the deterministic policy
        if step - self._last_eval > 10000:
            returns = self.eval_deterministic()
            self._last_eval = step
            ## CHeck that the returns is not None
            if returns is not None:
                writer.add_scalar("charts/deterministic_returns", np.mean(returns), step)
            returns = self.eval_deterministic(best=True)
            if returns is not None:
                writer.add_scalar("charts/replay_best_returns", np.mean(returns), step)
            returns = self.eval_stochastic()
            if returns is not None:
                writer.add_scalar("charts/replay_top_k_returns_stochastic", np.mean(returns), step)
            print(f"Stochastic Eval Return: {np.mean(returns)} at step {step}")

 
    def eval_deterministic(self, best=False) -> np.ndarray:
        """
        Evaluate the policy deterministically
        """
        """
        Evaluate the policy deterministically
        """
        obs, _ = self._envs.reset(seed=self._args.seed)
        # q_values = self._policy(torch.Tensor(obs).to(self._device))
        
        max_t = len(self._best_traj) if best==True else self._envs.envs[0].spec.max_episode_steps
        max_t = 100000 if max_t==None else max_t
        samples_ = 1
        returns = []
        for j in range(samples_):
            return_ = 0.0
            # obs, _ = self._envs.reset()
            # returns_ = np.zeros(self._envs.num_envs, dtype=np.float32)
            for t in range(max_t):
                
                actions = [self._best_traj[t] for _ in range(self._envs.num_envs)] if best==True else self._policy.get_action_deterministic(torch.Tensor(obs).to(self._device)).cpu().numpy()

                obs, reward, terminations, truncations, infos = self._envs.step(actions)
                return_ += reward[0]
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            return info['episode']['r']
            returns.append(return_)
        # assert(len(returns) == samples_), f"Returns length is {len(returns)} while expected {samples_}"
        return np.mean(returns)

    def eval_stochastic(self) -> np.ndarray:
        """
        Evaluate the stored action trajectories to get an estimate of their returns
        """
        # obs, _ = self._envs.reset(seed=self._args.seed)
        returns = []
        samples_ = 5
        for j in range(samples_):
            obs, _ = self._envs.reset(seed=self._args.seed)
            returns_ = np.zeros(self._envs.num_envs, dtype=np.float32)
            plan = self._top_k_plans[np.random.randint(0, len(self._top_k_plans))][1] ## Sample a plan from the top k plans
            max_t = len(plan)
            for t in range(max_t):
                
                actions = [plan[t] for _ in range(self._envs.num_envs)] 
                
                obs, reward, terminations, truncations, infos = self._envs.step(actions)
                ## At the end of an episode the info disctionary has other junk in it.
                if "final_info" in infos: ## This final info logic is not very clear
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            returns.extend(info['episode']['r'])
                            break
                    break 
                            # pass
                else:
                    returns_ += infos['reward']
                # Check if the episode is done
                if terminations.any() or truncations.any():
                    dones = np.logical_or(terminations, truncations)
                    returns.extend(np.where(dones, returns_, 0.0))
                    break
                elif t == (max_t - 1): ## If the code makes it to here then no episode finished early
                    returns.extend(returns_)
                    break
        # assert(len(returns) == samples_), f"Returns length is {len(returns)} while expected {samples_}"
        return np.mean(returns)
       

"""Wrapper that tracks the cumulative rewards and episode lengths."""
import time
from collections import deque
from typing import Optional

import numpy as np

import gymnasium as gym


class RecordEpisodeStatisticsV2(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        gym.Wrapper.__init__(self, env)

        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_actions = []
        self.episode_rewards = []

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self.episode_actions = []
        self.episode_rewards = []
        # self.episode_actions = np.zeros(self.num_envs, dtype=np.float32)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_actions.append(action)
        self.episode_rewards.append(rewards)
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones and "episode" in infos:
            infos["episode"]["actions"] = np.where(dones, self.episode_actions, 0) if isinstance(self.env.action_space, gym.spaces.Discrete) else np.where(dones, self.episode_actions, 0)
            infos["episode"]["rewards"] = np.where(dones, self.episode_rewards, 0)
            
            # print( infos["episode"])
            # print("Actions", np.where(dones, self.episode_actions, 0))
        infos['reward']= rewards
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )


"""A synchronous vector environment."""
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from gymnasium import Env
from gymnasium.spaces import Space
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.vector.vector_env import VectorEnv


__all__ = ["SyncVectorEnv"]


class SyncVectorEnvV2(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Example:
        >>> import gymnasium as gym
        >>> env = buffer_gap.SyncVectorEnvV2([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> env.reset(seed=42)
        (array([[-0.14995256,  0.9886932 , -0.12224312],
               [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32), {})
    """

    def __init__(
        self,
        env_fns: Iterable[Callable[[], Env]],
        observation_space: Space = None,
        action_space: Space = None,
        copy: bool = True,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: iterable of callable functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
        """
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            num_envs=len(self.envs),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None
        self._seed = 0

    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        """Sets the seed in all sub-environments.

        Args:
            seed: The seed
        """
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)
        self._seed = seed

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            seed: The reset environment seed
            options: Option information for the environment reset

        Returns:
            The reset observation of the environment and reset information
        """
        if seed is None:
            self._seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            self._seed = [seed + i for i in range(self.num_envs)]
        assert len(self._seed) == self.num_envs

        self._terminateds[:] = False
        self._truncateds[:] = False
        observations = []
        infos = {}
        for i, (env, single_seed) in enumerate(zip(self.envs, self._seed)):
            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options

            observation, info = env.reset(**kwargs)
            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step_async(self, actions):
        """Sets :attr:`_actions` for use by the :meth:`step_wait` by converting the ``actions`` to an iterable version."""
        self._actions = iterate(self.action_space, actions)

    def step_wait(self) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            if self._terminateds[i] or self._truncateds[i]:
                old_observation, old_info = observation, info
                # env.seed(self._seed)
                observation, info = env.reset(seed=self._seed[i])
                # observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )

    def call(self, name, *args, **kwargs) -> tuple:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name: str, values: Union[list, tuple, Any]):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self) -> bool:
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        return True
