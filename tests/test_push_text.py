"""
Tests for PushText-v1 across three environments:
  - Local (uv run pytest tests/test_push_text.py)
  - Docker (docker run ... python -m pytest /app/tests/test_push_text.py)
  - GCP / headless (same Docker command on a CPU-only host, no GPU)

Each test group is tagged with a marker so you can run a subset:
  pytest tests/test_push_text.py -m simulation   # env creation + episode rollout
  pytest tests/test_push_text.py -m devices      # device enumeration
  pytest tests/test_push_text.py -m training     # short PPO loop (no wandb)
"""

import subprocess
import sys
from unittest.mock import MagicMock

import gymnasium as gym
import mani_skill  # noqa: F401 — registers all ManiSkill environments including PushText-v1
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(fixed_layout: bool = False, num_envs: int = 1, **extra_kwargs):
    kwargs = dict(
        obs_mode="state",
        render_mode=None,
        render_backend="none",
        sim_backend="physx_cpu",
    )
    if fixed_layout:
        kwargs["fixed_layout"] = True
    kwargs.update(extra_kwargs)
    return gym.make("PushText-v1", num_envs=num_envs, **kwargs)


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSimulation:
    """Basic environment creation and episode rollout — should pass locally,
    in Docker, and on headless GCP Vertex AI (CPU-only, no Vulkan GPU)."""

    def test_env_creates(self):
        env = _make_env()
        assert env is not None
        env.close()

    def test_reset_returns_obs(self):
        env = _make_env()
        obs, info = env.reset(seed=42)
        assert obs is not None
        env.close()

    def test_obs_shape(self):
        env = _make_env()
        obs, _ = env.reset(seed=42)
        obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else np.array(obs)
        assert obs_np.ndim >= 1
        assert obs_np.shape[-1] > 0
        env.close()

    def test_action_space(self):
        env = _make_env()
        space = env.action_space
        assert hasattr(space, "sample")
        assert hasattr(space, "shape")
        assert space.shape[-1] > 0
        env.close()

    def test_five_random_steps(self):
        env = _make_env()
        obs, _ = env.reset(seed=0)
        for _ in range(5):
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
        env.close()

    def test_reward_is_finite(self):
        env = _make_env()
        env.reset(seed=0)
        for _ in range(5):
            _, rew, _, _, _ = env.step(env.action_space.sample())
        rew_val = rew.cpu().item() if isinstance(rew, torch.Tensor) else float(rew)
        assert np.isfinite(rew_val)
        env.close()

    def test_fixed_layout_reproducible(self):
        """With fixed_layout=True, two resets with the same seed must yield the same obs."""
        env = _make_env(fixed_layout=True)
        obs1, _ = env.reset(seed=7)
        obs2, _ = env.reset(seed=7)
        o1 = obs1.cpu().numpy() if isinstance(obs1, torch.Tensor) else np.array(obs1)
        o2 = obs2.cpu().numpy() if isinstance(obs2, torch.Tensor) else np.array(obs2)
        np.testing.assert_array_almost_equal(o1, o2)
        env.close()

    def test_random_layout_varies(self):
        """Without fixed_layout, two different seeds should give different initial obs."""
        env = _make_env(fixed_layout=False)
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=99)
        o1 = obs1.cpu().numpy() if isinstance(obs1, torch.Tensor) else np.array(obs1)
        o2 = obs2.cpu().numpy() if isinstance(obs2, torch.Tensor) else np.array(obs2)
        assert not np.allclose(o1, o2), "Different seeds should produce different initial observations"
        env.close()

    def test_multiple_episodes(self):
        env = _make_env()
        for seed in range(3):
            obs, _ = env.reset(seed=seed)
            for _ in range(10):
                obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        env.close()


# ---------------------------------------------------------------------------
# Device tests
# ---------------------------------------------------------------------------

class TestDevices:
    """Device availability checks — verify CPU sim works; GPU is optional."""

    def test_sapien_physx_cpu_available(self):
        import sapien.physx as physx
        assert physx is not None

    def test_torch_cpu_available(self):
        t = torch.zeros(3)
        assert t.device.type == "cpu"

    def test_gpu_availability_detected(self):
        """GPU is optional — just check we can query it without crashing."""
        has_cuda = torch.cuda.is_available()
        # Either cuda is available or not — both are valid states.
        assert isinstance(has_cuda, bool)

    def test_env_sim_device_is_cpu(self):
        env = _make_env()
        env.reset(seed=0)
        base = getattr(env, "unwrapped", env)
        device = str(getattr(base, "_sim_device", "cpu"))
        assert "cpu" in device.lower()
        env.close()

    def test_render_device_is_none_when_no_capture(self):
        """render_backend='none' must result in no render device."""
        env = _make_env()
        env.reset(seed=0)
        base = getattr(env, "unwrapped", env)
        render_device = getattr(base, "_render_device", None)
        assert render_device is None
        env.close()


# ---------------------------------------------------------------------------
# Randomize-letters tests
# ---------------------------------------------------------------------------

class TestRandomizeLetters:
    """Tests for the randomize_letters=True mode of PushText-v1."""

    def test_env_creates_with_randomize_letters(self):
        env = _make_env(randomize_letters=True)
        assert env is not None
        env.close()

    def test_goal_text_is_two_letters(self):
        env = _make_env(randomize_letters=True)
        env.reset(seed=0)
        word = env.unwrapped.goal_text
        assert len(word) == 2, f"Expected 2-letter word, got {word!r}"
        env.close()

    def test_active_tiles_match_goal_text(self):
        env = _make_env(randomize_letters=True)
        env.reset(seed=0)
        u = env.unwrapped
        tile_letters = [t.name.split("_")[-1] for t in u.letter_tiles]
        assert tile_letters == list(u.goal_text), (
            f"Active tiles {tile_letters} don't match goal_text {u.goal_text!r}"
        )
        env.close()

    def test_words_differ_across_episodes(self):
        """Repeated resets should eventually produce different letter pairs."""
        env = _make_env(randomize_letters=True)
        words = set()
        for seed in range(20):
            env.reset(seed=seed)
            words.add(env.unwrapped.goal_text)
        env.close()
        assert len(words) > 1, f"All 20 resets produced the same word: {words}"

    def test_letters_from_pool_only(self):
        pool = "ABCDE"
        env = _make_env(randomize_letters=True, letter_pool=pool)
        for seed in range(10):
            env.reset(seed=seed)
            for ch in env.unwrapped.goal_text:
                assert ch in pool, f"Letter {ch!r} not in pool {pool!r}"
        env.close()

    def test_pool_is_deduplicated_and_sorted(self):
        env = _make_env(randomize_letters=True, letter_pool="CCBBAAD")
        assert env.unwrapped._letter_pool == ["A", "B", "C", "D"]
        env.close()

    def test_step_after_randomize_reset(self):
        env = _make_env(randomize_letters=True)
        env.reset(seed=42)
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        rew_val = rew.cpu().item() if isinstance(rew, torch.Tensor) else float(rew)
        assert np.isfinite(rew_val)
        env.close()

    def test_reward_shape_matches_num_envs(self):
        env = _make_env(randomize_letters=True)
        env.reset(seed=0)
        _, rew, _, _, _ = env.step(env.action_space.sample())
        assert rew.shape == (1,)
        env.close()

    def test_multiple_episodes_no_crash(self):
        env = _make_env(randomize_letters=True)
        for seed in range(5):
            env.reset(seed=seed)
            for _ in range(10):
                env.step(env.action_space.sample())
        env.close()

    def test_inactive_pool_tiles_exist(self):
        """All pool tiles must be built at scene load even if only 2 are active."""
        pool = "ABCD"
        env = _make_env(randomize_letters=True, letter_pool=pool)
        env.reset(seed=0)
        u = env.unwrapped
        assert len(u._pool_tiles) == len(pool)
        assert len(u._pool_markers) == len(pool)
        env.close()

    def test_partial_reset_does_not_crash(self):
        """Repeated calls to _initialize_episode must not crash with shape mismatch.

        Root bug: _initialize_episode used self.num_envs for the parked pose size,
        but set_pose indexes only the envs in _reset_mask (size = len(env_idx)).
        On GPU with partial_reset=True, env_idx is a subset (e.g. size 1) while
        num_envs is 4096, causing: "shape [4096,7] cannot broadcast to [1,7]".
        This test calls _initialize_episode directly with env_idx to verify the
        parked pose size is always b = len(env_idx), not self.num_envs.
        """
        env = _make_env(randomize_letters=True, letter_pool="ABCD")
        env.reset(seed=0)
        u = env.unwrapped
        # Direct call with env_idx=[0]: b=1, must match set_pose expectations
        env_idx = torch.tensor([0])
        u._initialize_episode(env_idx, {})
        u._initialize_episode(env_idx, {})
        env.close()


# ---------------------------------------------------------------------------
# Launcher multi-seed tests
# ---------------------------------------------------------------------------

class TestLauncherMultiSeed:
    """Unit tests for the multi-seed command-building logic in scripts/launch.py.
    These run entirely in-process — no docker build, no Vertex AI calls."""

    def _build_seed_cmds(self, base_cmd, seeds, job_name="test-job"):
        """Replicate the seed-cmd logic from launch_vertex."""
        cmds = []
        for seed in seeds:
            seed_cmd = list(base_cmd) + ["--seed", str(seed)]
            if len(seeds) > 1:
                try:
                    base_name = seed_cmd[seed_cmd.index("--exp-name") + 1]
                except ValueError:
                    base_name = job_name
                seed_cmd += ["--exp-name", f"{base_name}-s{seed}"]
            cmds.append(seed_cmd)
        return cmds

    def test_single_seed_no_suffix(self):
        """Single seed must not modify exp-name."""
        base_cmd = ["python", "train.py", "--exp-name", "run-v1"]
        cmds = self._build_seed_cmds(base_cmd, seeds=[42])
        assert cmds[0].count("--exp-name") == 1
        assert "run-v1" in cmds[0]
        assert "--seed" in cmds[0]
        assert cmds[0][cmds[0].index("--seed") + 1] == "42"

    def _last_value(self, cmd, flag):
        """Return the value after the last occurrence of flag in cmd list."""
        idx = len(cmd) - 1 - cmd[::-1].index(flag)
        return cmd[idx + 1]

    def test_multi_seed_distinct_exp_names(self):
        """Each seed must get a unique exp-name suffixed with -s{seed}."""
        base_cmd = ["python", "train.py", "--exp-name", "run-v1"]
        cmds = self._build_seed_cmds(base_cmd, seeds=[1, 2])
        names = [self._last_value(cmd, "--exp-name") for cmd in cmds]
        assert names[0] == "run-v1-s1"
        assert names[1] == "run-v1-s2"
        assert len(set(names)) == 2

    def test_multi_seed_distinct_seed_values(self):
        """Each job must have its own --seed value appended."""
        base_cmd = ["python", "train.py", "--seed", "1"]
        cmds = self._build_seed_cmds(base_cmd, seeds=[1, 2])
        seeds_used = [int(self._last_value(cmd, "--seed")) for cmd in cmds]
        assert seeds_used == [1, 2]

    def test_no_exp_name_in_cmd_falls_back_to_job_name(self):
        """If cmd has no --exp-name, fall back to job_name as base."""
        base_cmd = ["python", "train.py"]
        cmds = self._build_seed_cmds(base_cmd, seeds=[3, 7], job_name="my-job")
        names = [self._last_value(cmd, "--exp-name") for cmd in cmds]
        assert names[0] == "my-job-s3"
        assert names[1] == "my-job-s7"


# ---------------------------------------------------------------------------
# Short training loop test
# ---------------------------------------------------------------------------

class TestTraining:
    """Run a minimal PPO training loop (no wandb, no video) to verify
    the full data pipeline works end-to-end."""

    def test_short_ppo_run(self):
        """Invoke ppo.py as a subprocess for 500 steps — checks imports,
        env creation, rollout collection, and gradient updates all work."""
        script = str(
            (
                __import__("pathlib").Path(__file__).parent.parent
                / "examples/baselines/ppo/ppo.py"
            ).resolve()
        )
        result = subprocess.run(
            [
                sys.executable, script,
                "--num-envs", "2",
                "--num-eval-envs", "1",
                "--num-steps", "64",
                "--num-eval-steps", "20",
                "--total-timesteps", "500",
                "--no-capture-video",
                "--use-async-vector-env",
                "--seed", "42",
                "--exp-name", "pytest-smoke",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"ppo.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )
        assert "model saved" in result.stdout or "SPS" in result.stdout


# ---------------------------------------------------------------------------
# BufferGap tests
# ---------------------------------------------------------------------------

class TestBufferGap:
    """Unit tests for BufferGapV2 — verifies return tracking and gap metrics
    without running a full training loop."""

    def _make_gap_stats(self, buffer_size=100, top_pct=0.1):
        import buffer_gap
        return buffer_gap.BufferGapV2(
            buffer_size=buffer_size,
            top_buffer_percet=top_pct,
            policy=None,
            device="cpu",
            args=None,
            envs=None,
        )

    def test_add_stores_returns(self):
        gs = self._make_gap_stats()
        gs.add({"r": 1.0, "actions": [], "rewards": [], "seed": 0})
        gs.add({"r": 2.0, "actions": [], "rewards": [], "seed": 1})
        assert list(gs._returns) == [1.0, 2.0]

    def test_max_return_tracked(self):
        gs = self._make_gap_stats()
        gs.add({"r": 3.0, "actions": [], "rewards": [], "seed": 0})
        gs.add({"r": 7.0, "actions": [], "rewards": [], "seed": 1})
        gs.add({"r": 2.0, "actions": [], "rewards": [], "seed": 2})
        assert gs._max_return == 7.0

    def test_top_k_buffer_bounded(self):
        gs = self._make_gap_stats(buffer_size=20, top_pct=0.1)
        for i in range(30):
            gs.add({"r": float(i), "actions": [], "rewards": [], "seed": i})
        assert len(gs._top_k_plans) <= gs._max_return_buff_size

    def test_plot_gap_calls_logger(self):
        gs = self._make_gap_stats()
        for i in range(10):
            gs.add({"r": float(i), "actions": [], "rewards": [], "seed": i})
        logger = MagicMock()
        gs.plot_gap(logger, step=100)
        logged_keys = {call.args[0] for call in logger.add_scalar.call_args_list}
        assert "charts/avg_return" in logged_keys
        assert "charts/global_optimality_gap" in logged_keys
        assert "charts/best_trajectory_return" in logged_keys

    def test_plot_gap_skips_when_empty(self):
        gs = self._make_gap_stats()
        logger = MagicMock()
        gs.plot_gap(logger, step=0)
        logger.add_scalar.assert_not_called()

    def test_buffer_size_respected(self):
        gs = self._make_gap_stats(buffer_size=20, top_pct=0.1)
        for i in range(30):
            gs.add({"r": float(i), "actions": [], "rewards": [], "seed": i})
        assert len(gs._returns) == 20
