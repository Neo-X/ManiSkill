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

import gymnasium as gym
import mani_skill  # noqa: F401 — registers all ManiSkill environments including PushText-v1
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(fixed_layout: bool = False, num_envs: int = 1):
    kwargs = dict(
        obs_mode="state",
        render_mode=None,
        render_backend="none",
        sim_backend="physx_cpu",
    )
    if fixed_layout:
        kwargs["fixed_layout"] = True
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
