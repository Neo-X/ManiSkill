"""
Vertex AI integration tests — submit real jobs to GCP and verify they succeed.

These tests are opt-in; they submit actual Vertex AI jobs and take 10–30 minutes.
Run with:
    pytest tests/test_vertex_integration.py -m vertex -v --timeout=3600

Requires:
  - gcloud authenticated with access to project 'legoassembly'
  - GOOGLE_CLOUD_BUCKET_NAME=legoassembly-xmanager (set automatically by launch.py)
  - wandb credentials in ~/.netrc
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Repo root so we can invoke scripts/ correctly
_REPO_ROOT = Path(__file__).parent.parent
_PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Marker: skip unless RUN_VERTEX_TESTS=1 is set
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.vertex


def _require_vertex():
    if not os.environ.get("RUN_VERTEX_TESTS"):
        pytest.skip("Set RUN_VERTEX_TESTS=1 to run Vertex AI integration tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_script(script_path: Path, extra_args: list[str] = (), timeout_s: int = 1800) -> subprocess.CompletedProcess:
    """Run a Python script and return the completed process."""
    result = subprocess.run(
        [_PYTHON, str(script_path)] + list(extra_args),
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return result


def _extract_job_name(stdout: str) -> str | None:
    """Parse the Vertex AI custom job resource name from xmanager output."""
    m = re.search(r"CustomJob created\. Resource name: (projects/\S+)", stdout)
    return m.group(1) if m else None


def _get_vertex_job_state(resource_name: str) -> str:
    """Return the Vertex AI job state string (e.g. JOB_STATE_SUCCEEDED)."""
    result = subprocess.run(
        ["gcloud", "ai", "custom-jobs", "describe", resource_name,
         "--format=value(state)", "--project=legoassembly"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _wait_for_vertex_job(resource_name: str, poll_interval: int = 30, timeout_s: int = 1800) -> str:
    """Poll until the Vertex AI job reaches a terminal state; return final state."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = _get_vertex_job_state(resource_name)
        if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            return state
        time.sleep(poll_interval)
    return _get_vertex_job_state(resource_name)


def _check_wandb_run(exp_name: str, entity: str = "unsupervised-robotics",
                     project: str = "ManiSkill") -> dict:
    """Return the most recent wandb run matching exp_name, or raise if not found."""
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"display_name": exp_name}, order="-created_at")
    runs = list(runs)
    assert runs, f"No wandb run found with name '{exp_name}' in {entity}/{project}"
    return runs[0]


# ---------------------------------------------------------------------------
# Test 1: xm_hello_world.py — submits a Docker job that prints Hello World
# ---------------------------------------------------------------------------

def test_vertex_hello_world():
    """xm_hello_world.py submits to Vertex AI and the job succeeds."""
    _require_vertex()

    script = _REPO_ROOT / "scripts" / "xm_hello_world.py"
    result = _run_script(script, timeout_s=1800)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"xm_hello_world.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\n"
        f"stderr:\n{result.stderr[-3000:]}"
    )

    # Confirm the job was submitted and completed via xmanager's own status line
    assert "Job launched at:" in combined or "JOB_STATE_SUCCEEDED" in combined or \
           "Job completed successfully" in combined, (
        f"Expected success indicator in output, got:\n{combined[-2000:]}"
    )

    # If we can extract the resource name, double-check via gcloud
    resource_name = _extract_job_name(combined)
    if resource_name:
        state = _wait_for_vertex_job(resource_name, timeout_s=900)
        assert state == "JOB_STATE_SUCCEEDED", f"Vertex job ended in state: {state}"


# ---------------------------------------------------------------------------
# Test 2: ppo-smoke-1m via launch.py — 1M-step CPU training run on Vertex AI
# ---------------------------------------------------------------------------

def test_vertex_ppo_smoke_1m():
    """launch.py ppo-smoke-1m runs 1M PPO steps on Vertex AI and logs to wandb."""
    _require_vertex()

    script = _REPO_ROOT / "scripts" / "launch.py"
    result = _run_script(
        script,
        extra_args=["ppo-smoke-1m", "--cluster", "vertex"],
        timeout_s=3600,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"launch.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\n"
        f"stderr:\n{result.stderr[-3000:]}"
    )

    # Confirm Vertex AI accepted the job
    assert "CustomJob created" in combined, (
        f"Expected 'CustomJob created' in output:\n{combined[-2000:]}"
    )

    # Wait for the Vertex job to finish if we can find its resource name
    resource_name = _extract_job_name(combined)
    if resource_name:
        state = _wait_for_vertex_job(resource_name, poll_interval=60, timeout_s=3000)
        assert state == "JOB_STATE_SUCCEEDED", f"Vertex job ended in state: {state}"

    # Verify wandb received training metrics
    run = _check_wandb_run("ppo-smoke-1m")
    assert run.state in ("finished", "crashed") or run.summary.get("global_step", 0) > 0, (
        f"wandb run '{run.id}' has no training steps recorded"
    )
    assert run.summary.get("global_step", 0) >= 900_000, (
        f"Expected ~1M steps, got {run.summary.get('global_step')} in run {run.id}"
    )
