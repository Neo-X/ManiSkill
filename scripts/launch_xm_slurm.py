#!/usr/bin/env python3
"""
Unified launcher for ManiSkill PPO training via XManager.

Supports:
  - Slurm clusters (Mila, DRAC) via xm-slurm
  - GCP Vertex AI via xmanager xm_local

Usage (must be run from the xm-slurm repo so xm_slurm is importable):

    cd /path/to/xm-slurm

    # Mila Slurm cluster
    uv run --python 3.12 /path/to/ManiSkill/scripts/launch_xm_slurm.py ppo-fixed-layout --cluster mila --user gberseth

    # DRAC / ComputeCanada
    uv run --python 3.12 /path/to/ManiSkill/scripts/launch_xm_slurm.py ppo-fixed-layout --cluster drac --user gberseth

    # GCP Vertex AI
    uv run --python 3.12 /path/to/ManiSkill/scripts/launch_xm_slurm.py ppo-fixed-layout --cluster gcp
"""

import argparse
import datetime as dt
import os
import sys

from absl import app
from xmanager import xm

DOCKER_IMAGE     = "gberseth/maniskill-ppo:latest"
GCP_PROJECT      = "legoassembly"
GCP_LOCATION     = "us-central1"
GCP_BUCKET       = "legoassembly-xmanager"

# ---------------------------------------------------------------------------
# Job definitions (shared across all backends)
# ---------------------------------------------------------------------------

JOBS = {
    "ppo-smoke": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "real-lab",
            "--use-async-vector-env", "--num-envs", "8", "--no-capture-video",
            "--num-eval-envs", "2", "--num-steps", "50", "--num-eval-steps", "100",
            "--total-timesteps", "50000", "--seed", "42",
            "--exp-name", "ppo-smoke",
        ],
        cpu=4,
        ram_gib=8,
        slurm_time=dt.timedelta(minutes=30),
    ),
    "ppo-fixed-layout": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "real-lab",
            "--use-async-vector-env", "--num-envs", "8", "--no-capture-video",
            "--num-eval-envs", "2", "--num-steps", "50", "--num-eval-steps", "200",
            "--total-timesteps", "1000000", "--seed", "42", "--fixed-layout",
            "--exp-name", "push-text-fixed-layout",
        ],
        cpu=8,
        ram_gib=16,
        slurm_time=dt.timedelta(hours=3),
    ),
    "ppo-training": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "real-lab",
            "--use-async-vector-env", "--num-envs", "8", "--no-capture-video",
            "--num-eval-envs", "2", "--num-steps", "50", "--num-eval-steps", "200",
            "--total-timesteps", "10000000", "--seed", "42", "--return-buffer-size", "10000",
            "--exp-name", "push-text-gap-metrics",
        ],
        cpu=8,
        ram_gib=16,
        slurm_time=dt.timedelta(hours=8),
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_wandb_key() -> str | None:
    netrc = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc):
        return None
    with open(netrc) as f:
        tokens = f.read().split()
    try:
        idx = tokens.index("api.wandb.ai")
        return tokens[idx + 4]
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# GCP / Vertex AI backend
# ---------------------------------------------------------------------------

def _patch_vertex_spot() -> None:
    """Monkey-patch CustomJob.submit to request SPOT provisioning."""
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1.types.custom_job import Scheduling

    _orig = aiplatform.CustomJob.submit

    def _spot_submit(self, **kwargs):
        self._gca_resource.job_spec.scheduling.strategy = Scheduling.Strategy.SPOT
        return _orig(self, **kwargs)

    aiplatform.CustomJob.submit = _spot_submit


def launch_gcp(job_name: str, job_def: dict, env_vars: dict) -> None:
    import xmanager.xm_local as xm_local

    os.environ.setdefault("GOOGLE_CLOUD_BUCKET_NAME", GCP_BUCKET)
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", GCP_PROJECT)
    _patch_vertex_spot()

    with xm_local.create_experiment(f"ManiSkill-{job_name}") as experiment:
        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=xm.Container(image_path=DOCKER_IMAGE),
                executor_spec=xm_local.Vertex.Spec(),
                args=xm.SequentialArgs.from_collection(job_def["cmd"]),
                env_vars=env_vars,
            )
        ])

        experiment.add(
            xm.Job(
                executable=executable,
                executor=xm_local.Vertex(
                    requirements=xm.JobRequirements(
                        cpu=job_def["cpu"],
                        ram=job_def["ram_gib"] * xm.GiB,
                    )
                ),
            )
        )
    print(f"Submitted '{job_name}' to GCP Vertex AI (project={GCP_PROJECT})")


# ---------------------------------------------------------------------------
# Slurm backend (Mila / DRAC)
# ---------------------------------------------------------------------------

@xm.run_in_asyncio_loop
async def launch_slurm(job_name: str, job_def: dict, env_vars: dict,
                        cluster_name: str, user: str | None, partition: str | None) -> None:
    import xm_slurm
    import xm_slurm.contrib.clusters

    if cluster_name == "mila":
        cluster = xm_slurm.contrib.clusters.mila(user=user, partition=partition)
    elif cluster_name in ("drac", "cc"):
        cluster = xm_slurm.contrib.clusters.drac(user=user, partition=partition)
    else:
        raise ValueError(f"Unknown cluster: {cluster_name!r}")

    async with xm_slurm.create_experiment(f"ManiSkill-{job_name}") as experiment:
        [executable] = experiment.package([
            xm_slurm.docker_image(
                image=DOCKER_IMAGE,
                args=job_def["cmd"],
                env_vars=env_vars,
            )
        ])

        wu = await experiment.add(
            xm.Job(
                executable=executable,
                executor=xm_slurm.Slurm(
                    requirements=xm_slurm.JobRequirements(
                        CPU=job_def["cpu"],
                        RAM=job_def["ram_gib"] * xm.GiB,
                        cluster=cluster,
                    ),
                    time=job_def["slurm_time"],
                ),
            )
        )

    print(f"Submitted '{job_name}' to {cluster_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Launch ManiSkill PPO via XManager.")
    parser.add_argument("job", choices=list(JOBS))
    parser.add_argument("--cluster", default="mila",
                        choices=["mila", "drac", "cc", "gcp"],
                        help="Target backend (default: mila)")
    parser.add_argument("--user", default=None,
                        help="SSH username for Slurm clusters")
    parser.add_argument("--partition", default=None,
                        help="Slurm partition override")
    parser.add_argument("--wandb-key", default=None,
                        help="W&B API key (auto-read from ~/.netrc if omitted)")
    return parser.parse_args()


_cli_args = parse_args()


def main(_):
    job_def = JOBS[_cli_args.job]

    wandb_key = _cli_args.wandb_key or os.environ.get("WANDB_API_KEY") or get_wandb_key()
    if wandb_key is None:
        print("WARNING: No wandb API key found. Run `wandb login` or set WANDB_API_KEY.")
    env_vars = {"WANDB_API_KEY": wandb_key} if wandb_key else {}

    if _cli_args.cluster == "gcp":
        launch_gcp(_cli_args.job, job_def, env_vars)
    else:
        launch_slurm(_cli_args.job, job_def, env_vars,
                     _cli_args.cluster, _cli_args.user, _cli_args.partition)


if __name__ == "__main__":
    # Pass only the script name to absl so it doesn't choke on our argparse flags.
    app.run(main, argv=sys.argv[:1])
