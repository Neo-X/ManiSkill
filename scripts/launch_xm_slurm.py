#!/usr/bin/env python3
"""
Unified launcher for ManiSkill PPO training.

Supports:
  - Slurm clusters (Mila, DRAC) via xm-slurm
  - GCP Compute Engine spot VMs (self-deleting) via gcloud

Usage:
    # Mila Slurm cluster
    uv run scripts/launch_xm_slurm.py ppo-fixed-layout --cluster mila --user gberseth

    # DRAC / ComputeCanada
    uv run scripts/launch_xm_slurm.py ppo-fixed-layout --cluster drac --user gberseth

    # GCP Compute Engine (spot VM, self-deletes on completion)
    uv run scripts/launch_xm_slurm.py ppo-fixed-layout --cluster gcp
"""

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap

from absl import app
from xmanager import xm

DOCKER_IMAGE  = "gberseth/maniskill-ppo:latest"
GCP_PROJECT   = "legoassembly"
GCP_ZONE      = "northamerica-northeast1-a"

# Add gcloud SDK to PATH if needed
_GCLOUD_SDK = os.path.expanduser("~/google-cloud-sdk/bin")
if _GCLOUD_SDK not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _GCLOUD_SDK + ":" + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Job definitions (shared across all backends)
# ---------------------------------------------------------------------------

JOBS = {
    "ppo-smoke": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo_bc.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "real-lab",
            "--use-async-vector-env", "--num-envs", "8", "--no-capture-video",
            "--num-eval-envs", "2", "--num-steps", "50", "--num-eval-steps", "100",
            "--total-timesteps", "50000", "--seed", "42",
            "--exp-name", "ppo-smoke",
        ],
        cpu=4,
        ram_gib=8,
        machine_type="e2-standard-4",
        slurm_time=dt.timedelta(minutes=30),
    ),
    "ppo-fixed-layout": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo_bc.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "real-lab",
            "--use-async-vector-env", "--num-envs", "8", "--no-capture-video",
            "--num-eval-envs", "2", "--num-steps", "50", "--num-eval-steps", "200",
            "--total-timesteps", "1000000", "--seed", "42", "--fixed-layout",
            "--exp-name", "push-text-fixed-layout",
        ],
        cpu=8,
        ram_gib=16,
        machine_type="e2-standard-8",
        slurm_time=dt.timedelta(hours=3),
    ),
    "ppo-training": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo_bc.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "real-lab",
            "--use-async-vector-env", "--num-envs", "8", "--no-capture-video",
            "--num-eval-envs", "2", "--num-steps", "50", "--num-eval-steps", "200",
            "--total-timesteps", "10000000", "--seed", "42", "--return-buffer-size", "10000",
            "--exp-name", "push-text-gap-metrics",
        ],
        cpu=8,
        ram_gib=16,
        machine_type="e2-standard-8",
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
# GCP Compute Engine backend (spot VM, self-deleting)
# ---------------------------------------------------------------------------

def _make_startup_script(docker_cmd: str, wandb_key: str | None) -> str:
    wandb_env = f"-e WANDB_API_KEY={wandb_key}" if wandb_key else ""
    run_cmd = (
        f"docker run --rm -w /app "
        f"{wandb_env} "
        f"{DOCKER_IMAGE} {docker_cmd}"
    )
    return textwrap.dedent(f"""\
        #!/bin/bash
        set -eo pipefail

        self_delete() {{
          local TOKEN NAME ZONE PROJECT
          TOKEN=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \\
                  -H "Metadata-Flavor: Google" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
          NAME=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" \\
                 -H "Metadata-Flavor: Google")
          ZONE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" \\
                 -H "Metadata-Flavor: Google" | awk -F/ '{{print $NF}}')
          PROJECT=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/project/project-id" \\
                    -H "Metadata-Flavor: Google")
          echo "Deleting $NAME in $ZONE ..."
          curl -sf -X DELETE \\
            "https://compute.googleapis.com/compute/v1/projects/$PROJECT/zones/$ZONE/instances/$NAME" \\
            -H "Authorization: Bearer $TOKEN" || true
        }}
        trap self_delete EXIT

        echo "=== Starting job ==="
        {run_cmd}

        echo "=== Job complete ==="
    """)


def launch_gcp(job_name: str, job_def: dict, wandb_key: str | None,
               zone: str, dry_run: bool = False) -> None:
    docker_cmd = " ".join(shlex.quote(a) for a in job_def["cmd"])
    startup_script = _make_startup_script(docker_cmd, wandb_key)
    instance_name = f"{job_name}-{dt.datetime.now().strftime('%m%d-%H%M')}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(startup_script)
        script_path = f.name

    cmd = [
        "gcloud", "compute", "instances", "create", instance_name,
        f"--zone={zone}",
        f"--project={GCP_PROJECT}",
        f"--machine-type={job_def['machine_type']}",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        "--image-family=cos-stable",
        "--image-project=cos-cloud",
        "--boot-disk-size=100GB",
        f"--metadata-from-file=startup-script={script_path}",
        "--scopes=cloud-platform",
    ]

    print(f"Launching '{instance_name}' ({job_def['machine_type']}, SPOT) in {zone}")
    if dry_run:
        print("DRY RUN — gcloud command:\n  " + " ".join(cmd))
        print("\nStartup script:\n" + startup_script)
        os.unlink(script_path)
        return

    try:
        result = subprocess.run(cmd, text=True)
    finally:
        os.unlink(script_path)

    if result.returncode != 0:
        print("ERROR: instance creation failed", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"Instance '{instance_name}' launched — will self-delete when done.")


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

        await experiment.add(
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
    parser = argparse.ArgumentParser(description="Launch ManiSkill PPO on Slurm or GCP.")
    parser.add_argument("job", choices=list(JOBS))
    parser.add_argument("--cluster", default="mila",
                        choices=["mila", "drac", "cc", "gcp"],
                        help="Target backend (default: mila)")
    parser.add_argument("--user", default=None,
                        help="SSH username for Slurm clusters")
    parser.add_argument("--partition", default=None,
                        help="Slurm partition override")
    parser.add_argument("--zone", default=GCP_ZONE,
                        help=f"GCP zone for Compute Engine (default: {GCP_ZONE})")
    parser.add_argument("--wandb-key", default=None,
                        help="W&B API key (auto-read from ~/.netrc if omitted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the gcloud command without launching (GCP only)")
    return parser.parse_args()


_cli_args = parse_args()


def main(_):
    job_def = JOBS[_cli_args.job]

    wandb_key = _cli_args.wandb_key or os.environ.get("WANDB_API_KEY") or get_wandb_key()
    if wandb_key is None:
        print("WARNING: No wandb API key found. Run `wandb login` or set WANDB_API_KEY.")

    if _cli_args.cluster == "gcp":
        launch_gcp(_cli_args.job, job_def, wandb_key, _cli_args.zone, _cli_args.dry_run)
    else:
        env_vars = {"WANDB_API_KEY": wandb_key} if wandb_key else {}
        launch_slurm(_cli_args.job, job_def, env_vars,
                     _cli_args.cluster, _cli_args.user, _cli_args.partition)


if __name__ == "__main__":
    # Pass only the script name to absl so it doesn't choke on our argparse flags.
    app.run(main, argv=sys.argv[:1])
