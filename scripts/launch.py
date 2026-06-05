#!/usr/bin/env python3
"""
Unified launcher for ManiSkill PPO training.

Supports:
  - Slurm clusters (Mila, DRAC) via xm-slurm
  - GCP Compute Engine spot VMs (CPU, self-deleting) via gcloud  [--cluster gcp]
  - GCP Vertex AI (CPU or GPU) via xmanager                      [--cluster vertex]

The Vertex AI path builds from docker/Dockerfile (project root as context) and
pushes to GCR automatically — local code changes are included without a separate
docker build/push step.

Usage:
    # Mila Slurm cluster
    uv run python scripts/launch.py ppo-fixed-layout --cluster mila --user gberseth

    # DRAC / ComputeCanada
    uv run python scripts/launch.py ppo-fixed-layout --cluster drac --user gberseth

    # GCP Compute Engine (spot VM, CPU only, self-deletes on completion)
    uv run python scripts/launch.py ppo-fixed-layout --cluster gcp

    # GCP Vertex AI — CPU job, automatic local code injection
    uv run python scripts/launch.py ppo-training --cluster vertex

    # GCP Vertex AI — T4 GPU job, automatic local code injection
    uv run python scripts/launch.py ppo-training-t4 --cluster vertex

    # Dry-run (print what would happen without launching)
    uv run python scripts/launch.py ppo-training-t4 --cluster vertex --dry-run
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
GCP_REGION    = "northamerica-northeast1"

# Absolute repo root — xm.Dockerfile resolves paths relative to the launcher
# script, so we must pass absolute paths to avoid landing inside scripts/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _RepoDockerfile(xm.Dockerfile):
    """xm.Dockerfile with a fixed lowercase name.

    xm.Dockerfile derives the GCR image name from os.path.basename(path).
    The repo root is 'ManiSkill' which Docker rejects (must be lowercase).
    """
    @property
    def name(self) -> str:
        return "maniskill"

# Add gcloud SDK to PATH if needed
_GCLOUD_SDK = os.path.expanduser("~/google-cloud-sdk/bin")
if _GCLOUD_SDK not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _GCLOUD_SDK + ":" + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

JOBS = {
    # --- CPU jobs ---
    "ppo-smoke": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo_bc.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "unsupervised-robotics",
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
    "ppo-smoke-1m": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo_bc.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "unsupervised-robotics",
            "--use-async-vector-env", "--num-envs", "8", "--no-capture-video",
            "--num-eval-envs", "2", "--num-steps", "50", "--num-eval-steps", "100",
            "--total-timesteps", "1000000", "--seed", "42",
            "--exp-name", "ppo-smoke-1m",
        ],
        cpu=8,
        ram_gib=16,
        machine_type="e2-standard-8",
        slurm_time=dt.timedelta(hours=2),
    ),
    "ppo-fixed-layout": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo_bc.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "unsupervised-robotics",
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
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "unsupervised-robotics",
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
    # --- GPU jobs (Vertex AI / GCP Compute Engine T4) ---
    "ppo-test-t4": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo_upstream.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "unsupervised-robotics",
            "--env_id", "PushText-v1",
            "--num_envs", "1024", "--no-capture-video",
            "--num-eval-envs", "4", "--num-steps", "50", "--num_eval_steps", "100",
            "--total_timesteps", "500000",
            "--exp-name", "gcp-t4-ppo-test",
        ],
        cpu=8,
        ram_gib=16,
        gpu_type="t4",
        gpu_count=1,
        machine_type="n1-standard-8",   # used for gcloud compute engine fallback
    ),
    "ppo-training-t4": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo_upstream.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "unsupervised-robotics",
            "--env_id", "PushText-v1",
            "--num_envs", "2048",
            "--num-eval-envs", "16", "--num-steps", "16", "--num_eval_steps", "100",
            "--update_epochs", "8", "--num_minibatches", "32",
            "--gamma", "0.8", "--total_timesteps", "100000000",
            "--num-videos", "10",
            "--exp-name", "push-text-v1-t4-100M",
        ],
        cpu=8,
        ram_gib=16,
        gpu_type="t4",
        gpu_count=1,
        machine_type="n1-standard-8",   # used for gcloud compute engine fallback
    ),
    "ppo-video-test-l4": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo_upstream.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "unsupervised-robotics",
            "--env_id", "PushText-v1",
            "--num_envs", "256",
            "--num-eval-envs", "4", "--num-steps", "16", "--num_eval_steps", "50",
            "--update_epochs", "4", "--num_minibatches", "8",
            "--gamma", "0.8", "--total_timesteps", "500000",
            "--num-videos", "2",
            "--exp-name", "ppo-video-test-l4",
        ],
        cpu=8,
        ram_gib=16,
        gpu_type="l4_24th",
        gpu_count=1,
        machine_type="n1-standard-8",
    ),
    "ppo-training-l4": dict(
        cmd=[
            "python", "/app/examples/baselines/ppo/ppo_upstream.py",
            "--track", "--wandb-project-name", "ManiSkill", "--wandb-entity", "unsupervised-robotics",
            "--env_id", "PushText-v1",
            "--num_envs", "4096",
            "--num-eval-envs", "16", "--num-steps", "16", "--num_eval_steps", "100",
            "--update_epochs", "8", "--num_minibatches", "32",
            "--gamma", "0.8", "--total_timesteps", "100000000",
            "--num-videos", "10",
            "--exp-name", "push-text-v1-l4-100M",
        ],
        cpu=8,
        ram_gib=30,
        gpu_type="l4_24th",
        gpu_count=1,
        machine_type="n1-standard-8",   # used for gcloud compute engine fallback
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_wandb_key() -> str | None:
    """Read wandb API key from ~/.netrc; never printed to stdout."""
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
# GCP Compute Engine backend (spot VM, self-deleting) — CPU only
# ---------------------------------------------------------------------------

def _make_gcp_startup_script(docker_cmd: str, wandb_key: str | None,
                              use_gpu: bool = False, docker_image: str = DOCKER_IMAGE) -> str:
    wandb_env = (
        f"-e WANDB_API_KEY={wandb_key} -e WANDB_ENTITY=unsupervised-robotics"
        if wandb_key else ""
    )
    gpu_flags = "--gpus all " if use_gpu else ""
    run_cmd = (
        f"docker run {gpu_flags}--rm -w /app "
        f"{wandb_env} "
        f"{docker_image} {docker_cmd}"
    )

    if use_gpu:
        # NGC VMI has NVIDIA drivers, Docker, and nvidia-container-toolkit pre-installed.
        # Pull with retries to handle network drops on large images.
        setup_block = textwrap.dedent(f"""\
            echo "=== Waiting for Docker to be ready ==="
            timeout 120 bash -c 'until docker info &>/dev/null; do sleep 3; done'
            nvidia-smi
            echo "=== Pulling image (with retries) ==="
            for i in 1 2 3; do docker pull {docker_image} && break; echo "Pull attempt $i failed, retrying in 30s..."; sleep 30; done
        """)
    else:
        setup_block = textwrap.dedent("""\
            echo "=== Docker ready ==="
        """)

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

        {setup_block}
        echo "=== Starting job: {docker_image} ==="
        {run_cmd}

        echo "=== Job complete ==="
    """)


def launch_gcp(job_name: str, job_def: dict, wandb_key: str | None,
               zone: str, dry_run: bool = False) -> None:
    use_gpu = bool(job_def.get("gpu_type"))
    docker_cmd = " ".join(shlex.quote(a) for a in job_def["cmd"])
    startup_script = _make_gcp_startup_script(docker_cmd, wandb_key, use_gpu=use_gpu)
    instance_name = f"{job_name}-{dt.datetime.now().strftime('%m%d-%H%M')}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(startup_script)
        script_path = f.name

    if use_gpu:
        image_args = [
            "--image=nvidia-gpu-cloud-vmi-base-2025-9-1-x86-64",
            "--image-project=nvidia-ngc-public",
        ]
        accel_args = [
            f"--accelerator=type={job_def['gpu_type'].replace('t4', 'nvidia-tesla-t4')},count={job_def.get('gpu_count', 1)}",
            "--maintenance-policy=TERMINATE",
        ]
    else:
        image_args = ["--image-family=cos-stable", "--image-project=cos-cloud"]
        accel_args = []

    cmd = [
        "gcloud", "compute", "instances", "create", instance_name,
        f"--zone={zone}",
        f"--project={GCP_PROJECT}",
        f"--machine-type={job_def['machine_type']}",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        *accel_args,
        *image_args,
        "--boot-disk-size=200GB",
        f"--metadata-from-file=startup-script={script_path}",
        "--scopes=cloud-platform",
    ]

    print(f"Launching '{instance_name}' ({job_def['machine_type']}, SPOT) in {zone}")
    if dry_run:
        print("DRY RUN — gcloud command:\n  " + " ".join(cmd))
        masked = startup_script.replace(wandb_key, "***") if wandb_key else startup_script
        print("\nStartup script:\n" + masked)
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
# Vertex AI backend (xmanager) — CPU or GPU, builds from local Dockerfile
# ---------------------------------------------------------------------------

def _patch_xm_vertex() -> None:
    """Patch xmanager's Vertex AI integration for two issues.

    1. Docker push: the Python SDK returns a GCR response that lacks '"Digest":'
       (capital D), causing xmanager's validator to raise even when the push
       succeeded. Replace with subprocess docker push.

    2. L4 accelerator type: xmanager generates 'NVIDIA_TESLA_' + resource.upper()
       for all GPU types, but Vertex AI uses 'NVIDIA_L4' (no TESLA prefix).
       Patch get_machine_spec to handle the L4_24TH resource type correctly.
    """
    from xmanager.cloud import build_image, docker_lib, vertex as vertex_mod
    from google.cloud.aiplatform_v1.types import AcceleratorType
    from xmanager import xm as _xm

    # Fix 1: docker push via subprocess
    def _push_via_subprocess(image: str) -> str:
        repository, tag = image.rsplit(":", 1)
        subprocess.run(["docker", "push", image], check=True)
        subprocess.run(["docker", "push", f"{repository}:latest"], check=True)
        print(f"Your image URI is: {image}")
        return image

    docker_lib.push_docker_image = _push_via_subprocess
    build_image.push = _push_via_subprocess

    # Fix 2: L4 accelerator type — Vertex AI uses NVIDIA_L4 (not NVIDIA_TESLA_L4_24TH)
    # and requires a g2-standard-* machine (not n1-standard-*).
    _GPU_OVERRIDES = {
        _xm.ResourceType.L4_24TH: ("NVIDIA_L4", "g2-standard-8"),
    }
    _original_get_machine_spec = vertex_mod.get_machine_spec

    def _patched_get_machine_spec(job):
        requirements = job.executor.requirements
        for resource, (accel_name, machine_type) in _GPU_OVERRIDES.items():
            if resource in requirements.task_requirements:
                return {
                    "accelerator_type": AcceleratorType[accel_name],
                    "accelerator_count": int(requirements.task_requirements[resource]),
                    "machine_type": machine_type,
                }
        return _original_get_machine_spec(job)

    vertex_mod.get_machine_spec = _patched_get_machine_spec


def launch_vertex(job_name: str, job_def: dict, env_vars: dict,
                  dry_run: bool = False, region: str = GCP_REGION) -> None:
    from xmanager import xm_local
    _patch_xm_vertex()
    os.environ.setdefault("GOOGLE_CLOUD_BUCKET_NAME", "legoassembly-xmanager")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", GCP_PROJECT)
    os.environ["GOOGLE_CLOUD_REGION"] = region

    use_gpu = bool(job_def.get("gpu_type"))

    if use_gpu:
        gpu_type = job_def["gpu_type"]           # e.g. "t4"
        gpu_count = job_def.get("gpu_count", 1)
        requirements = xm.JobRequirements(**{gpu_type: gpu_count})
    else:
        requirements = xm.JobRequirements(
            cpu=job_def["cpu"],
            memory=job_def["ram_gib"] * xm.GiB,
        )

    if dry_run:
        print(f"DRY RUN — Vertex AI job '{job_name}'")
        print(f"  dockerfile:   {os.path.join(_REPO_ROOT, 'docker', 'Dockerfile')} (build context: {_REPO_ROOT})")
        print(f"  requirements: {requirements}")
        print(f"  args:         {job_def['cmd']}")
        print(f"  env_vars:     { {k: '***' if 'KEY' in k else v for k, v in env_vars.items()} }")
        return

    # Builds the image from docker/Dockerfile (project root as context), pushes
    # to GCR, and submits to Vertex AI — local code changes are included
    # automatically without a separate docker build/push step.
    with xm_local.create_experiment(
        experiment_title=f"ManiSkill-{job_name}",
    ) as experiment:
        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=_RepoDockerfile(
                    path=_REPO_ROOT,
                    dockerfile=os.path.join(_REPO_ROOT, "docker", "Dockerfile"),
                ),
                executor_spec=xm_local.Vertex.Spec(),
            )
        ])

        experiment.add(
            xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args=job_def["cmd"],
                env_vars=env_vars,
            )
        )

    print(f"Submitted '{job_name}' to Vertex AI (gpu={use_gpu})")


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
                    time=job_def.get("slurm_time", dt.timedelta(hours=8)),
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
                        choices=["mila", "drac", "cc", "gcp", "vertex"],
                        help=(
                            "Target backend: mila/drac/cc = Slurm, "
                            "gcp = Compute Engine spot VM, "
                            "vertex = Vertex AI with local code injection (default: mila)"
                        ))
    parser.add_argument("--user", default=None,
                        help="SSH username for Slurm clusters")
    parser.add_argument("--partition", default=None,
                        help="Slurm partition override")
    parser.add_argument("--zone", default=GCP_ZONE,
                        help=f"GCP zone for Compute Engine (default: {GCP_ZONE})")
    parser.add_argument("--region", default=GCP_REGION,
                        help=f"GCP region for Vertex AI (default: {GCP_REGION})")
    parser.add_argument("--wandb-key", default=None,
                        help="W&B API key (auto-read from ~/.netrc if omitted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the launch command without executing")
    args, extra_args = parser.parse_known_args()
    args.extra_args = extra_args
    return args


_cli_args = parse_args()


def main(_):
    job_def = dict(JOBS[_cli_args.job])  # shallow copy so we don't mutate the global
    if _cli_args.extra_args:
        job_def["cmd"] = list(job_def["cmd"]) + list(_cli_args.extra_args)

    wandb_key = _cli_args.wandb_key or get_wandb_key()
    if wandb_key is None:
        print("WARNING: No wandb API key found. Run `wandb login` or set WANDB_API_KEY.")

    if _cli_args.cluster == "gcp":
        launch_gcp(_cli_args.job, job_def, wandb_key, _cli_args.zone, _cli_args.dry_run)

    elif _cli_args.cluster == "vertex":
        env_vars = {"WANDB_API_KEY": wandb_key, "WANDB_ENTITY": "unsupervised-robotics"} if wandb_key else {}
        launch_vertex(_cli_args.job, job_def, env_vars, _cli_args.dry_run, _cli_args.region)

    else:
        env_vars = {"WANDB_API_KEY": wandb_key, "WANDB_ENTITY": "unsupervised-robotics"} if wandb_key else {}
        launch_slurm(_cli_args.job, job_def, env_vars,
                     _cli_args.cluster, _cli_args.user, _cli_args.partition)


if __name__ == "__main__":
    # Pass only the script name to absl so it doesn't choke on our argparse flags.
    app.run(main, argv=sys.argv[:1])
