#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Launch a Docker job on a GCP spot instance that self-deletes when done.

Usage:
    uv run scripts/launch_gcp_job.py --help
    uv run scripts/launch_gcp_job.py hello-world
    uv run scripts/launch_gcp_job.py ppo-test      # 5K steps, verifies wandb
    uv run scripts/launch_gcp_job.py ppo-training  # full 10M step run
    uv run scripts/launch_gcp_job.py ppo-training --gcs-bucket my-bucket
"""

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap

# Add gcloud SDK to PATH if not already present
_GCLOUD_SDK = os.path.expanduser("~/google-cloud-sdk/bin")
if _GCLOUD_SDK not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _GCLOUD_SDK + ":" + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

REPO_URL = "https://github.com/Neo-X/ManiSkill.git"

JOBS = {
    "hello-world": dict(
        image="hello-world",
        cmd="",
        machine_type="e2-micro",
        use_wandb=False,
    ),
    "ppo-test": dict(
        image="gberseth/maniskill-ppo:latest",
        cmd=(
            "python /app/examples/baselines/ppo/ppo.py"
            " --track --wandb-project-name ManiSkill --wandb-entity real-lab"
            " --use-async-vector-env --num-envs 32 --no-capture-video"
            " --num-eval-envs 4 --num-steps 50 --num-eval-steps 100"
            " --total-timesteps 100000 --eval-freq 10"
            " --exp-name gcp-ppo-test"
        ),
        machine_type="e2-standard-32",
        use_wandb=True,
    ),
    "ppo-training": dict(
        image="gberseth/maniskill-ppo:latest",
        cmd=(
            "python /app/examples/baselines/ppo/ppo.py"
            " --track --wandb-project-name ManiSkill --wandb-entity real-lab"
            " --use-async-vector-env --num-envs 32 --no-capture-video"
            " --num-eval-envs 4 --num-steps 50 --num-eval-steps 200"
            " --total-timesteps 10000000 --eval-freq 125"
            " --seed 42 --return-buffer-size 10000"
            " --exp-name push-text-gap-metrics-10M"
        ),
        machine_type="e2-standard-32",
        use_wandb=True,
    ),
    "ppo-fixed-layout": dict(
        image="gberseth/maniskill-ppo:latest",
        cmd=(
            "python /app/examples/baselines/ppo/ppo.py"
            " --track --wandb-project-name ManiSkill --wandb-entity real-lab"
            " --use-async-vector-env --num-envs 32 --no-capture-video"
            " --num-eval-envs 4 --num-steps 50 --num-eval-steps 200"
            " --total-timesteps 1000000 --eval-freq 25"
            " --seed 42 --return-buffer-size 10000 --fixed-layout"
            " --exp-name push-text-fixed-layout-gcp"
        ),
        machine_type="e2-standard-32",
        use_wandb=True,
    ),
}


# ---------------------------------------------------------------------------
# Startup script template
# ---------------------------------------------------------------------------

def make_startup_script(
    docker_image: str,
    docker_cmd: str,
    gcs_bucket: str | None,
    wandb_key: str | None = None,
    install_docker: bool = True,
    runs_dir: str = "/var/runs",
    capture_video: bool = False,
) -> str:
    sync_block = ""
    if gcs_bucket:
        sync_block = f"gsutil -m rsync -r {runs_dir} gs://{gcs_bucket}/runs || true"

    wandb_env = f"-e WANDB_API_KEY={wandb_key}" if wandb_key else ""

    # xvfb-run provides a virtual display required by sapien_cpu renderer
    xvfb_prefix = "xvfb-run -a " if capture_video else ""

    run_cmd = (
        f"docker run --rm -w /app -v {runs_dir}:/app/runs "
        f"{wandb_env} "
        f"{docker_image} {xvfb_prefix}{docker_cmd}"
    )

    docker_install_block = textwrap.dedent("""\
        echo "=== Installing Docker ==="
        curl -fsSL https://get.docker.com | sh
        systemctl enable --now docker
    """) if install_docker else ""

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

        {docker_install_block}
        echo "=== Docker ready ===" && docker info

        mkdir -p {runs_dir}
        echo "=== Starting job: {docker_image} ==="
        {run_cmd}

        {sync_block}
        echo "=== Job complete ==="
    """)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_wandb_key() -> str | None:
    """Read wandb API key from ~/.netrc (written by `wandb login`)."""
    netrc = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc):
        return None
    with open(netrc) as f:
        tokens = f.read().split()
    try:
        idx = tokens.index("api.wandb.ai")
        # format: machine api.wandb.ai login user password <key>
        return tokens[idx + 4]
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# GCP launch
# ---------------------------------------------------------------------------

def launch(
    job_name: str,
    instance_name: str,
    zone: str = "us-central1-a",
    gcs_bucket: str | None = None,
    wandb_key: str | None = None,
    spot: bool = True,
    use_cos: bool = True,
    dry_run: bool = False,
    capture_video: bool = False,
) -> None:
    job = JOBS[job_name]

    if job.get("use_wandb") and wandb_key is None:
        wandb_key = get_wandb_key()
        if wandb_key:
            print("Using wandb key from ~/.netrc")
        else:
            print("WARNING: wandb key not found — pass --wandb-key or run `wandb login` first.")

    # Strip --no-capture-video if video is requested, otherwise ensure it's present
    docker_cmd = job["cmd"]
    if capture_video:
        docker_cmd = docker_cmd.replace(" --no-capture-video", "")
    elif "--no-capture-video" not in docker_cmd:
        docker_cmd += " --no-capture-video"

    startup_script = make_startup_script(
        docker_image=job["image"],
        docker_cmd=docker_cmd,
        gcs_bucket=gcs_bucket,
        wandb_key=wandb_key if job.get("use_wandb") else None,
        install_docker=not use_cos,
        runs_dir="/var/runs",
        capture_video=capture_video,
    )

    # Write startup script to a temp file to avoid gcloud metadata parsing issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(startup_script)
        script_path = f.name

    if use_cos:
        image_args = ["--image-family=cos-stable", "--image-project=cos-cloud"]
    else:
        image_args = ["--image-family=debian-12", "--image-project=debian-cloud"]

    provisioning = ["--provisioning-model=SPOT", "--instance-termination-action=DELETE"] if spot else []
    cmd = [
        "gcloud", "compute", "instances", "create", instance_name,
        f"--zone={zone}",
        f"--machine-type={job['machine_type']}",
        *provisioning,
        *image_args,
        "--boot-disk-size=200GB",
        f"--metadata-from-file=startup-script={script_path}",
        "--scopes=cloud-platform",
    ]

    print(f"Launching '{instance_name}' for job '{job_name}' in {zone}")
    if dry_run:
        print("DRY RUN — gcloud command:")
        print(" ".join(cmd))
        print("\nStartup script:")
        print(startup_script)
        return

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
    finally:
        os.unlink(script_path)
    if result.returncode != 0:
        print("ERROR: instance creation failed", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"Instance '{instance_name}' launched — will self-delete when done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Docker job on a GCP spot instance.")
    parser.add_argument("job", choices=list(JOBS), help="Which job to run")
    parser.add_argument("--instance-name", default=None,
                        help="GCP instance name (default: <job>-instance)")
    parser.add_argument("--zone", default="northamerica-northeast1-a", help="GCP zone")
    parser.add_argument("--gcs-bucket", default=None,
                        help="GCS bucket to sync runs/ to on completion")
    parser.add_argument("--wandb-key", default=None,
                        help="W&B API key (auto-read from ~/.netrc if omitted)")
    parser.add_argument("--no-spot", action="store_true",
                        help="Use on-demand pricing instead of spot (more reliable, more expensive)")
    parser.add_argument("--no-cos", action="store_true",
                        help="Use Debian instead of Container-Optimized OS (installs Docker at boot)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the gcloud command without launching")
    parser.add_argument("--capture-video", action="store_true",
                        help="Record eval videos via xvfb + sapien_cpu renderer (rebuilds without --no-capture-video)")
    args = parser.parse_args()

    instance_name = args.instance_name or f"{args.job}-instance"
    launch(
        job_name=args.job,
        instance_name=instance_name,
        zone=args.zone,
        gcs_bucket=args.gcs_bucket,
        wandb_key=args.wandb_key,
        spot=not args.no_spot,
        use_cos=not args.no_cos,
        dry_run=args.dry_run,
        capture_video=args.capture_video,
    )


if __name__ == "__main__":
    main()
