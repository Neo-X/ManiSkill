"""
Launch a Docker job on a GCP spot instance that self-deletes when done.

Usage:
    python scripts/launch_gcp_job.py --help
    python scripts/launch_gcp_job.py hello-world
    python scripts/launch_gcp_job.py ppo-training
"""

import argparse
import subprocess
import sys
import textwrap


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

JOBS = {
    "hello-world": dict(
        image="hello-world",
        cmd="",
        machine_type="e2-micro",
    ),
    "ppo-training": dict(
        image="gberseth/maniskill-ppo:latest",
        cmd=(
            "python /examples/baselines/ppo/ppo.py"
            " --use-async-vector-env --num-envs 32 --no-capture-video"
            " --num-eval-envs 4 --num-steps 50"
            " --num-eval-steps 200 --total-timesteps 10000000"
            " --eval-freq 125"
            " --exp-name push-text-state-ppo"
        ),
        machine_type="e2-standard-8",
    ),
}


# ---------------------------------------------------------------------------
# Startup script template
# ---------------------------------------------------------------------------

def make_startup_script(docker_image: str, docker_cmd: str, gcs_bucket: str | None) -> str:
    """
    Returns a bash startup script that:
      1. Runs a docker container.
      2. Optionally syncs /app/runs to a GCS bucket when done.
      3. Self-deletes the instance on exit (success or failure).
    """
    sync_block = ""
    if gcs_bucket:
        sync_block = f"gsutil -m rsync -r /app/runs gs://{gcs_bucket}/runs || true"

    run_cmd = f"docker run --rm {docker_image}"
    if docker_cmd:
        run_cmd += f" {docker_cmd}"

    return textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail

        # Self-delete this instance on exit (success, failure, or preemption)
        self_delete() {{
          local NAME ZONE
          NAME=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" \\
                 -H "Metadata-Flavor: Google")
          ZONE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" \\
                 -H "Metadata-Flavor: Google" | awk -F/ '{{print $NF}}')
          echo "Job finished. Deleting instance $NAME in $ZONE ..."
          gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || true
        }}
        trap self_delete EXIT

        echo "=== Starting job: {docker_image} ==="
        {run_cmd}

        {sync_block}
        echo "=== Job complete ==="
    """)


# ---------------------------------------------------------------------------
# GCP launch
# ---------------------------------------------------------------------------

def launch(
    job_name: str,
    instance_name: str,
    zone: str = "us-central1-a",
    gcs_bucket: str | None = None,
    dry_run: bool = False,
) -> None:
    job = JOBS[job_name]
    startup_script = make_startup_script(job["image"], job["cmd"], gcs_bucket)

    cmd = [
        "gcloud", "compute", "instances", "create", instance_name,
        f"--zone={zone}",
        f"--machine-type={job['machine_type']}",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        "--image-family=cos-stable",
        "--image-project=cos-cloud",
        f"--metadata=startup-script={startup_script}",
        "--scopes=cloud-platform",
    ]

    print(f"Launching instance '{instance_name}' for job '{job_name}' in {zone}")
    if dry_run:
        print("DRY RUN — command that would be run:")
        print(" ".join(cmd))
        print("\nStartup script:")
        print(startup_script)
        return

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print("ERROR: instance creation failed", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"Instance '{instance_name}' launched. It will self-delete when the job finishes.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Docker job on a GCP spot instance.")
    parser.add_argument("job", choices=list(JOBS), help="Which job to run")
    parser.add_argument("--instance-name", default=None,
                        help="GCP instance name (default: <job>-instance)")
    parser.add_argument("--zone", default="us-central1-a", help="GCP zone")
    parser.add_argument("--gcs-bucket", default=None,
                        help="GCS bucket to sync /app/runs to when training finishes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the gcloud command without running it")
    args = parser.parse_args()

    instance_name = args.instance_name or f"{args.job}-instance"
    launch(
        job_name=args.job,
        instance_name=instance_name,
        zone=args.zone,
        gcs_bucket=args.gcs_bucket,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
