#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Start/stop a persistent GCP GPU dev VM and configure VS Code Remote SSH.

The VM uses a T4 GPU, has Docker + CUDA pre-installed (NGC VMI), clones the
ManiSkill repo, and auto-shuts down after 20 minutes of CPU inactivity.

Usage:
    uv run scripts/dev_vm.py start          # create VM, configure SSH, print connect instructions
    uv run scripts/dev_vm.py stop           # delete VM
    uv run scripts/dev_vm.py status         # show VM status and SSH host name
"""

import argparse
import os
import subprocess
import sys
import textwrap
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INSTANCE_NAME = "dev-gpu"
ZONE          = "us-central1-a"
PROJECT       = "legoassembly"
MACHINE_TYPE  = "n1-standard-8"
GPU_TYPE      = "nvidia-tesla-t4"
REPO_URL      = "https://github.com/Neo-X/ManiSkill.git"
DOCKER_IMAGE  = "gberseth/maniskill-ppo:latest"
INACTIVITY_MINUTES = 20

_GCLOUD_SDK = os.path.expanduser("~/google-cloud-sdk/bin")
if _GCLOUD_SDK not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _GCLOUD_SDK + ":" + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# Startup script
# ---------------------------------------------------------------------------

STARTUP_SCRIPT = textwrap.dedent(f"""\
    #!/bin/bash
    set -eo pipefail

    # Auto-shutdown after {INACTIVITY_MINUTES} minutes of CPU inactivity (<10% usage).
    # Checks every minute; resets counter on activity.
    (
      IDLE=0
      while true; do
        sleep 60
        CPU=$(top -bn1 | grep "Cpu(s)" | awk '{{print $2+$4}}' | cut -d. -f1)
        if [ "${{CPU:-100}}" -lt 10 ]; then
          IDLE=$((IDLE+1))
        else
          IDLE=0
        fi
        if [ "$IDLE" -ge {INACTIVITY_MINUTES} ]; then
          echo "Auto-shutdown: ${{IDLE}} consecutive minutes below 10% CPU"
          TOKEN=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \\
                  -H "Metadata-Flavor: Google" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
          NAME=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" \\
                 -H "Metadata-Flavor: Google")
          ZONE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" \\
                 -H "Metadata-Flavor: Google" | awk -F/ '{{print $NF}}')
          PROJ=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/project/project-id" \\
                 -H "Metadata-Flavor: Google")
          curl -sf -X DELETE \\
            "https://compute.googleapis.com/compute/v1/projects/$PROJ/zones/$ZONE/instances/$NAME" \\
            -H "Authorization: Bearer $TOKEN" || true
          break
        fi
      done
    ) &

    echo "=== Waiting for Docker ==="
    timeout 120 bash -c 'until docker info &>/dev/null; do sleep 3; done'
    nvidia-smi

    echo "=== Pulling Docker image ==="
    for i in 1 2 3; do docker pull {DOCKER_IMAGE} && break; sleep 30; done

    echo "=== Cloning repo ==="
    git clone {REPO_URL} /root/ManiSkill || true

    echo "=== Dev VM ready ==="
""")

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _gcloud(*args, check=True, capture=False):
    cmd = ["gcloud", *args]
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.stdout.strip()
    return subprocess.run(cmd, check=check)


def start():
    # Check if already running
    existing = _gcloud(
        "compute", "instances", "list",
        f"--filter=name={INSTANCE_NAME}",
        "--format=value(status)",
        f"--project={PROJECT}",
        capture=True,
    )
    if existing:
        print(f"Instance '{INSTANCE_NAME}' already exists (status: {existing}).")
    else:
        print(f"Creating '{INSTANCE_NAME}' ({MACHINE_TYPE} + T4) in {ZONE}...")
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(STARTUP_SCRIPT)
            script_path = f.name
        try:
            _gcloud(
                "compute", "instances", "create", INSTANCE_NAME,
                f"--zone={ZONE}",
                f"--project={PROJECT}",
                f"--machine-type={MACHINE_TYPE}",
                f"--accelerator=type={GPU_TYPE},count=1",
                "--maintenance-policy=TERMINATE",
                "--image=nvidia-gpu-cloud-vmi-base-2025-9-1-x86-64",
                "--image-project=nvidia-ngc-public",
                "--boot-disk-size=200GB",
                f"--metadata-from-file=startup-script={script_path}",
                "--scopes=cloud-platform",
            )
        finally:
            os.unlink(script_path)

    # Wait for SSH
    print("Waiting for SSH...")
    for _ in range(24):
        r = subprocess.run(
            ["gcloud", "compute", "ssh", INSTANCE_NAME,
             f"--zone={ZONE}", f"--project={PROJECT}",
             "--tunnel-through-iap", "--command=echo ready"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            break
        time.sleep(10)
    else:
        print("WARNING: SSH not available yet — try connecting manually in a moment.")

    # Update ~/.ssh/config with gcloud-generated entries
    print("Updating ~/.ssh/config...")
    _gcloud("compute", "config-ssh", f"--project={PROJECT}", check=False)

    ssh_host = f"{INSTANCE_NAME}.{ZONE}.{PROJECT}"
    remote_dir = "/root/ManiSkill"

    print(f"Opening VS Code connected to {ssh_host}:{remote_dir} ...")
    result = subprocess.run(
        ["code", "--remote", f"ssh-remote+{ssh_host}", remote_dir],
    )
    if result.returncode != 0:
        print(f"\nCould not launch VS Code automatically. Connect manually:")
        print(f"  code --remote ssh-remote+{ssh_host} {remote_dir}")
        print(f"\nOr SSH directly:")
        print(f"  gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --project={PROJECT} --tunnel-through-iap")
    print(f"\nAuto-shutdown: VM self-deletes after {INACTIVITY_MINUTES} minutes of CPU inactivity.")
    print(f"Stop manually: uv run scripts/dev_vm.py stop")


def stop():
    print(f"Deleting '{INSTANCE_NAME}'...")
    _gcloud(
        "compute", "instances", "delete", INSTANCE_NAME,
        f"--zone={ZONE}", f"--project={PROJECT}", "--quiet",
        check=False,
    )
    print("Done.")


def status():
    out = _gcloud(
        "compute", "instances", "list",
        f"--filter=name={INSTANCE_NAME}",
        "--format=table(name,zone,status,networkInterfaces[0].accessConfigs[0].natIP)",
        f"--project={PROJECT}",
        capture=True,
    )
    if out:
        print(out)
        print(f"\nSSH host for VS Code: {INSTANCE_NAME}.{ZONE}.{PROJECT}")
    else:
        print(f"No instance named '{INSTANCE_NAME}' found.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Manage the ManiSkill GPU dev VM.")
    parser.add_argument("command", choices=["start", "stop", "status"])
    args = parser.parse_args()
    {"start": start, "stop": stop, "status": status}[args.command]()


if __name__ == "__main__":
    main()
