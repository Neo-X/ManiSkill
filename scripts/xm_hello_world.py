"""Minimal XManager Dockerfile example — mirrors examples/dockerfile/launcher.py.

Builds docker/Dockerfile (project root as context), pushes to GCR,
and runs 'python -c print("Hello World")' on Vertex AI.

Usage:
    uv run python scripts/xm_hello_world.py
"""

import os
import subprocess
from absl import app
from xmanager import xm
from xmanager import xm_local
from google.cloud.aiplatform_v1.types import AcceleratorType

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("GOOGLE_CLOUD_BUCKET_NAME", "legoassembly-xmanager")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "legoassembly")


# ---------------------------------------------------------------------------
# Patches (same as launch.py)
# ---------------------------------------------------------------------------

def _patch_xm() -> None:
    from xmanager.cloud import build_image, docker_lib

    def _push_via_subprocess(image: str) -> str:
        repository, tag = image.rsplit(":", 1)
        subprocess.run(["docker", "push", image], check=True)
        subprocess.run(["docker", "push", f"{repository}:latest"], check=True)
        print(f"Your image URI is: {image}")
        return image

    docker_lib.push_docker_image = _push_via_subprocess
    build_image.push = _push_via_subprocess


class _RepoDockerfile(xm.Dockerfile):
    """xm.Dockerfile with fixed lowercase name (repo dir is 'ManiSkill')."""
    @property
    def name(self) -> str:
        return "maniskill"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv) -> None:
    del argv
    _patch_xm()

    with xm_local.create_experiment(experiment_title="hello-world-dockerfile") as experiment:
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
                executor=xm_local.Vertex(
                    requirements=xm.JobRequirements(cpu=2, memory=4 * xm.GiB),
                ),
                args=["python", "-c", "print('Hello World')"],
            )
        )


if __name__ == "__main__":
    app.run(main)
