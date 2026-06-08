# Agent Notes for ManiSkill Development

## On startup
- Always read `progress.md` at the start of each session to understand the current state of the project, active experiments, and known issues before doing any work.
- See `README_UR.md` for how to run experiments locally, on Lightning AI, and on GCP Vertex AI.

## Python tooling
- Always use `uv run` to invoke Python scripts, not `python` or `python3`
- **Exception — Lightning AI remote**: `uv run script.py` treats the file as a PEP 723 standalone script and uses uv's Python cache (not the `.venv`). Use `uv run python script.py` instead — this routes through the project `.venv`.

## Git workflow
- Do not commit files speculatively or mid-experiment
- Only commit changes that have been verified to work (tests pass, training runs without error)
- After confirming a change works, then commit and push to GitHub, then `git pull` on the remote to keep it in sync

## Debugging GCP jobs

When a GCP job doesn't appear in wandb or silently fails, follow these steps:

1. **Check if the instance still exists** — if not, it self-deleted (script failed):
   ```bash
   gcloud compute instances list --filter="name=<instance-name>"
   ```
2. **If instance exists**, read the startup script logs:
   ```bash
   gcloud compute ssh <instance> --zone=<zone> --tunnel-through-iap \
     --command="sudo journalctl -u google-startup-scripts -n 100 --no-pager"
   ```
3. **If instance self-deleted**, relaunch with `--no-spot` so it stays up on script failure, SSH in, and run the docker command manually as root:
   ```bash
   uv run scripts/launch_gcp_job.py ppo-test-t4 --instance-name debug --zone us-central1-a --no-spot
   gcloud compute ssh debug --zone=us-central1-a --tunnel-through-iap
   # On instance:
   sudo docker run --gpus all --rm -w /app gberseth/maniskill-ppo:latest nvidia-smi
   sudo docker run --gpus all --rm -w /app gberseth/maniskill-ppo:latest python /app/examples/baselines/ppo/ppo_upstream.py --total_timesteps 100 --num_envs 4 --no-capture-video
   ```
4. **Clean up** debug instance when done:
   ```bash
   gcloud compute instances delete debug --zone=us-central1-a --quiet
   ```

## GCP GPU instances
- **Always use a Deep Learning VM image** for GPU instances — it has Docker, CUDA, NVIDIA drivers, and nvidia-container-toolkit pre-installed. No manual driver installation needed.
  - Image: `nvidia-gpu-cloud-vmi-base-2025-9-1-x86-64`, project: `nvidia-ngc-public` (NGC VMI — has Docker + CUDA + nvidia-container-toolkit pre-installed)
- GPU instances require `--maintenance-policy=TERMINATE` and `--accelerator=type=<gpu>,count=N`
- Launch via `uv run scripts/launch_gcp_job.py ppo-training-t4` (or `--dry-run` to preview)
- T4 jobs use `ppo_upstream.py` (physx_cuda, GPU-parallelized sim) with `docker run --gpus all`

## Training
- Remote (GCP Vertex AI): use `ppo_upstream.py` with `physx_cuda` for GPU-parallelized simulation; launch via `scripts/launch.py`
- Local machine has no GPU: use `--sim-backend physx_cpu` (default) with small `--num-envs` for smoke tests only
- PushT-v1 known-good settings (from baselines.sh): `--num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 --gamma=0.99 --total_timesteps=50_000_000 --num_eval_steps=100 --num_eval_envs=16`
