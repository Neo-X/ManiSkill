# Agent Notes for ManiSkill Development

## Python tooling
- Always use `uv run` to invoke Python scripts, not `python` or `python3`

## Remote development on Lightning AI (ssh.lightning.ai)
- SSH host: `s_01kszc7wvb2vcm5dnpq75f8ds7@ssh.lightning.ai`
- Workspace: `/teamspace/studios/this_studio/ManiSkill`
- GPU: NVIDIA L4 (23 GB VRAM), CUDA 13.0
- When making code changes to test on the remote, **use `scp` to copy files** — do NOT commit and push first
- Only commit code that has been tested and confirmed working
- Example scp command:
  ```
  scp examples/baselines/ppo/ppo.py s_01kszc7wvb2vcm5dnpq75f8ds7@ssh.lightning.ai:/teamspace/studios/this_studio/ManiSkill/examples/baselines/ppo/ppo.py
  ```

## Git workflow
- Do not commit files speculatively or mid-experiment
- Only commit changes that have been verified to work (tests pass, training runs without error)
- After confirming a change works, then commit and push to GitHub, then `git pull` on the remote to keep it in sync

## Training
- Remote has GPU: use `--sim-backend physx_gpu` with `ppo.py` for fast GPU-parallelized simulation
- Local machine has no GPU: use `--sim-backend physx_cpu` (default) with small `--num-envs` for smoke tests only
- PushT-v1 known-good settings (from baselines.sh): `--num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 --gamma=0.99 --total_timesteps=50_000_000 --num_eval_steps=100 --num_eval_envs=16`
