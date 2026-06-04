# PushText Environment — Progress

## Goal

Build a ManiSkill task environment called **PushText-v1** for robotic letter-spelling research.

The robot must pick up and place fridge-magnet-style letter tiles on a white table to spell a target word. This serves as a simpler proxy for LEGO assembly — structured manipulation toward a language-specified goal.

---

## Task Design

**Object geometry:** Real letter-shaped OBJ meshes (40mm tall, 12mm thick, 2× scale for visibility), extracted from `assets/objects/Alphabet.stl` and `assets/objects/numbers.stl`. Ghost (semi-transparent) target markers show target positions and orientation.

**Robot:** `panda_wristcam` (full gripper — can pick, place, and push).

**Goal specification:** A text string (e.g. `"AT"`, `"CAT"`) passed at env construction. Exposed via `get_language_instruction()` for VLA consumption.

**Target layout:** Letters arranged left-to-right in a centered row on the table, spaced 0.130 m apart (doubled with 2× scale). Target orientation is 90° CW around Z so letters read correctly in the top-down camera view.

**Randomization:** Each letter tile spawns at a random (x, y, z-rotation) each episode, with non-overlapping placement via `UniformPlacementSampler`.

**Success:** Every letter within 0.025 m of its target, stationary, and released by gripper.

**Reward:** Staged per-letter reward (reach → grasp → place → release), summed across letters. Mirrors StackCube-v1 structure. Max raw = n_letters × 8 + 2 (OCR bonus), normalized to [0, 1].

**Top-down camera (`base_camera`):** 256×256, eye=[-0.1, 0, 1.1], up=(1,0,0). Robot arm is hidden before rendering via `get_table_view()` — used for the OCR reward bonus and debugging. OCR (EasyOCR, 2× upscale) detects the goal word every 20 steps and adds a +2 reward bonus.

**State observation (43-dim):** `qpos(9) + qvel(9) + tcp_pose(7) + letter_rot_to_target(n×3) + tcp_to_letters(n×3) + letters_to_targets(n×3)`. Rotation error is expressed as axis-angle (axis defaults to world-Z at zero error).

---

## File Locations

| File | Purpose |
|------|---------|
| `mani_skill/envs/tasks/tabletop/push_text.py` | Main environment class (`PushText-v1`) |
| `mani_skill/envs/tasks/tabletop/__init__.py` | Registers `PushTextEnv` |
| `mani_skill/assets/objects/letters/` | Per-letter OBJ + STL meshes (A–Z, 0–9) |
| `scripts/extract_letters.py` | Extracts A–Z from `Alphabet.stl` |
| `examples/baselines/ppo/ppo.py` | PPO training script (default env changed to `PushText-v1`) |

---

## Commands

**Render a random-action episode:**
```bash
.venv/bin/python -m mani_skill.examples.demo_random_action \
  -e PushText-v1 --render-mode rgb_array -b cpu --record-dir /tmp/pushtext_out
```

**Train with PPO (CPU, 4 parallel envs):**
```bash
.venv/bin/python examples/baselines/ppo/ppo.py \
  --use-async-vector-env \
  --num-envs 8 \
  --num-eval-envs 2 \
  --num-steps 50 \
  --num-eval-steps 200 \
  --total-timesteps 2000000 \
  --exp-name push-text-ppo
```

**Smoke-test (quick 50K steps):**
```bash
.venv/bin/python examples/baselines/ppo/ppo.py \
  --use-async-vector-env \
  --num-envs 4 \
  --num-eval-envs 2 \
  --num-steps 50 \
  --num-eval-steps 200 \
  --total-timesteps 50000 \
  --exp-name push-text-smoke \
  --no-capture-video
```

---

## Status

- [x] Write `push_text.py` — based on StackCube-v1 (panda_wristcam, full gripper)
- [x] Register in `tabletop/__init__.py`
- [x] Import smoke-test passes
- [x] Render episode — A and T letter meshes visible, overhead camera, video at `/tmp/pushtext_at6/0.mp4`
- [x] Extract A–Z letter meshes from `Alphabet.stl` → `assets/objects/letters/` (OBJ + STL, 40mm tall, 12mm thick)
- [x] Extract 0–9 digit meshes from `numbers.stl` → same folder
- [x] PPO training runs on CPU with async multiprocessing (~1850 SPS with 32 envs)
- [x] Verify reward goes to max on perfect placement — normalized=1.0, raw=n×8 on perfect teleport (`scripts/check_reward.py`)
- [x] Docker image built and pushed to `gberseth/maniskill-ppo:latest` (Docker Hub)
- [x] Training confirmed working inside Docker container (headless, no GPU required)
- [x] GCP spot instance hello-world test passing — self-deletes after job completes
- [x] Rebuilt Docker image (May 25 2026) to include `WORKDIR /app` and bake in `examples/` — original image (May 23) was missing these, causing `/app` not found errors on GCP
- [ ] Add Docker Hub credentials to GCP startup script (currently image must be public; should support private repos via `DOCKERHUB_TOKEN` passed alongside `WANDB_API_KEY`)
- [ ] Fix GPU rendering in Docker for AMD (gfx1151 / Radeon 8060S): base image `nvidia/cudagl:11.3.1-devel-ubuntu20.04` is too old — `libvulkan_radeon.so` depends on `libLLVM.so.20.1` and `/opt/amdgpu/libdrm_amdgpu.so.1` which don't exist in Ubuntu 20.04. Options: (A) rebase to `ubuntu:22.04` (Mesa 22+ supports gfx1151, no CUDA needed since we use CPU physx); (B) run render tests outside Docker via `.venv` ✅ confirmed working
- [ ] RLinf AMD image (`rlinf/rlinf:agentic-rlinf0.2-libero-rocm6.4`) also fails Vulkan init (gfx1151 + SAPIEN 3.0.1 `ErrorInitializationFailed`). Use `.venv` locally; RLinf image is suitable for NVIDIA GCP instances only
- [x] Spot instance confirmed working in northamerica-northeast1-a — 100k step ppo-test run logged to wandb
- [x] 1M step PPO run started locally (wandb: `real-lab/ManiSkill`, run `push-text-1M`, ~687 SPS on 32 envs)
- [ ] Full training run (10M steps) on GCP spot instance
- [ ] Evaluate success rate after full training

## Training Experiments

### Experiment 1 — Baseline PPO (2026-05-27)
- **Run:** `push-text-state-ppo` · wandb: `real-lab/ManiSkill/runs/zbb798ez`
- **Config:** 32 envs, async CPU, 10M steps (killed early), ~1100 SPS
- **Result:** `eval_success_rate=0.0000` throughout
- **Notes:** Video capture was rendering every step (major slowdown). Fixed: rendering now on-demand only, triggered every 200k steps via a temporary single GPU env (`record_video_episode()`). Checkpoints every 200k steps.

### Experiment 2 — Pre-grasp bridging reward (2026-05-29)
- **Run:** `push-text-pregrasp-reward` · wandb: `real-lab/ManiSkill/runs/qawp9umu`
- **Config:** 32 envs, async CPU, 1M steps, ~1030 SPS
- **Result:** `eval_success_rate=0.0000`, `eval_mean_reward≈0.135`
- **Notes:** Added a bridging reward term: `proximity(tcp→tile) × gripper_close_frac` to encourage closing the gripper when near a tile (filling the reward gap between reach=2 and grasped=4). No improvement. Conclusion: reward shaping alone does not solve the exploration problem — the policy cannot discover structured grasping behaviour through random Gaussian noise on a 9-DoF arm. **Better structured exploration is needed.**

### Experiment 3 — Gap metrics + deterministic eval videos (2026-05-30)
- **Run:** `push-text-gap-metrics` · seed 42 · wandb: `real-lab/ManiSkill`
- **Config:** 32 envs, async CPU, 1M steps, ~795 SPS
- **Gap metrics logged:** `charts/best_trajectory_return`, `charts/avg_top_returns_global`, `charts/avg_top_returns_local`, `charts/global_optimality_gap`, `charts/local_optimality_gap`, `charts/avg_return`, `charts/deterministic_returns`
- **Two eval videos per checkpoint:** `eval/policy_video` (deterministic policy, seed varies) and `eval/deterministic_eval_video` (wraps `BufferGapV2.eval_deterministic()`, fixed seed)

#### Blocker: best-trajectory replay is meaningless due to environment stochasticity
The `eval/deterministic_eval_video` replays the best stored action sequence from a **fixed seed**, but the original episode used a **different random seed** — so the letter positions at the start of the replay differ from the original episode. The replayed actions no longer correspond to the scene geometry they were optimised for, making the video uninterpretable.

More broadly, this reveals a fundamental measurement problem: **we cannot currently determine whether the policy is an exploration problem or an exploitation problem**, because:
1. **Goal stochasticity** — letter spawn positions differ every episode, so the same action sequence produces different outcomes. The best trajectory in one episode cannot be transferred to another.
2. **Possible additional stochasticity** — there may be other sources of randomness (physics, control noise) that further prevent clean trajectory replay.

#### What is needed to diagnose exploration vs exploitation
To use `BufferGapV2` as intended (measuring whether high-return trajectories exist in the buffer but the policy fails to reproduce them), we need either:
- **Fix the initial state** — use a single fixed seed for all training episodes so the same scene is always presented and trajectories are comparable. Then the gap metrics cleanly measure exploitation ability.
- **Or condition trajectory storage on the initial state** — store `(seed, actions, return)` triples and only replay on matching seeds. Complex to implement.

The **fixed-seed approach** is simpler and directly answers the question: given a fixed goal configuration, does the agent ever stumble upon a near-successful trajectory (exploration), and if so, does PPO learn to reproduce it (exploitation)?

### Experiment 4 — PPO-BC imitation quality check (2026-06-01)
- **Run:** `ppo_bc.py` (async vector env)
- **Observation:** Adding imitation from buffered high-return trajectories is currently **not improving performance**.
- **Primary concern:** Need to confirm `eval_deterministic()` is correctly evaluating policy quality, and whether copying the highest-value trajectory is reproducible enough to produce stable quality gains.

#### Immediate verification notes
- [ ] Validate `eval_deterministic()` correctness end-to-end:
  use a known fixed seed and replay both (a) current policy deterministic actions and (b) stored best-trajectory actions; confirm returns match expected rollout behavior.
- [ ] Verify reproducibility for highest-return trajectory replay:
  same trajectory + same initial seed should produce consistent returns across repeated runs.
- [ ] Compare PPO-BC against PPO baseline on identical fixed-layout/fixed-seed settings:
  if imitation is working, deterministic eval return and success should improve earlier than baseline.
- [ ] Track imitation utility directly:
  log and monitor BC loss alongside deterministic return from replayed top trajectory to detect whether cloning signal is meaningful.
- [ ] Gate conclusions on deterministic diagnostics first:
  if deterministic replay is not reproducible, imitation quality measurements are unreliable and must be fixed before further reward/model tuning.

## PushT-v1 Baseline Experiments

### Goal
Replicate the known-good ManiSkill PPO result on PushT-v1 before attempting push-text.
Confirmed PushT-v1 runs at ~232 SPS with 64 envs on Lightning AI (NVIDIA L4).

### Setup — Lightning AI (GPU)
- SSH: `s_01kszc7wvb2vcm5dnpq75f8ds7@ssh.lightning.ai`
- Workspace: `/teamspace/studios/this_studio/ManiSkill`
- GPU: NVIDIA L4, 23 GB VRAM, CUDA 13.0
- Code installed via `git clone https://github.com/Neo-X/ManiSkill.git` + `uv venv --python 3.11 && uv pip install -e .`
- Use `scp` to copy files to remote when testing — **do not commit until confirmed working**
- Run training: `uv run python ppo_upstream.py --env_id='PushT-v1' ...` (from `examples/baselines/ppo/`)

### Known-good settings (from baselines.sh)
```bash
uv run python ppo_upstream.py \
  --env_id="PushT-v1" \
  --num_envs=4096 --num-steps=16 \
  --update_epochs=8 --num_minibatches=32 \
  --gamma=0.99 --total_timesteps=50_000_000 \
  --num_eval_steps=100 --num_eval_envs=16 \
  --no-capture-video
```

### Status
- [x] GPU instance running and code installed
- [x] Smoke test (64 envs, 2000 steps) passes — 232 SPS (~1000 SPS)
- [x] Full 50M step run confirmed working — **17,000 SPS** with 4096 envs on L4; `eval_success_once > 0` observed after ~20M steps
- [x] Killed after success confirmed at ~20M steps — no need to run to 50M
- [ ] Transfer successful hyperparameters to push-text

## StackCube-v1 Baseline Experiments

### Goal
StackCube-v1 is a pick-and-place task (same robot + gripper as push-text) — a closer proxy
than PushT. Replicating PPO success here validates the GPU training pipeline for
manipulation tasks before applying it to push-text.

### Known-good settings (from baselines.sh)
```bash
/teamspace/studios/this_studio/ManiSkill/.venv/bin/python ppo_upstream.py \
  --env_id="StackCube-v1" \
  --num_envs=4096 --num-steps=16 \
  --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=50_000_000 \
  --num_eval_envs=16 \
  --no-capture-video \
  --track \
  --wandb-project-name ManiSkill \
  --wandb-entity real-lab \
  --exp-name stack-cube-v1-gpu-l4
```
Run from: `/teamspace/studios/this_studio/ManiSkill/examples/baselines/ppo/`

### Results
- **~100M steps to reach ~60% success rate** on L4 GPU with 4096 envs
- At 17,000 SPS this is ~100 minutes of wall time
- 60% is the practical ceiling observed — not 80-90% as the official benchmark claims at 50M steps (may require more tuning or longer runs)

### Baseline for push-text
**100M samples / ~60% success** is the reference point. Push-text should be evaluated against this bar:
- If push-text reaches 60% success within ~100M samples, it is on par with StackCube
- If it needs significantly more, the task is harder or the reward/observation needs improvement
- Total timesteps for push-text full runs: set to **100M** to match this baseline

### TODO: Merge upstream ppo.py changes into our ppo.py
Our `ppo.py` was based on the upstream but diverged significantly (added buffer_gap,
wandb tracking, fixed_layout, async CPU path, etc.). The upstream version at commit
`b2cc4334ae5c6162ecddb9fe78465d2ef2911028` uses `sim_backend="physx_cuda"` and works
natively on GPU. Merge strategy when ready:
- Start from upstream version (physx_cuda, no CPUGymWrapper)
- Add `--sim-backend` arg to support CPU fallback
- Re-add buffer_gap, fixed_layout, async path on top
- Test CPU and GPU paths before committing

---

## PushText Environment — Planned Improvements

### Short-term
- [x] **1. Verify letter spawn randomization** — confirmed: tile positions differ across episodes (tested 3 episodes)
- [x] **2. Bigger, more visible letters** — 2× scale (`LETTER_SCALE=[2,2,2]`), `TILE_HALF_SIZE` and `TILE_SPACING` doubled
- [x] **3. Image-based reward (OCR)** — top-down `base_camera` (robot hidden via `get_table_view()`); EasyOCR at 2× upscale detects goal word every 20 steps; +2 reward bonus; videos in `runs/push-text/`
- [ ] **4. Randomize goal letter** — sample goal word randomly each episode instead of hardcoding "AT"
- [ ] **7. Fixed-seed training mode** — add a `--fixed_seed` flag that resets all envs to the same seed every episode, eliminating goal stochasticity. Required to make `BufferGapV2` gap metrics meaningful: with stochastic initial states the best stored trajectory cannot be replayed or compared across episodes, so exploration vs exploitation cannot be diagnosed. With a fixed scene, `charts/global_optimality_gap` directly measures whether the policy fails to learn from good trajectories it has already seen.

### Medium-term
- [ ] **5. Dictionary word sampling** — start with 2–3 letter words from a filtered word list; use the text detection model from (3) to verify the formed word matches the goal; curriculum: increase word length as success rate improves
- [ ] **6. HRL / policy switching** — add a high-level controller that selects which letter to manipulate next; low-level policies handle individual pick-place; enables longer-horizon planning for multi-letter words

### Long-term
- [ ] (Future) Bi-manual variant: two panda_wristcam arms

### Infrastructure
- [ ] **SSH compute backend for `launch_xm_slurm.py`** — add a `--cluster ssh --host <user@host>` option that SSHes into an arbitrary machine and runs the Docker job there, so spare local computers can be used as training workers without needing Slurm or GCP.

---

---

## Docker & GCP Deployment

### Testing

`tests/test_push_text.py` covers simulation correctness, device checks, and a short PPO loop.
Run it in three environments:

```bash
# Local (uses .venv)
uv run python -m pytest tests/test_push_text.py -v

# Docker — headless, no GPU
docker run --rm gberseth/maniskill-ppo:latest \
  bash -c "pip install pytest -q --root-user-action=ignore && \
           python -m pytest /app/tests/test_push_text.py -v"

# GCP Vertex AI smoke job (submits via xmanager, checks wandb for output)
uv run scripts/launch_xm_slurm.py ppo-smoke --cluster gcp
```

The Docker test is the key headless gate — if it passes, the GCP job will work.
Tests include: env creation, episode rollout, fixed/random layout reproducibility,
device enumeration, and a full ppo.py subprocess run.

### Docker image: `gberseth/maniskill-ppo:latest`
- Based on `nvidia/cudagl:11.3.1-devel-ubuntu20.04` (supports GPU via `--gpus all`)
- Works **CPU-only** (no GPU) via `_patch_render_material_noop()` in `mani_skill/envs/sapien_env.py`
- When `render_backend="none"`, SAPIEN's `RenderMaterial` is replaced with a no-op so the URDF
  loader and task code can construct visual assets without a Vulkan device (visuals are never
  added to the scene since `can_render()` is False)
- `render_backend="none"` set automatically when `--no-capture-video`
- `tests/` baked in at `/app/tests/` (added June 2026)
- `WORKDIR /app`, `examples/` baked in at `/app/examples/`, `mani_skill/` installed to site-packages
- Standard run command (no extra mounts needed after May 25 rebuild):
  ```bash
  docker run --rm \
    -v $(pwd)/runs:/app/runs \
    -e WANDB_API_KEY=<key> \
    gberseth/maniskill-ppo:latest python /app/examples/baselines/ppo/ppo.py \
    --track --wandb-project-name ManiSkill --wandb-entity real-lab \
    --use-async-vector-env --num-envs 32 --no-capture-video \
    --exp-name push-text-state-ppo
  ```
- GCP interactive debug: launch `ppo-debug` (n1-standard-8, COS, northamerica-northeast1-a), SSH via IAP, run docker manually to see errors live

### Unified launcher: `scripts/launch_xm_slurm.py` ← **use this for all remote jobs**

XManager-based launcher supporting GCP Vertex AI and Slurm clusters from a single script.
All new experiments should be launched through this script rather than the older `launch_gcp_job.py`.

```bash
# GCP Vertex AI (spot, 8 CPUs)
uv run scripts/launch_xm_slurm.py ppo-fixed-layout --cluster gcp
uv run scripts/launch_xm_slurm.py ppo-training     --cluster gcp

# Mila Slurm cluster
uv run scripts/launch_xm_slurm.py ppo-fixed-layout --cluster mila --user <username>

# DRAC / ComputeCanada
uv run scripts/launch_xm_slurm.py ppo-fixed-layout --cluster drac --user <username>
```

One-time GCP setup (already done):
- GCS staging bucket: `gs://legoassembly-xmanager`
- Vertex AI API enabled on project `legoassembly`
- Service account `xmanager@legoassembly.iam.gserviceaccount.com` with `roles/aiplatform.user`, `roles/storage.objectAdmin`, `roles/logging.logWriter`
- Docker image mirrored to `gcr.io/legoassembly/gberseth/maniskill-ppo:latest` (xmanager pushes automatically)
- All GCP jobs use **SPOT** provisioning (patched into `aiplatform.CustomJob.submit`)
- All jobs capped at **8 CPUs / 16 GiB RAM**

### GCP T4 GPU training: `scripts/launch_gcp_job.py` (GPU jobs)
- Jobs: `ppo-test-t4` (smoke, 5K steps) and `ppo-training-t4` (100M steps, PushText-v1)
- Uses **NGC VMI** image (`nvidia-gpu-cloud-vmi-base-2025-9-1-x86-64`, project `nvidia-ngc-public`) — has Docker + CUDA + nvidia-container-toolkit pre-installed. No manual driver or Docker install needed.
- Machine: `n1-standard-8` + T4 GPU (`nvidia-tesla-t4`), SPOT provisioning, `us-central1-a`
- Runs `ppo_upstream.py` with `physx_cuda` (GPU-parallelized sim), `docker run --gpus all`
- Self-deletes instance on job completion via `trap self_delete EXIT`
- Launch: `uv run scripts/launch_gcp_job.py ppo-training-t4 --zone us-central1-a`
- Smoke test: `uv run scripts/launch_gcp_job.py ppo-test-t4 --zone us-central1-a`

#### GCP T4 setup history (June 2026)
- GCP Deep Learning VM (`deeplearning-platform-release/common-cu129-ubuntu-2204-nvidia-580`) has CUDA but NOT Docker — abandoned
- NGC VMI (`nvidia-ngc-public/nvidia-gpu-cloud-vmi-base-2025-9-1-x86-64`) has Docker + CUDA pre-installed — **use this**
- T4 GPUs not available in `northamerica-northeast1-a` — use `us-central1-a`

### GCP spot instance launcher: `scripts/launch_gcp_job.py` ← legacy CPU jobs, prefer launch_xm_slurm.py
- CPU jobs use Debian 12 + installs Docker at startup
- Self-deletes instance on job completion or crash via `trap self_delete EXIT`
- Requires `roles/compute.instanceAdmin.v1` on the default compute SA (one-time setup):
  ```bash
  PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")
  gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/compute.instanceAdmin.v1" --condition=None
  ```
- Launch jobs:
  ```bash
  python scripts/launch_gcp_job.py hello-world
  python scripts/launch_gcp_job.py ppo-training --gcs-bucket my-bucket
  ```

### GCP project
- Project: `legoassembly`
- Account: `glen.berseth@unsupervisedrobotics.ai`
- Default zone: `us-central1-a`

---

## Key Design Decisions

- **Based on StackCube-v1** — uses `panda_wristcam` (full gripper) so robot can pick, place, and push.
- **Real letter geometry** — OBJ meshes from STL assets, loaded via `add_multiple_convex_collisions_from_file` + `add_visual_from_file`, following `AssemblingKits-v1` pattern.
- **`PACKAGE_ASSET_DIR`** — uses ManiSkill's built-in asset path constant, not fragile relative path computation.
- **Color = identity** — each letter maps to a fixed color from an 8-color palette; partial emission added so colors are visible under overhead lighting.
- **CPU PPO** — uses `--use-async-vector-env` for multiprocessing; eval loop fixed to convert numpy obs to tensors.
- **State obs design** — `target_positions` removed (redundant; `letters_to_targets` encodes the same info). `letter_poses` replaced by `letter_rot_to_target` (axis-angle, 3D per letter, zero = aligned with target, axis defaults to world-Z). Total: 43-dim for "AT".
- **Target orientation** — ghost markers and tiles use 90° CW Z-rotation (`q=[√2/2, 0, 0, -√2/2]`) so letters read correctly in the top-down camera view.
