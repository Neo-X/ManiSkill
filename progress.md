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

## PushText Environment — Planned Improvements

### Short-term
- [x] **1. Verify letter spawn randomization** — confirmed: tile positions differ across episodes (tested 3 episodes)
- [x] **2. Bigger, more visible letters** — 2× scale (`LETTER_SCALE=[2,2,2]`), `TILE_HALF_SIZE` and `TILE_SPACING` doubled
- [x] **3. Image-based reward (OCR)** — top-down `base_camera` (robot hidden via `get_table_view()`); EasyOCR at 2× upscale detects goal word every 20 steps; +2 reward bonus; videos in `runs/push-text/`
- [ ] **4. Randomize goal letter** — sample goal word randomly each episode instead of hardcoding "AT"

### Medium-term
- [ ] **5. Dictionary word sampling** — start with 2–3 letter words from a filtered word list; use the text detection model from (3) to verify the formed word matches the goal; curriculum: increase word length as success rate improves
- [ ] **6. HRL / policy switching** — add a high-level controller that selects which letter to manipulate next; low-level policies handle individual pick-place; enables longer-horizon planning for multi-letter words

### Long-term
- [ ] (Future) Bi-manual variant: two panda_wristcam arms

---

---

## Docker & GCP Deployment

### Docker image: `gberseth/maniskill-ppo:latest`
- Based on `nvidia/cudagl:11.3.1-devel-ubuntu20.04` (supports GPU via `--gpus all`)
- Works **CPU-only** (no GPU) via null-render stubs in `mani_skill/envs/sapien_env.py`
- `render_backend="none"` set automatically when `--no-capture-video` to skip Vulkan init
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

### GCP spot instance launcher: `scripts/launch_gcp_job.py`
- Uses Debian 12 + installs Docker at startup
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
