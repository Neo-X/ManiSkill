# PushText Environment — Progress

## Goal

Build a ManiSkill task environment called **PushText-v1** for robotic letter-spelling research.

The robot must pick up and place fridge-magnet-style letter tiles on a white table to spell a target word. This serves as a simpler proxy for LEGO assembly — structured manipulation toward a language-specified goal.

---

## Task Design

**Object geometry:** Real letter-shaped OBJ meshes (40mm tall, 12mm thick), extracted from `assets/objects/Alphabet.stl` and `assets/objects/numbers.stl`. Ghost (semi-transparent) target markers show target positions.

**Robot:** `panda_wristcam` (full gripper — can pick, place, and push).

**Goal specification:** A text string (e.g. `"AT"`, `"CAT"`) passed at env construction. Exposed via `get_language_instruction()` for VLA consumption.

**Target layout:** Letters arranged left-to-right in a centered row on the table, spaced 0.065 m apart.

**Randomization:** Each letter tile spawns at a random (x, y, z-rotation) each episode, with non-overlapping placement via `UniformPlacementSampler`.

**Success:** Every letter within 0.025 m of its target, stationary, and released by gripper.

**Reward:** Staged per-letter reward (reach → grasp → place → release), summed across letters. Mirrors StackCube-v1 structure. Max raw = n_letters × 8, normalized to [0, 1].

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
- [ ] Full training run (10M steps) on GCP spot instance
- [ ] Evaluate success rate after full training
- [ ] (Future) Randomise goal word per episode
- [ ] (Future) Bi-manual variant: two panda_wristcam arms

---

---

## Docker & GCP Deployment

### Docker image: `gberseth/maniskill-ppo:latest`
- Based on `nvidia/cudagl:11.3.1-devel-ubuntu20.04` (supports GPU via `--gpus all`)
- Works **CPU-only** (no GPU) via null-render stubs in `mani_skill/envs/sapien_env.py`
- `render_backend="none"` set automatically when `--no-capture-video` to skip Vulkan init
- Bind-mounts needed at runtime (local source overrides PyPI install):
  ```bash
  docker run --rm -w /app \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/examples:/examples \
    -v $(pwd)/mani_skill:/opt/conda/lib/python3.9/site-packages/mani_skill \
    gberseth/maniskill-ppo:latest python /examples/baselines/ppo/ppo.py \
    --use-async-vector-env --num-envs 32 --no-capture-video \
    --exp-name push-text-state-ppo
  ```

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
- **Color = identity** — each letter maps to a fixed color from an 8-color palette.
- **CPU PPO** — uses `--use-async-vector-env` for multiprocessing; eval loop fixed to convert numpy obs to tensors.
