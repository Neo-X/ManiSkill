# PushText-v1 — Experimental Progress

## Current Phase: Policy Quality

Infrastructure (simulation, Docker, GCP, Vertex AI, wandb) is stable. Focus is now on getting the policy to learn reliably and selecting the best checkpoint.

---

## Baselines

| Task | Steps | Success | Notes |
|------|-------|---------|-------|
| PushT-v1 | 50M | confirmed learning | ~17k SPS on L4; `eval_success_once > 0` after ~20M steps |
| StackCube-v1 | 100M | **~62.5%** | 21k SPS on L4 — reference bar for push-text |
| PushText-v1 | 100M | TBD | current runs in progress |

---

## Experiments

### Experiment 1 — Baseline PPO (2026-05-27)
- **Run:** `push-text-state-ppo` · wandb: `real-lab/ManiSkill/runs/zbb798ez`
- **Config:** 32 CPU envs, 10M steps (killed early)
- **Result:** `eval_success_rate=0.0000` throughout
- **Notes:** Video capture was rendering every step (major slowdown). Fixed.

### Experiment 2 — Pre-grasp bridging reward (2026-05-29)
- **Run:** `push-text-pregrasp-reward` · wandb: `real-lab/ManiSkill/runs/qawp9umu`
- **Config:** 32 CPU envs, 1M steps
- **Result:** `eval_success_rate=0.0000`, `eval_mean_reward≈0.135`
- **Conclusion:** Reward shaping alone does not solve exploration. Random Gaussian noise on a 9-DoF arm cannot discover structured grasping.

### Experiment 3 — Gap metrics + deterministic eval (2026-05-30)
- **Run:** `push-text-gap-metrics` · seed 42
- **Config:** 32 CPU envs, 1M steps
- **Gap metrics logged:** `best_trajectory_return`, `avg_top_returns_global/local`, `global/local_optimality_gap`, `avg_return`, `deterministic_returns`
- **Finding:** Best-trajectory replay is meaningless with stochastic initial states — the best stored action sequence was recorded at a different seed than the replay seed, so actions no longer match the scene geometry.

### Experiment 4 — PPO-BC imitation (2026-06-01)
- **Run:** `ppo_bc.py` (async vector env)
- **Result:** Imitation from buffered high-return trajectories is **not improving performance**
- **Root cause hypothesis:** `eval_deterministic()` may not be correctly attributing quality (see open questions below)

### Experiment 5 — GPU training, 100M steps (2026-06-08)
- **Run:** `push-text-v1-t4-100M` · wandb: `unsupervised-robotics/ManiSkill`
- **Config:** 2048 envs, physx_cuda, T4, num_steps=16, gamma=0.8, update_epochs=8
- **Status:** In progress
- **num_eval_envs increased to 128** to reduce checkpoint-selection variance

---

## Experiment 6 — Architecture & Hyperparameter Search (2026-06-08)

Motivated by the observation that PushText-v1 requires longer-horizon planning than StackCube-v1 (100-step episodes, multi-letter sequencing). Three axes under investigation:

### 6a. Discount factor
- [x] `--gamma 0.95` (current default) — **running**
- [ ] `--gamma 0.99` — longer effective horizon; expect slower early learning but better long-range credit assignment
- [ ] `--gamma 0.995` — approaching undiscounted; may destabilize value estimates

### 6b. Activation functions
- [ ] `tanh` (current default) — bounded, smooth; known to saturate in deep nets
- [ ] `relu` — sparse gradients, no saturation; standard for deep networks
- [ ] `elu` — smooth relu variant; better gradient flow near zero
- [x] `leaky_relu` — **running**
- [ ] `silu` — smooth, self-gated; strong empirical performance in recent work

### 6c. Network size (depth × width)
- [ ] 2 layers × 128 units (current default)
- [ ] 2 layers × 256 units — wider, same depth
- [ ] 4 layers × 256 units — deeper and wider
- [ ] 3 layers × 512 units — large network

---

## Open Questions / Active Work

### 1. Checkpoint selection variance (primary bottleneck)
Three compounding variance sources make it hard to identify the best checkpoint:
1. **Stochastic policy at eval** — same checkpoint produces different returns each run
2. **Single-episode video** — one rollout is far too noisy to judge quality
3. **Random object initialization** — different letter spawn positions each episode

**Fixes needed:**
- [ ] Average eval over N episodes (10–20) at fixed seeds before updating best checkpoint
- [ ] Use `deterministic=True` for checkpoint scoring (already done in `eval_deterministic()`)
- [ ] Fixed eval seeds — same seeds every `eval_freq`, eliminates initialization variance
- [ ] Gate best-checkpoint on `success_once`, not mean reward

### 2. Exploration vs exploitation diagnosis
Cannot currently determine which is the bottleneck because trajectory replay is meaningless with stochastic initial states.

**Options:**
- [ ] `--fixed_seed` training mode — same scene every episode, gap metrics become interpretable
- [ ] Condition trajectory storage on `(seed, actions, return)` and replay only on matching seeds (complex)

### 3. PPO-BC correctness verification
- [ ] Validate `eval_deterministic()` end-to-end with a known fixed seed
- [ ] Verify same trajectory + same seed produces consistent returns across repeated runs
- [ ] Log BC loss alongside deterministic return to detect whether cloning signal is meaningful

---

## Network Architecture (current)

Configurable via CLI flags in `ppo_upstream.py`:

| flag | default | options |
|------|---------|---------|
| `--num_layers` | 2 | any int |
| `--num_units` | 128 | any int |
| `--activation` | tanh | tanh, relu, elu, leaky_relu, silu |
| `--use_layer_norm` | false | — |

Actor output layer always uses `nn.Tanh()` regardless of `--activation`.

---

## Key Design Decisions

- **Based on StackCube-v1** — `panda_wristcam`, full gripper; pick, place, push.
- **Real letter geometry** — OBJ meshes from STL assets, 40mm tall, 2× scale; `TILE_SPACING=0.130m`.
- **State observation (43-dim):** `qpos(9) + qvel(9) + tcp_pose(7) + letter_rot_to_target(n×3) + tcp_to_letters(n×3) + letters_to_targets(n×3)`
- **Episode length:** 100 steps (changed from 200, June 2026)
- **`buffer_gap` tensor fix:** `eval_deterministic()` casts `reward[0]` via `float()` — ManiSkillVectorEnv returns tensors but buffer_gap was designed for numpy; this is the boundary conversion point.
- **Target orientation:** 90° CW Z-rotation so letters read correctly in top-down camera.
