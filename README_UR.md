# Running Experiments — Unsupervised Robotics Lab

This document covers how to run PushText-v1 training experiments locally and on GCP Vertex AI.

---

## Environment

**Task:** `PushText-v1` — a Panda arm pushes fridge-magnet letter tiles to spell a target word on a table.

**Key options:**
| kwarg | default | description |
|---|---|---|
| `goal_text` | `"AT"` | letters to spell |
| `randomize_letters` | `False` | pick 2 random letters from pool each episode |
| `letter_pool` | `"ABCDEFGHIJKLMNOPQRSTUVWXYZ"` | pool to sample from (keep ≤ 8 with 4096 envs) |
| `fixed_layout` | `False` | deterministic tile spawn positions |

---

## Quick local smoke test

Requires no GPU. Uses CPU physics with a small number of envs.

```bash
uv run python examples/baselines/ppo/ppo.py \
  --num-envs 4 \
  --num-eval-envs 2 \
  --num-steps 50 \
  --num-eval-steps 50 \
  --total-timesteps 5000 \
  --no-capture-video \
  --exp-name smoke-local
```

---

## GCP Vertex AI via `scripts/launch.py`

All remote GPU jobs go through `scripts/launch.py`. It builds a Docker image from the local repo, pushes it to GCR, and submits to Vertex AI — local code changes are included automatically without a separate `docker build && push`.

### Available job definitions

| name | GPU | envs | steps | purpose |
|---|---|---|---|---|
| `ppo-test-t4` | T4 | 1024 | 500k | quick smoke test |
| `ppo-training-t4` | T4 | 2048 | 100M | full T4 training run |
| `ppo-video-test-l4` | L4 | 256 | 500k | video capture test |
| `ppo-training-l4` | L4 | 4096 | 100M | full L4 training run |
| `ppo-smoke` | CPU | 8 | 50k | CPU correctness check |

### Basic launch

```bash
uv run python scripts/launch.py ppo-training-l4 --cluster vertex
```

### Pass extra hyperparameters

Any unrecognised flags are appended to the job command. tyro uses the **last occurrence** of a flag, so these override any defaults baked into the job definition:

```bash
uv run python scripts/launch.py ppo-training-l4 --cluster vertex \
  --env-kwargs "randomize_letters=true,letter_pool=ABCDEFGH" \
  --anneal-lr \
  --gamma 0.99
```

### Multiple seeds (recommended — run at least 2)

```bash
uv run python scripts/launch.py ppo-training-l4 --cluster vertex \
  --seeds 1 2 \
  --env-kwargs "randomize_letters=true,letter_pool=ABCDEFGH" \
  --anneal-lr
```

One docker build, one Vertex AI job per seed. Runs appear in wandb as `push-text-v1-l4-100M-s1` and `push-text-v1-l4-100M-s2`.

### Dry-run (preview without launching)

```bash
uv run python scripts/launch.py ppo-training-l4 --cluster vertex \
  --seeds 1 2 --dry-run \
  --env-kwargs "randomize_letters=true,letter_pool=ABCDEFGH"
```

### Override total timesteps for a quick check

```bash
uv run python scripts/launch.py ppo-test-t4 --cluster vertex \
  --seeds 1 2 \
  --total_timesteps 50000 \
  --no-capture-video
```

---

## Monitoring

```bash
# Check job state
gcloud ai custom-jobs describe <resource-name> \
  --project=legoassembly --format="value(state,error)"

# Stream logs
gcloud logging read "resource.type=\"ml_job\" AND resource.labels.job_id=\"<id>\"" \
  --project=legoassembly --limit=100 --order=asc --format="value(textPayload)"
```

**wandb runs:** https://wandb.ai/unsupervised-robotics/ManiSkill
- Entity: `unsupervised-robotics`
- Project: `ManiSkill`
- API key: read automatically from `~/.netrc` (written by `wandb login`) — never passed as a CLI argument
- Each job logs to a run named by `--exp-name`; multi-seed launches append `-s{seed}` (e.g. `push-text-v1-l4-100M-s1`)

---

## Debugging a failed Vertex AI job

1. Check exit code — `code=3` is a process crash, `code=2` is argument error.
2. Pull the logs with the `gcloud logging read` command above.
3. If logs are empty (crash before Python starts), the job likely crashed during PhysX scene loading:
   - Reduce `letter_pool` size (each letter adds 2 actors per env × num_envs).
   - Reduce `num_envs`.
4. To debug interactively, relaunch via `--cluster gcp` (Compute Engine spot VM) with `--no-spot` if needed, SSH in, and run the docker command manually.

See `CLAUDE.md` for full debugging steps.

---

## Running tests

```bash
# All tests (CPU, ~30s)
uv run python -m pytest tests/test_push_text.py -v

# Just environment correctness
uv run python -m pytest tests/test_push_text.py::TestSimulation -v

# Just randomize_letters
uv run python -m pytest tests/test_push_text.py::TestRandomizeLetters -v

# Just launcher logic
uv run python -m pytest tests/test_push_text.py::TestLauncherMultiSeed -v
```

---

## Known limits and gotchas

- **Letter pool size:** each pool letter pre-builds 2 actors per env (tile + marker). With 4096 envs, a pool of 8 adds 65k actors. Stay at ≤ 8 letters unless you also increase `max_rigid_contact_count` in `sim_config`.
- **Partial resets:** ManiSkill's GPU `set_pose` operates only on the envs in the current reset batch. Pose tensors must have shape `[len(env_idx), 7]`, not `[num_envs, 7]`.
- **LR annealing:** pass `--anneal-lr` at launch time to test it before baking it into a job definition.
- **Region:** Vertex AI L4 jobs default to `northamerica-northeast1`; use `--region us-central1` if that region is oversubscribed. T4 jobs require `us-central1-a` (T4s not available in `northamerica-northeast1`).
- **wandb key:** read automatically from `~/.netrc`. Never passed as a CLI argument.
