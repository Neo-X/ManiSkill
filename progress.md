# PushText Environment — Progress

## Goal

Build a ManiSkill task environment called **PushText-v1** for robotic letter-spelling research.

The robot must push flat letter tiles (like fridge magnets) on a white table into a target row that spells a given goal word. This serves as a simpler proxy for LEGO assembly — structured manipulation toward a language-specified goal.

---

## Task Design

**Object geometry:** Flat rectangular tiles (0.05 × 0.05 × 0.006 m), one per letter in the goal word. Each letter has a distinct color. Ghost (semi-transparent) target tiles mark where each letter should end up.

**Robot:** PandaStick (same as Push-T). Designed to be easy to upgrade to bi-manual (two PandaStick arms) for longer words.

**Goal specification:** A text string (e.g. `"HI"`, `"CAT"`) passed at env construction. Exposed as `get_language_instruction()` for VLA consumption.

**Target layout:** Letters arranged left-to-right in a centered row on the table, spaced 0.07 m apart.

**Randomization:** Each letter tile spawns at a random (x, y) position and random z-rotation each episode.

**Success:** Every letter within 0.025 m of its target position.

**Reward:** Sum of per-letter `(1 - tanh(5 * dist))^2` placement rewards + small TCP-to-nearest-letter shaping term. Max reward = 2.0 (normalized to 1.0).

---

## File Locations

| File | Purpose |
|------|---------|
| `mani_skill/envs/tasks/tabletop/push_text.py` | Main environment class |
| `mani_skill/envs/tasks/tabletop/__init__.py` | Register `PushTextEnv` import |

---

## Status

- [x] Write `push_text.py` — based on StackCube-v1 (panda_wristcam, full gripper)
- [x] Register in `tabletop/__init__.py`
- [x] Import smoke-test passes
- [x] Smoke-test: episode runs cleanly (`demo_random_action -e PushText-v1 --render-mode rgb_array -b cpu`) — rewards, dists, is_placed/is_grasped/is_static all flowing correctly
- [ ] Verify reward goes to max on perfect placement
- [ ] Add `get_language_instruction()` hook for VLA wrapper
- [ ] Import geometry for all 26 alphabet letters (e.g. STL/OBJ meshes shaped like fridge magnet letters) and replace placeholder box tiles with real letter collision/visual meshes
- [ ] (Future) Randomise goal word per episode
- [ ] (Future) Bi-manual variant: two panda_wristcam arms

---

## Key Design Decisions

- **Start from Push-T** — reuse `WhiteTableSceneBuilder`, `PandaStick`, and the same reward shaping style.
- **Flat tiles not shaped letters** — correct letter geometry is hard to simulate stably; tiles are sufficient for the pushing task and VLA goal-specification research.
- **Color = identity** — each letter character maps to a fixed color from an 8-color palette, making visual disambiguation straightforward.
- **`goal_text` as constructor arg** — keeps the env simple; randomising the word per episode is a later extension.
