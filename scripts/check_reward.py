"""Verify that perfect letter placement yields max reward and save a render."""
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: registers all envs
from PIL import Image, ImageDraw

GOAL = "AT"
OUT_IMAGE = "/tmp/check_reward.png"

env = gym.make("PushText-v1", obs_mode="rgbd", goal_text=GOAL, num_envs=1,
               render_mode="rgb_array")
obs, _ = env.reset()
inner = env.unwrapped

from mani_skill.utils.structs.pose import Pose

# Teleport every tile to its exact target position, stationary
for tile, (tx, ty) in zip(inner.letter_tiles, inner.target_xys):
    z = inner.tile_half_size[2].item() + 1e-3
    p = torch.tensor([[tx, ty, z]], device=inner.device)
    tile.set_pose(Pose.create_from_pq(p=p))
    tile.set_linear_velocity(torch.zeros(1, 3, device=inner.device))
    tile.set_angular_velocity(torch.zeros(1, 3, device=inner.device))

# Move arm out of the overhead camera's view: rotate base joint 90° to the side
qpos = inner.agent.robot.get_qpos().clone()
qpos[:, 0] = np.pi / 2   # swing arm sideways away from table view
inner.agent.robot.set_qpos(qpos)
inner.agent.robot.set_qvel(torch.zeros_like(qpos))

# Step once so physics, sensors, and info all update
action = torch.zeros(1, env.action_space.shape[-1])
obs, reward, terminated, truncated, info = env.step(action)

n = len(GOAL)
max_raw = n * 8.0
norm_reward = reward.item()
raw = inner.compute_dense_reward(obs=obs, action=action, info=info).item()

print(f"goal: {GOAL!r}  n_letters={n}  max_raw={max_raw}")
print(f"reward_mode : {inner.reward_mode}")
print(f"norm reward : {norm_reward:.4f}  (expected 1.0)")
print(f"raw reward  : {raw:.4f}  (expected {max_raw})")
print(f"success     : {info['success'].item()}")
print(f"letter_dists: {info['letter_dists']}")
print(f"is_placed   : {info['is_placed']}")
print(f"is_static   : {info['is_static']}")
print(f"is_grasped  : {info['is_grasped']}")

def sensor_to_pil(obs, key):
    """Extract a camera's RGB image from obs as a PIL Image."""
    arr = obs["sensor_data"][key]["rgb"]  # (1, H, W, 3) torch uint8
    return Image.fromarray(arr[0].cpu().numpy())

img_overhead = sensor_to_pil(obs, "base_camera")
img_wrist    = sensor_to_pil(obs, "hand_camera")

# Scale both to 256×256 for a cleaner side-by-side
size = 256
img_overhead = img_overhead.resize((size, size), Image.NEAREST)
img_wrist    = img_wrist.resize((size, size), Image.NEAREST)

# Stitch side by side with a 4px gap and labels
gap = 4
label_h = 20
canvas_w = size * 2 + gap
canvas_h = size + label_h
canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))
canvas.paste(img_overhead, (0, label_h))
canvas.paste(img_wrist,    (size + gap, label_h))

draw = ImageDraw.Draw(canvas)
draw.text((4, 2),              "overhead (base_camera)", fill=(220, 220, 220))
draw.text((size + gap + 4, 2), "wrist (hand_camera)",    fill=(220, 220, 220))

# Reward overlay on bottom-left of overhead image
lines = [
    f"norm reward: {norm_reward:.4f}  (max=1.0)",
    f"success: {info['success'].item()}",
]
y = label_h + size - len(lines) * 16 - 4
for line in lines:
    draw.text((5, y + 1), line, fill=(0, 0, 0))
    draw.text((4, y),     line, fill=(255, 255, 100))
    y += 16

canvas.save(OUT_IMAGE)
print(f"image saved → {OUT_IMAGE}")
env.close()
