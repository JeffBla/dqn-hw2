import gymnasium as gym
import ale_py

# 確保 Atari 註冊
gym.register_envs(ale_py)

env_id = "ALE/Breakout-v5"
env = gym.make(env_id, render_mode="human")  # 單環境、人類視窗
print("render_modes:", env.metadata.get("render_modes"))

obs, info = env.reset()
# Breakout 常要先按 FIRE 才會開始
if "FIRE" in env.unwrapped.get_action_meanings():
    for _ in range(2):
        obs, r, term, trunc, info = env.step(1)
        if term or trunc:
            obs, info = env.reset()
            break

for _ in range(200):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        obs, info = env.reset()
env.close()
