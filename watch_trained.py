import gymnasium as gym
from stable_baselines3 import DQN, PPO
import numpy as np


env = gym.make("LunarLander-v3", render_mode="human")
model = DQN.load("dqn_lunar")
#model = PPO.load("ppo_lunar")
for episode in range(20):
    obs, info = env.reset()
    total, done = 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    print(f"Episode {episode + 1}: score {total:.1f}")

env.close()