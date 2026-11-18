import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO,DQN

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, "videos_rl", episode_trigger=lambda e: True, name_prefix="rl")
model = PPO.load("ppo_lunar")
#model = DQN.load("dqn_lunar")
for episode in range(5):
    obs, info = env.reset(seed=42 + episode)
    total, done = 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    print(f"Episode {episode + 1}: score {total:.1f}")

env.close()