import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")
model = PPO("MlpPolicy", env, device="cpu")   # untrained brain

print(model.policy)
print("Total numbers inside the brain:",
      sum(p.numel() for p in model.policy.parameters()))

obs, info = env.reset(seed=42)
obs_tensor, _ = model.policy.obs_to_tensor(obs)
probs = model.policy.get_distribution(obs_tensor).distribution.probs
print("Action probabilities:", probs.detach().numpy().round(3))