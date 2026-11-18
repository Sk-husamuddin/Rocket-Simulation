from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 8 copies of the lander run at once, so learning is faster
env = make_vec_env("LunarLander-v3", n_envs=8)

# "MlpPolicy" = the policy is a small neural network
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")

# 1 million moves of practice
model.learn(total_timesteps=1_000_000)

model.save("ppo_lunar")