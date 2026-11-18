import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("LunarLander-v3")

model = DQN(
    "MlpPolicy", env,
    learning_rate=6.3e-4,
    batch_size=128,
    buffer_size=50_000,
    learning_starts=0,
    gamma=0.99,
    target_update_interval=250,
    train_freq=4,
    gradient_steps=-1,
    exploration_fraction=0.12,
    exploration_final_eps=0.1,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    tensorboard_log="./logs",
    device="cpu",
)

model.learn(total_timesteps=500_000)
model.save("dqn_lunar")