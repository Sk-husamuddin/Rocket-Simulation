import gymnasium as gym
from gymnasium.utils.play import play

score = 0

def track(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global score
    score += rew
    if terminated or truncated:
        print("Episode finished. Your score:", round(score, 1))
        score = 0

env = gym.make("LunarLander-v3", render_mode="rgb_array")

play(
    env,
    keys_to_action={"w": 2, "a": 1, "d": 3},
    noop=0,
    callback=track,
    zoom=1.5,
)