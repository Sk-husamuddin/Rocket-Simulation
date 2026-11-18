import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

def pilot(obs):
    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = obs

    if vy < -0.3:        # falling too fast -> fire main engine
        return 2
    if angle > 0.1:      # tilted one way -> correct with left engine
        return 1
    if angle < -0.1:     # tilted the other way -> right engine
        return 3
    return 0             # otherwise do nothing

for episode in range(10):
    obs, info = env.reset()
    total, done = 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(pilot(obs))
        total += reward
        done = terminated or truncated
    print(f"Episode {episode + 1}: score {total:.1f}")

env.close()