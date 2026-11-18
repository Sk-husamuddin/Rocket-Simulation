import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tb(prefix):
    """Read the learning curve from the longest TensorBoard run for an algorithm."""
    best = None
    for run in glob.glob(f"logs/{prefix}_*"):
        acc = EventAccumulator(run)
        acc.Reload()
        if "rollout/ep_rew_mean" not in acc.Tags()["scalars"]:
            continue
        events = acc.Scalars("rollout/ep_rew_mean")
        steps = [e.step for e in events]
        values = [e.value for e in events]
        if best is None or steps[-1] > best[0][-1]:
            best = (steps, values)
    return best


ppo = read_tb("PPO")
dqn = read_tb("DQN")

plt.figure(figsize=(10, 6))
if ppo:
    plt.plot(*ppo, label="PPO")
    print(f"PPO final average reward: {ppo[1][-1]:.1f} after {ppo[0][-1]} steps")
if dqn:
    plt.plot(*dqn, label="DQN")
    print(f"DQN final average reward: {dqn[1][-1]:.1f} after {dqn[0][-1]} steps")

plt.axhline(200, linestyle="--", color="gray", label="Solved (200)")
plt.xlabel("Environment steps (practice moves)")
plt.ylabel("Average episode reward")
plt.title("LunarLander: PPO vs DQN")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
plt.show()