import re
import sys
import numpy as np
import matplotlib.pyplot as plt


def extract_rewards(log_path):
    rewards = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if "'reward':" in line:
                match = re.search(r"'reward':\s*([-\d\.eE]+)", line)
                if match:
                    rewards.append(float(match.group(1)))

    return rewards


def moving_average(x, window):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_reward.py <log_file>")
        sys.exit(1)

    log_path = sys.argv[1]

    rewards = extract_rewards(log_path)

    if len(rewards) == 0:
        print("No rewards found!")
        return

    print(f"Found {len(rewards)} reward points.")

    steps = np.arange(len(rewards))

    # ===== smoothing =====
    window = 10   # 👈 你可以调这个（5 / 10 / 20）
    smoothed = moving_average(rewards, window)
    smoothed_steps = np.arange(len(smoothed)) + window - 1

    # ===== plot =====
    plt.figure(figsize=(10, 5))

    # 原始曲线（淡一点）
    plt.plot(steps, rewards, alpha=0.3, label="raw")

    # 平滑曲线（主角）
    plt.plot(smoothed_steps, smoothed, linewidth=2, label=f"moving avg (w={window})")

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Trend (Smoothed)")
    plt.legend()
    plt.grid()

    output_path = log_path + "_reward.png"
    plt.savefig(output_path)
    print(f"Saved plot to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()