"""Plot paired checkpoint evaluations on the same physical initial states."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 6.2), layout="constrained", sharex=True)
    figure.suptitle("G1 speed commands · Fixed initial states · Seed 42",
                    x=0.02, ha="left", fontweight="bold")
    for variant, color, marker in (("ordinary", "#13689c", "o"), ("conditional", "#b15821", "s")):
        checkpoints = data[variant]["checkpoints"]
        millions = [checkpoint["environment_steps"] / 1e6 for checkpoint in checkpoints]
        for column, (command, speed) in enumerate((("walk", 1.0), ("jog", 1.9))):
            groups = [checkpoint["groups"][command] for checkpoint in checkpoints]
            axes[0, column].plot(millions, [100 * row["successes"] / row["episodes"] for row in groups],
                                 linewidth=1.6, color=color, marker=marker, label=f"{variant.capitalize()} D")
            axes[1, column].plot(millions, [row["mean_mae_vx_full_m_s"] for row in groups],
                                 linewidth=1.6, color=color, marker=marker)
            axes[0, column].set_title(f"{speed:.1f} m/s command")
    for column in range(2):
        axes[0, column].set_ylim(-4, 104)
        axes[1, column].set_ylim(bottom=0)
        axes[1, column].set_xlabel("Environment steps (millions)")
    axes[0, 0].set_ylabel("Command and gait success (%)")
    axes[1, 0].set_ylabel("10-second forward MAE (m/s)")
    axes[0, 0].legend(loc="upper left", frameon=False)
    for axis in axes.flat:
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(args.summary.with_name("fixed_command_trends.png"), dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
