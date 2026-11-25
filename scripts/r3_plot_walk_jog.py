"""Plot fixed-initial-state results before and after the R3 weight reload."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    directory = parser.parse_args().report_dir
    first, resumed = (json.loads((directory / name).read_text())
                      for name in ("formal_4m.json", "warm_4m.json"))
    boundary = first["environment_steps"]
    fig, (survival, coverage) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    fig.suptitle("G1 walk + jog · Unconditioned MimicKit AMP · Seed 42",
                 x=0.02, ha="left", fontweight="bold")

    for run, prior_steps in ((first, 0), (resumed, boundary)):
        stages = [stage for stage in run["checkpoints"] if "fixed" in stage]
        steps = [(stage["environment_steps"] + prior_steps) / 1e6 for stage in stages]
        for skill, color in (("walk", "#0369a1"), ("jog", "#a24b12")):
            scores = [stage["fixed"]["by_initial_skill"][skill] for stage in stages]
            survival.plot(steps, [score["mean_duration_s"] for score in scores],
                          color=color, marker="o", markersize=4,
                          label=f"{skill.title()} reset" if not prior_steps else None)
            coverage.plot(steps, [score["sustained"][skill] * 100 / score["episodes"] for score in scores],
                          color=color, marker="o", markersize=4,
                          label=f"{skill.title()} reset → {skill}" if not prior_steps else None)
        jog = [stage["fixed"]["by_initial_skill"]["jog"] for stage in stages]
        coverage.plot(steps, [score["sustained"]["walk"] * 100 / score["episodes"] for score in jog],
                      color="#834aa2", linestyle="--", marker="s", markersize=4, zorder=2,
                      label="Jog reset → walk" if not prior_steps else None)

    for axis, title, units, ymax in (
        (survival, "Mean survival by reset skill", "Simulated seconds", 10),
        (coverage, "Completed episodes with a sustained gait", "Percent (five resets per skill)", 100),
    ):
        axis.set_title(title, loc="left")
        axis.set_xlabel("Cumulative environment steps (millions)")
        axis.set_ylabel(units)
        axis.set_ylim(-0.025 * ymax, 1.08 * ymax)
        axis.axvline(boundary / 1e6, ls=":", lw=1.3, color="#64748b", label="Weights reloaded")
        axis.grid(axis="y", alpha=0.26)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper left", frameon=False)
    fig.savefig(directory / "fixed_skill_trends.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
