"""Archive R3 baseline resource use and matched closed-loop gait coverage."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime
from itertools import groupby
from pathlib import Path


MODES = ("walk", "jog", "unclassified")


def episode_metrics(episodes, streak):
    windows = [window for episode in episodes for window in episode["windows"]]
    counts = Counter(window["mode"] for window in windows)
    transitions = Counter()
    for episode in episodes:
        if episode["terminal"] != "TIME":
            continue
        bouts = [
            label for label, consecutive in groupby(window["mode"] for window in episode["windows"])
            if label != "unclassified" and len(list(consecutive)) >= streak
        ]
        for left, right in zip(bouts, bouts[1:]):
            if left != right:
                transitions[f"{left}_to_{right}"] += 1

    return {
        "episodes": len(episodes),
        "completed": sum(ep["terminal"] == "TIME" for ep in episodes),
        "falls": sum(ep["terminal"] == "FAIL" for ep in episodes),
        "mean_duration_s": statistics.mean(ep["duration_s"] for ep in episodes),
        "mean_forward_m": statistics.mean(ep["world_forward_m"] for ep in episodes),
        "mean_abs_lateral_m": statistics.mean(abs(ep["world_lateral_m"]) for ep in episodes),
        "sustained": {mode: sum(mode in ep["sustained_skills"] for ep in episodes) for mode in MODES[:2]},
        "both_sustained": sum(len(ep["sustained_skills"]) == 2 for ep in episodes),
        "observed_windows": {mode: counts[mode] for mode in MODES},
        "sustained_bout_transitions_in_completed_episodes": dict(transitions),
        "window_forward_m_s_median": (
            statistics.median(window["median_forward_m_s"] for window in windows) if windows else None
        ),
        "classified_window_means": {
            mode: {
                key: statistics.mean(window[key] for window in windows if window["mode"] == mode)
                for key in ("median_forward_m_s", "mean_lateral_m_s", "mean_yaw_rate_rad_s",
                            "foot_support_switches", "no_foot_support_fraction")
            }
            for mode in MODES[:2] if counts[mode]
        },
    }


def summarize(run):
    with (run / "gpu.csv").open() as stream:
        reader = csv.reader(stream)
        next(reader)
        samples = [row for row in reader if len(row) == 4]
    gpu = [tuple(map(int, row[1:4])) for row in samples]
    timestamps = [datetime.fromisoformat(row[0].replace("/", "-")) for row in (samples[0], samples[-1])]
    gpu_seconds = (timestamps[1] - timestamps[0]).total_seconds()
    stdout = (run / "stdout.log").read_text()
    header, *rows = (line.split() for line in (run / "log.txt").read_text().splitlines() if line.strip())
    checkpoints = [dict(zip(header, map(float, row), strict=True)) for row in rows]
    assert all(math.isfinite(value) for row in checkpoints for value in row.values())
    updates = [int(value) for value in re.findall(r"^\|\s*Samples\s*\|\s*(\d+)\s*\|", stdout, re.M)]
    assert updates == list(range(4096, updates[-1] + 1, 4096)), "Unexpected rollout step sequence"
    assert not re.search(r"\|\s*[-+]?(?:nan|inf)\s*\|", stdout, re.I), "Non-finite training log"
    exit_file = run / "exit_code.txt"
    timer_file = run / "wall_seconds.txt"
    timer = timer_file.read_text().strip() if timer_file.is_file() else ""
    last_checkpoint = run / "int_models" / f"model_{int(checkpoints[-1]['Iteration']):010d}.pt"
    elapsed = (float(timer.splitlines()[-1]) if timer else
               last_checkpoint.stat().st_mtime - (run / "gpu_before.csv").stat().st_mtime)
    summary = {
        "run": run.name,
        "mimickit_commit": (run / "upstream_commit.txt").read_text().strip(),
        "seed": 42, "environments": 128, "rollout_steps_per_environment": 32,
        "updates": len(updates), "environment_steps": updates[-1],
        "exit_code": int(exit_file.read_text()) if exit_file.is_file() else None,
        "wall_seconds": elapsed,
        "wall_seconds_source": "process_timer" if timer else "gpu_before_snapshot_to_last_checkpoint",
        "gpu_sample_count": len(samples),
        "gpu_sampling_seconds": gpu_seconds,
        "gpu_max_used_mib": max(row[0] for row in gpu), "gpu_min_free_mib": min(row[1] for row in gpu),
        "gpu_average_utilization_pct": statistics.mean(row[2] for row in gpu),
        "capacity_warnings": re.findall(r"(?im)^.*overflow - please increase.*$", stdout),
        "checkpoints": [],
    }
    fixed_starts = None
    for stats in checkpoints:
        index, steps = int(stats["Iteration"]), int(stats["Samples"])
        checkpoint = run / "int_models" / f"model_{index:010d}.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        result = {"environment_steps": steps, "checkpoint": checkpoint.name, "sha256": digest,
                  "trainer_diagnostics": stats}
        for split in ("fixed", "held_out"):
            measured = run / f"eval_{split}_{index:010d}.json"
            if not measured.exists():
                continue
            eval_result = json.loads(measured.read_text())
            assert eval_result["checkpoint_sha256"] == digest
            assert eval_result["split"] == split
            protocol = eval_result["protocol"]
            episodes = eval_result["episodes"]
            starts = [episode["initial_observation_sha256"] for episode in episodes]
            assert len(starts) == len(set(starts)) == 2 * protocol[f"{split}_episodes_per_initial_skill"]
            if split == "fixed":
                fixed_starts = starts if fixed_starts is None else fixed_starts
                assert fixed_starts == starts, "Fixed evaluation starts changed between checkpoints"
            if split == "held_out" and fixed_starts is not None:
                assert set(starts).isdisjoint(fixed_starts)
            streak = protocol["sustained_windows"]
            result[split] = {
                "all": episode_metrics(episodes, streak),
                "by_initial_skill": {
                    skill: episode_metrics([ep for ep in episodes if ep["initial_skill"] == skill], streak)
                    for skill in MODES[:2]
                },
                "episodes": episodes,
            }
        summary["checkpoints"].append(result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(summary["run"], summary["environment_steps"], summary["wall_seconds"], "s")
    for result in summary["checkpoints"]:
        for split in ("fixed", "held_out"):
            if split in result:
                overall = result[split]["all"]
                print(result["environment_steps"], split, overall["completed"], "/", overall["episodes"],
                      "completed", overall["sustained"], overall["observed_windows"])


if __name__ == "__main__":
    main()
