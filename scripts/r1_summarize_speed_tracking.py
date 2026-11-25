"""Validate and summarize the paired seed42 Humanoid speed-tracking comparison."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from r1_walk_motion_metrics import MIN_UPRIGHT_TORSO_HEIGHT_M


STAGES = {"1m": 480, "4m": 1920, "10m": 4880, "20m": 9760, "40m": 19520}
NUM_ENVS = 2048
ROLLOUTS = 16
UPDATES = 1221
EVAL_SEEDS = list(range(42000, 42100))
COLORS = {"ppo": "#26619c", "amp": "#bd5428"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run_metrics(run_dir, num_envs, updates):
    rows = [json.loads(line) for line in (run_dir / "speed_metrics.jsonl").read_text().splitlines()]
    require(len(rows) == updates, f"unexpected number of reward windows in {run_dir}")
    for update, row in enumerate(rows, 1):
        require(row["environment_steps"] == ROLLOUTS and row["samples"] == num_envs * ROLLOUTS,
                f"incorrect sample count in {run_dir} update {update}")
        require(all(math.isfinite(value) for value in row.values()), f"non-finite metric in {run_dir}")
        row["environment_transitions"] = update * row["samples"]
    require((run_dir / "exit.txt").read_text().strip() == "exit_code=0", f"training failed in {run_dir}")
    with (run_dir / "gpu.csv").open() as file:
        gpu = [tuple(map(int, row[3:6])) for row in csv.reader(file)]
    used, free, utilization = zip(*gpu)
    wall_seconds = (
        int((run_dir / "finished_epoch_ns.txt").read_text()) - int((run_dir / "started_epoch_ns.txt").read_text())
    ) / 1e9
    resource = {
        "path": str(run_dir),
        "exit_code": 0,
        "num_envs": num_envs,
        "updates": updates,
        "samples_per_update": num_envs * ROLLOUTS,
        "environment_transitions": rows[-1]["environment_transitions"],
        "started_at": (run_dir / "started_at.txt").read_text().strip(),
        "finished_at": (run_dir / "finished_at.txt").read_text().strip(),
        "wall_seconds": wall_seconds,
        "wall_env_steps_per_second": rows[-1]["environment_transitions"] / wall_seconds,
        "gpu_samples": len(gpu),
        "first_used_mib": used[0],
        "peak_used_mib": max(used),
        "min_free_mib": min(free),
        "gpu_samples_below_100_mib_free": sum(value < 100 for value in free),
        "gpu_util_mean_percent": mean(utilization),
        "gpu_util_peak_percent": max(utilization),
        "compute_apps_before": (run_dir / "gpu_processes_start.csv").read_text().splitlines(),
        "compute_apps_after": (run_dir / "gpu_processes_end.csv").read_text().splitlines(),
        "all_sampled_training_metrics_finite": True,
    }
    return resource, rows


def read_scalars(log_dir):
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    events.Reload()
    curves = {}
    for name in events.Tags()["scalars"]:
        if name.startswith(("Loss / ", "Reward / ", "Episode / ")):
            points = events.Scalars(name)
            require(all(math.isfinite(point.value) for point in points), f"non-finite training signal {name} in {log_dir}")
            curves[name] = [[point.step * NUM_ENVS, point.value] for point in points]
    return curves


def completed(episode):
    return episode["truncated"] and not episode["terminated"]


def wilson_interval(count, total):
    z = 1.959963984540054
    rate = count / total
    center = rate + z**2 / (2 * total)
    radius = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2))
    scale = 1 + z**2 / total
    lower = (center - radius) / scale if count else 0.0
    upper = (center + radius) / scale if count != total else 1.0
    return [lower, upper]


def motion_summary(episodes):
    paths = (
        ("joint_position_w1_rad",),
        ("joint_velocity_w1_radps",),
        ("gait", "step_length_m"),
        ("gait", "step_length_m_abs_error"),
        ("gait", "cadence_steps_per_min"),
        ("gait", "cadence_steps_per_min_abs_error"),
        ("gait", "step_length_asymmetry"),
        ("gait", "step_time_asymmetry"),
        ("leg_symmetry", "rom_asymmetry"),
        ("leg_symmetry", "velocity_rms_asymmetry"),
        ("action_delta", "mean_abs"),
        ("action_delta", "p95_abs"),
        ("low_foot_motion_proxy", "horizontal_speed_mean_mps"),
        ("low_foot_motion_proxy", "sample_fraction"),
    )
    summary = {}
    for path in paths:
        values = []
        for episode in episodes:
            value = episode["motion"]
            for key in path:
                value = value[key]
            if value is not None:
                values.append(value)
        summary[".".join(path)] = {"mean": mean(values) if values else None, "episodes": len(values)}
    return summary


def evaluation_summary(episodes):
    full = [e for e in episodes if completed(e)]
    forward = [e for e in episodes if e["complete_forward_motion"]]
    walked = [e for e in episodes if e["complete_walk"]]
    terminal_heights = [e["terminal_torso_height_m"] for e in full]
    return {
        "episodes": len(episodes),
        "completed": len(full),
        "falls": sum(e["terminated"] for e in episodes),
        "complete_forward_motion": len(forward),
        "unresolved_upright_walks": sum(e["complete_walk"] is None for e in episodes),
        "completed_with_forward_progress": sum(
            min(e["motion"]["forward_window_speed_mps"]) >= 0.75 for e in full
        ),
        "walked": len(walked),
        "complete_wilson_95": wilson_interval(len(full), len(episodes)),
        "forward_motion_wilson_95": wilson_interval(len(forward), len(episodes)),
        "walk_wilson_95": wilson_interval(len(walked), len(episodes)),
        "max_terminal_torso_height_completed_m": max(terminal_heights, default=None),
        "mean_survival_s": mean(e["duration_s"] for e in episodes),
        "mean_forward_dx_m": mean(e["forward_dx_m"] for e in episodes),
        "mean_fixed_horizon_vx_mae_mps": mean(e["target_vx_fixed_horizon_mae_mps"] for e in episodes),
        "complete_only_vx_mae_mps": mean(e["target_vx_mae_mps"] for e in full) if full else None,
        "complete_only_vx_w1_mps": mean(e["reference_vx_wasserstein_mps"] for e in full) if full else None,
        "mean_target_angle_violation": mean(e["target_angle_violation_ratio"] for e in episodes),
        "mean_target_position_error_rad": mean(e["target_position_error_mean_rad"] for e in episodes),
        "motion_full_episodes": motion_summary(full),
        "motion_forward_episodes": motion_summary(forward),
        "motion_walking_episodes": motion_summary(walked),
    }


def first_thresholds(stages, key):
    return {
        str(threshold): next(
            (stage["environment_transitions"] for stage in stages.values() if stage[key] >= threshold), None,
        )
        for threshold in (50, 90)
    }


def plot_curves(output, results, curves):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.2), layout="constrained")
    figure.suptitle("Humanoid28 speed tracking at 1.5 m/s | train seed 42")
    ax_walk, ax_error, ax_distance, ax_reward = axes.flat
    for group, stages in results.items():
        color = COLORS[group]
        x = [s["environment_transitions"] / 1e6 for s in stages.values()]
        name = group.upper()
        ax_walk.plot(x, [s["complete_forward_motion"] for s in stages.values()], "o-", color=color, label=f"{name} sustained forward")
        ax_walk.plot(x, [s["completed"] for s in stages.values()], "x--", color=color, label=f"{name} ran full time")
        ax_error.plot(x, [s["mean_fixed_horizon_vx_mae_mps"] for s in stages.values()], "o-", color=color, label=name)
        ax_distance.plot(x, [s["mean_forward_dx_m"] for s in stages.values()], "o-", color=color, label=name)
        windows = curves[group]["task_reward_per_update"]
        ax_reward.plot(
            [row["environment_transitions"] / 1e6 for row in windows],
            [row["task_reward"] for row in windows], color=color, alpha=0.16, linewidth=0.7,
        )
        batches = [windows[i : i + 20] for i in range(0, len(windows), 20)]
        ax_reward.plot(
            [mean(row["environment_transitions"] for row in batch) / 1e6 for batch in batches],
            [mean(row["task_reward"] for row in batch) for batch in batches], color=color, label=name,
        )
    upright = [stage["walked"] for stages in results.values() for stage in stages.values()]
    if not any(upright):
        ax_walk.plot([], [], " ", label="Upright walking: 0% for both")
    else:
        for group, stages in results.items():
            x = [s["environment_transitions"] / 1e6 for s in stages.values()]
            ax_walk.plot(x, [s["walked"] for s in stages.values()], ":+", color=COLORS[group], label=f"{group.upper()} upright")
    ax_walk.set(ylabel="Episodes (%)", ylim=(-3, 103), title="Paired evaluation (100 initial states)")
    ax_error.set(ylabel="MAE (m/s)", ylim=(0, None), title="Fixed-horizon speed error")
    ax_distance.set(ylabel="Displacement (m)", title="Mean forward displacement")
    ax_reward.set(ylabel="Task reward / step", title="Training task reward (20-update averages)")
    for ax in axes.flat:
        ax.set_xlabel("Environment interactions (millions)")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("amp-log", "ppo-log", "run-root", "capacity-root", "eval-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    log_dirs = {"ppo": args.ppo_log, "amp": args.amp_log}
    configs = {name: yaml.safe_load((path / "params/agent.yaml").read_text()) for name, path in log_dirs.items()}
    ppo, amp = configs["ppo"], configs["amp"]
    require(ppo["seed"] == amp["seed"] == 42 and ppo["memory"] == amp["memory"], "training conditions differ")
    require(ppo["trainer"] == amp["trainer"] and amp["trainer"]["timesteps"] == UPDATES * ROLLOUTS,
            "trainer configuration or timestep budget differs")
    require(ppo["agent"]["rollouts"] == amp["agent"]["rollouts"] == ROLLOUTS, "rollout lengths differ")
    for name in ("policy", "value"):
        require(ppo["models"][name] == amp["models"][name], f"{name} network differs")
    shared = sorted((ppo["agent"].keys() & amp["agent"].keys()) - {"class", "experiment"})
    require(all(ppo["agent"][key] == amp["agent"][key] for key in shared), "shared agent setting differs")
    require(amp["agent"]["task_reward_weight"] == amp["agent"]["style_reward_weight"] == 1, "AMP weights differ")
    env_configs = [path / "params/env.yaml" for path in log_dirs.values()]
    require(env_configs[0].read_bytes() == env_configs[1].read_bytes(), "serialized environment configs differ")

    capacity = [
        run_metrics(args.capacity_root / f"amp_env{num_envs}_updates20", num_envs, 20)[0]
        for num_envs in (256, 512, 1024, 2048)
    ]
    resources = {}
    curves = {}
    results = {group: {} for group in log_dirs}
    evaluation = {group: {} for group in log_dirs}
    for group, log_dir in log_dirs.items():
        run_dir = args.run_root / group
        resources[group], windows = run_metrics(run_dir, NUM_ENVS, UPDATES)
        gpu_archive = args.output / f"{group}_gpu.csv.gz"
        gpu_archive.write_bytes(gzip.compress((run_dir / "gpu.csv").read_bytes(), mtime=0))
        resources[group]["gpu_trace"] = str(gpu_archive)
        curves[group] = {"task_reward_per_update": windows, "tensorboard": read_scalars(log_dir)}
        expected_losses = ["Loss / Policy loss", "Loss / Value loss"]
        if group == "amp":
            expected_losses.append("Loss / Discriminator loss")
        for name in expected_losses:
            require(curves[group]["tensorboard"].get(name), f"missing {name} in {log_dir}")
        for stage, timestep in STAGES.items():
            source = args.eval_root / group / stage / "episodes.json"
            evaluation_status = (source.parent / "exit.txt").read_text().strip()
            require(evaluation_status in ("exit_code=0", "verified_complete_output"), f"evaluation failed in {source.parent}")
            data = json.loads(source.read_text())
            checkpoint = log_dir / "checkpoints" / f"agent_{timestep}.pt"
            require(data["checkpoint_sha256"] == sha256(checkpoint), f"incorrect checkpoint in {source}")
            require(data["training_seed"] == 42 and data["target_vx_mps"] == 1.5, f"incorrect seed or command in {source}")
            episodes = data["episodes"]
            require([e["seed"] for e in episodes] == EVAL_SEEDS, f"incorrect evaluation seed list in {source}")
            require(all(
                (completed(e) and e["steps"] == data["horizon_steps"]) or (e["terminated"] and not e["complete_walk"])
                for e in episodes
            ), f"unexpected episode terminal state in {source}")
            for episode in episodes:
                if "complete_forward_motion" in episode:
                    continue
                # Earlier snapshots only recorded the terminal height. A low terminal frame
                # rules out an upright walking pass without assuming the intermediate posture.
                episode["complete_forward_motion"] = episode["complete_walk"]
                episode["complete_walk"] = (
                    None if episode["complete_forward_motion"]
                    and episode["terminal_torso_height_m"] >= MIN_UPRIGHT_TORSO_HEIGHT_M else False
                )
                episode["posture_samples_available"] = False
            require(not any(e["complete_walk"] is None for e in episodes), f"missing posture traces in {source}")
            evaluation[group][stage] = data
            archive = args.output / "episodes" / group / f"{stage}.json.gz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(gzip.compress(json.dumps(data, allow_nan=False).encode(), mtime=0))
            results[group][stage] = {
                "environment_transitions": timestep * NUM_ENVS,
                "updates": timestep // ROLLOUTS,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": data["checkpoint_sha256"],
                "episodes_archive": str(archive),
                "evaluation_status": evaluation_status,
                **evaluation_summary(episodes),
            }

    paired = {}
    reference = evaluation["ppo"]["1m"]["reference_motion"]
    initial_hashes = [episode["initial_observation_sha256"] for episode in evaluation["ppo"]["1m"]["episodes"]]
    for stage in STAGES:
        for group in log_dirs:
            data = evaluation[group][stage]
            hashes = [episode["initial_observation_sha256"] for episode in data["episodes"]]
            require(hashes == initial_hashes and data["reference_motion"] == reference,
                    f"evaluation initial states or reference differ at {group}/{stage}")
        pairs = list(zip(evaluation["ppo"][stage]["episodes"], evaluation["amp"][stage]["episodes"]))
        same_full = [(p, a) for p, a in pairs if completed(p) and completed(a)]
        same_walk = [(p, a) for p, a in pairs if p["complete_walk"] and a["complete_walk"]]
        paired[stage] = {
            "initial_hash_matches": len(pairs),
            "both_completed": len(same_full),
            "both_walked": len(same_walk),
            "full_episode_vx_mae_ppo_mps": mean(p["target_vx_mae_mps"] for p, _ in same_full) if same_full else None,
            "full_episode_vx_mae_amp_mps": mean(a["target_vx_mae_mps"] for _, a in same_full) if same_full else None,
        }

    summary = {
        "evaluation_seeds": EVAL_SEEDS,
        "target_vx_mps": 1.5,
        "reference_motion": reference,
        "min_upright_torso_height_m": MIN_UPRIGHT_TORSO_HEIGHT_M,
        "capacity_amp": capacity,
        "training": resources,
        "environment_config_sha256": sha256(env_configs[0]),
        "shared_agent_settings": {key: amp["agent"][key] for key in shared},
        "policy_and_value_configs": {name: amp["models"][name] for name in ("policy", "value")},
        "evaluation": results,
        "paired": paired,
        "first_sampled_threshold_steps": {
            group: {key: first_thresholds(stages, key) for key in ("completed", "complete_forward_motion", "walked")}
            for group, stages in results.items()
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output / "training_curves.json.gz").write_bytes(
        gzip.compress(json.dumps(curves, allow_nan=False).encode(), mtime=0),
    )
    plot_curves(args.output / "learning_curves.png", results, curves)


if __name__ == "__main__":
    main()
