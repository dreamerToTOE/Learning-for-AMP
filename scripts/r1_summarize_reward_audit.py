"""Summarize paired frozen-policy replays and verify the upright task acceptance checks."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


GROUPS = ("official", "ppo", "amp")
FOOT_NAMES = ("left_foot", "right_foot")
HAND_NAMES = ("left_hand", "right_hand")
SHIN_NAMES = ("left_shin", "right_shin")
FORCE_PREFIX = "ground_normal_force_n/"


def quantiles(values):
    return dict(zip(("min", "p05", "median", "p95", "max"), np.quantile(values, [0, 0.05, 0.5, 0.95, 1]).tolist()))


def complete(episode):
    return episode["truncated"] and not episode["terminated"]


def collect(episodes, key):
    return np.concatenate([episode["reward_audit"]["traces"][key] for episode in episodes])


def replay_summary(payload, contact_threshold):
    episodes = payload["episodes"]
    traces = episodes[0]["reward_audit"]["traces"]
    alive = collect(episodes, "applied_alive")
    bonus = collect(episodes, "speed_bonus_before_gate")
    task_rewards = collect(episodes, "applied_task_reward")
    if not np.allclose(task_rewards, alive * (1 + bonus), atol=2e-7):
        raise ValueError("task reward and its components disagree")
    result = {
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "episodes": len(episodes),
        "steps": int(alive.size),
        "timeouts": sum(complete(e) for e in episodes),
        "upright_walks": sum(e["complete_walk"] for e in episodes),
        "original_height_terminations": sum(e["reward_audit"]["original_terminated"] for e in episodes),
        "upright_rule_terminations": sum(e["failure_reason"] == "upright_task_rule" for e in episodes),
        "termination_steps": [e["steps"] for e in episodes],
        "termination_reasons_by_episode": [e.get("termination_rule_reasons", []) for e in episodes],
        "mean_forward_m": float(np.mean([e["forward_dx_m"] for e in episodes])),
        "torso_height_m": quantiles(collect(episodes, "torso_height_m")),
        "torso_tilt_deg": quantiles(collect(episodes, "torso_tilt_deg")),
        "mean_alive_reward": float(alive.mean()),
        "mean_earned_speed_reward": float((alive * bonus).mean()),
        "mean_task_reward": float(task_rewards.mean()),
        "mean_style_reward_from_frozen_discriminators": {
            key: float(collect(episodes, key).mean()) for key in traces if key.endswith("_style_reward")
        },
    }
    forces = {
        key.removeprefix(FORCE_PREFIX): collect(episodes, key)
        for key in traces if key.startswith(FORCE_PREFIX)
    }
    if forces:
        other = tuple(name for name in forces if name not in FOOT_NAMES + HAND_NAMES + SHIN_NAMES)
        contact_groups = {
            "feet": FOOT_NAMES, "hands": HAND_NAMES, "shins": SHIN_NAMES, "other_non_foot": other,
            "any_non_foot": HAND_NAMES + SHIN_NAMES + other,
        }
        result["ground_contact"] = {
            "per_link": {
                name: {"force_n_max": float(force.max()), "fraction_above_threshold": float((force >= contact_threshold).mean())}
                for name, force in forces.items()
            },
            "fraction_by_group": {
                name: float((np.stack([forces[body] for body in names]) >= contact_threshold).any(axis=0).mean())
                for name, names in contact_groups.items()
            },
        }
    if payload["reward_audit"]["upright_task"]:
        height = collect(episodes, "torso_height_m")
        tilt = collect(episodes, "torso_tilt_deg")
        rule = payload["reward_audit"]["rules"]
        non_foot = np.stack([force for name, force in forces.items() if name not in FOOT_NAMES])
        eligible = (height >= rule["min_torso_height_m"]) & (tilt <= rule["max_torso_tilt_deg"])
        eligible &= (non_foot < contact_threshold).all(axis=0)
        if not np.array_equal(eligible, alive.astype(bool)):
            raise ValueError("new task reward eligibility disagrees with the recorded posture and forces")
    return result


def assert_same_states(payloads):
    first = payloads[0]["episodes"]
    for payload in payloads[1:]:
        for left, right in zip(first, payload["episodes"], strict=True):
            for key in ("seed", "initial_observation_sha256"):
                if left[key] != right[key]:
                    raise ValueError(f"paired initial state mismatch on {key}")


def same_trajectory(a, b):
    return all(a[key] == b[key] for key in ("steps", "terminated", "truncated", "forward_dx_m", "torso_vx_mps"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("runs", "output", "expert", "historical-training"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    modes = ("calibration_original", "calibration_contacts", "validation_original", "validation_upright")
    replays = {}
    for mode in modes:
        replays[mode] = {}
        for group in GROUPS:
            run = args.runs / mode / group
            if (run / "exit.txt").read_text().strip() != "exit_code=0":
                raise ValueError(f"failed replay: {run}")
            raw = (run / "episodes.json").read_bytes()
            replays[mode][group] = json.loads(raw)
            (args.output / f"{mode}_{group}.json.gz").write_bytes(gzip.compress(raw, mtime=0))

    for split in ("calibration", "validation"):
        assert_same_states([payload for mode in modes if mode.startswith(split) for payload in replays[mode].values()])
    threshold = replays["validation_upright"]["official"]["reward_audit"]["rules"]["non_foot_contact_force_n"]
    summary = {mode: {g: replay_summary(data, threshold) for g, data in groups.items()} for mode, groups in replays.items()}
    parity = {}
    for group in GROUPS:
        without_sensor = replays["calibration_original"][group]["episodes"]
        with_sensor = replays["calibration_contacts"][group]["episodes"]
        parity[group] = sum(same_trajectory(a, b) for a, b in zip(without_sensor, with_sensor, strict=True))

    baseline = replays["validation_original"]["official"]["episodes"]
    upright = replays["validation_upright"]["official"]["episodes"]
    walking_pairs = [(a, b) for a, b in zip(baseline, upright, strict=True) if a["complete_walk"]]
    validation = {
        "official_legacy_upright_walks": len(walking_pairs),
        "official_upright_walks_preserved": sum(b["complete_walk"] for _, b in walking_pairs),
        "official_walking_trajectories_unchanged": sum(same_trajectory(a, b) for a, b in walking_pairs),
        "all_speed_policy_replays_ended_by_upright_rule": all(
            e["failure_reason"] == "upright_task_rule"
            for group in ("ppo", "amp") for e in replays["validation_upright"][group]["episodes"]
        ),
    }
    with np.load(args.expert, allow_pickle=False) as motion:
        names = motion["body_names"].tolist()
        torso = names.index("torso")
        up_z = 1 - 2 * np.square(motion["body_rotations"][:, torso, 1:3]).sum(axis=1)
        expert = {
            "frames": len(up_z),
            "torso_height_m": quantiles(motion["body_positions"][:, torso, 2]),
            "torso_tilt_deg": quantiles(np.rad2deg(np.arccos(np.clip(up_z, -1, 1)))),
            "position_height_min_m_by_link": dict(zip(names, motion["body_positions"][..., 2].min(axis=0).tolist())),
            "contact_force_data_available": False,
        }

    with gzip.open(args.historical_training, "rt") as source:
        history = json.load(source)["amp"]["task_reward_per_update"]
    historical = {}
    for lo, hi in ((1, 30), (101, 120), (286, 305), (591, 610), (1201, 1220)):
        windows = history[lo - 1 : hi]
        historical[f"updates_{lo}_{hi}"] = {
            "alive_reward": float(np.mean([w["alive_reward"] for w in windows])),
            "earned_speed_reward": float(np.mean([w["task_reward"] - w["alive_reward"] for w in windows])),
            "task_reward": float(np.mean([w["task_reward"] for w in windows])),
        }
    result = {
        "calibration_reset_seeds": [45000, 45009],
        "validation_reset_seeds": [45010, 45029],
        "rules": replays["validation_upright"]["official"]["reward_audit"]["rules"],
        "style_references": replays["validation_upright"]["official"]["reward_audit"]["style_references"],
        "expert": expert,
        "calibration_contact_sensor_exact_matches_out_of_10": parity,
        "validation": validation,
        "replays": summary,
        "historical_velocity_amp_task_reward": historical,
        "historical_velocity_amp_actual_style_rewards_logged": False,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
