"""Kinematic metrics for the speed-tracking evaluation, independent of AMP's discriminator."""

import math

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance


FOOT_NAMES = ("left_foot", "right_foot")
POSTURE_BODY_NAMES = ("torso", "left_hand", "right_hand", "left_shin", "right_shin")
MIN_UPRIGHT_TORSO_HEIGHT_M = 0.80


def gait_metrics(positions, dt):
    """Estimate footfalls at the end of distinct swings using foot-link height."""
    landings = []
    for side in range(len(FOOT_NAMES)):
        height = positions[:, side, 2]
        peaks, properties = find_peaks(
            height, distance=math.ceil(0.25 / dt), prominence=0.04,
            wlen=2 * math.ceil(0.55 / dt) + 1,
        )
        for peak, base in zip(peaks, properties["right_bases"]):
            # Accept only a swing with a descent observed before the clip ends.
            low = height[base] + 0.2 * (height[peak] - height[base])
            landing = peak + 1 + np.flatnonzero(height[peak + 1 : base + 1] <= low)[0]
            if landing < len(height) - 1:
                landings.append((int(landing), side, float(positions[landing, side, 0])))
    landings.sort()

    step_lengths = []
    step_times = []
    lengths_by_side = [[], []]
    times_by_side = [[], []]
    forward_steps = []
    for (previous_frame, previous_side, previous_x), (frame, side, x) in zip(landings, landings[1:]):
        interval = (frame - previous_frame) * dt
        if side == previous_side or not 0.15 <= interval <= 1.0:
            continue
        length = x - previous_x
        step_lengths.append(length)
        step_times.append(interval)
        lengths_by_side[side].append(length)
        times_by_side[side].append(interval)
        if length > 0:
            forward_steps.append(frame)

    return {
        "estimated_landings": len(landings),
        "alternating_steps": len(step_lengths),
        "forward_step_frames": forward_steps,
        "step_length_m": float(np.mean(step_lengths)) if step_lengths else None,
        "cadence_steps_per_min": float(60 / np.mean(step_times)) if step_times else None,
        "step_length_asymmetry": side_asymmetry(lengths_by_side),
        "step_time_asymmetry": side_asymmetry(times_by_side),
    }


def side_asymmetry(samples_by_side):
    if not all(samples_by_side):
        return None
    left, right = map(np.mean, samples_by_side)
    denominator = abs(left) + abs(right)
    return float(2 * abs(left - right) / denominator) if denominator > 0 else None


def leg_symmetry(joint_pos, joint_vel, joint_names):
    left = [
        index for index, name in enumerate(joint_names)
        if name.startswith(("left_hip_", "left_knee", "left_ankle_"))
    ]
    right = [joint_names.index("right_" + joint_names[index][5:]) for index in left]
    low, high = np.quantile(joint_pos, [0.05, 0.95], axis=0)
    joint_range = high - low
    velocity_rms = np.sqrt(np.mean(joint_vel**2, axis=0))
    result = {}
    for name, values in (("rom_asymmetry", joint_range), ("velocity_rms_asymmetry", velocity_rms)):
        left_values, right_values = values[left], values[right]
        denominator = left_values + right_values
        moving = denominator > 1e-6
        per_joint = 2 * np.abs(left_values[moving] - right_values[moving]) / denominator[moving]
        result[name] = float(per_joint.mean()) if len(per_joint) else None
    return result


def low_foot_motion(positions, dt):
    """Horizontal speed near the lower foot trajectory, a proxy without contact-force sensing."""
    speed = np.diff(positions, axis=0) / dt
    low_band = np.quantile(positions[..., 2], 0.1, axis=0) + 0.025
    low = np.maximum(positions[:-1, :, 2], positions[1:, :, 2]) <= low_band
    candidates = low & (np.abs(speed[..., 2]) <= 0.15)
    horizontal = np.linalg.norm(speed[..., :2][candidates], axis=-1)
    return {
        "samples": int(horizontal.size),
        "sample_fraction": float(candidates.mean()) if candidates.size else None,
        "horizontal_speed_mean_mps": float(horizontal.mean()) if horizontal.size >= 5 else None,
        "horizontal_speed_p95_mps": float(np.quantile(horizontal, 0.95)) if horizontal.size >= 5 else None,
    }


def reference_motion(motion, joint_names, dt):
    joint_order = motion["dof_names"].tolist()
    foot_order = motion["body_names"].tolist()
    joint_indices = [joint_order.index(name) for name in joint_names]
    foot_indices = [foot_order.index(name) for name in FOOT_NAMES]
    fps = float(motion["fps"])
    stride = round(fps * dt)
    if stride < 1 or not math.isclose(stride, fps * dt):
        raise ValueError("expected reference FPS to be a multiple of the control frequency")
    joint_pos = motion["dof_positions"][:, joint_indices]
    joint_vel = motion["dof_velocities"][:, joint_indices]
    feet = motion["body_positions"][::stride, foot_indices]
    return joint_pos, joint_vel, {
        "fps": fps,
        "joint_frames": len(joint_pos),
        "gait_frames": len(feet),
        "gait": gait_metrics(feet, dt),
        "leg_symmetry": leg_symmetry(joint_pos, joint_vel, joint_names),
        "low_foot_motion_proxy": low_foot_motion(feet, dt),
    }


def episode_motion(joint_pos, joint_vel, actions, feet, reference, joint_names, dt):
    if not all(np.isfinite(values).all() for values in (joint_pos, joint_vel, actions, feet)):
        raise ValueError("non-finite state or action in motion evaluation")

    expert_pos, expert_vel, reference_gait = reference
    result = {}
    for name, sample, expert in (
        ("joint_position_w1_rad", joint_pos, expert_pos),
        ("joint_velocity_w1_radps", joint_vel, expert_vel),
    ):
        distances = [float(wasserstein_distance(joint, ref)) for joint, ref in zip(sample.T, expert.T)]
        result[name] = float(np.mean(distances))
        result[name + "_by_joint"] = dict(zip(joint_names, distances))

    delta = np.abs(np.diff(actions, axis=0))
    result["action_delta"] = {
        "mean_abs": float(delta.mean()) if delta.size else None,
        "p95_abs": float(np.quantile(delta, 0.95)) if delta.size else None,
        "max_abs": float(delta.max()) if delta.size else None,
    }
    result["leg_symmetry"] = leg_symmetry(joint_pos, joint_vel, joint_names)
    result["gait"] = gait_metrics(feet, dt)
    for metric in ("step_length_m", "cadence_steps_per_min"):
        actual = result["gait"][metric]
        expected = reference_gait["gait"][metric]
        result["gait"][metric + "_abs_error"] = abs(actual - expected) if actual is not None and expected is not None else None
    result["low_foot_motion_proxy"] = low_foot_motion(feet, dt)
    return result


def walking_windows(torso_x, dt):
    steps_per_second = round(1 / dt)
    displacements = np.diff(torso_x)
    return [
        float(displacements[start : start + steps_per_second].mean() / dt)
        for start in range(0, len(displacements), steps_per_second)
    ]


def is_complete_forward_motion(complete, windows, forward_step_frames, steps, target_vx):
    middle = steps // 2
    early_steps = sum(frame < middle for frame in forward_step_frames)
    late_steps = len(forward_step_frames) - early_steps
    return complete and min(windows) >= 0.5 * target_vx and min(early_steps, late_steps) >= 2


def upright_posture(body_heights, torso_quaternions):
    torso, left_hand, right_hand, left_shin, right_shin = body_heights.T
    if not np.isfinite(body_heights).all() or not np.isfinite(torso_quaternions).all():
        raise ValueError("non-finite body pose in posture evaluation")
    # Isaac Lab quaternions are wxyz. This is the vertical component of the torso's local +Z.
    torso_up_z = 1 - 2 * np.square(torso_quaternions[:, 1:3]).sum(axis=1)
    hands_low = np.minimum(left_hand, right_hand) < 0.20
    shins_low = np.minimum(left_shin, right_shin) < 0.18
    upright = (torso >= MIN_UPRIGHT_TORSO_HEIGHT_M) & (torso_up_z >= math.cos(math.pi / 4))
    upright &= ~hands_low & ~shins_low
    return {
        "upright_fraction": float(upright.mean()),
        "terminal_upright": bool(upright[-1]),
        "torso_height_median_m": float(np.median(torso)),
        "torso_up_z_median": float(np.median(torso_up_z)),
        "hand_link_below_0_20_m_fraction": float(hands_low.mean()),
        "shin_link_below_0_18_m_fraction": float(shins_low.mean()),
    }
