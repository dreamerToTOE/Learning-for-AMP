"""Archive comparisons from matching G1 training budgets and physical resets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import math
import re
from statistics import mean


def summarize(run):
    protocol = json.loads((run / "speed_amp.json").read_text())
    train = protocol["training"]
    steps_per_update = train["environments"] * 32
    lines = (run / "stdout.log").read_text()
    samples = [int(value) for value in re.findall(r"^\|\s*Samples\s*\|\s*(\d+)\s*\|", lines, re.M)]
    assert samples == list(range(steps_per_update, train["formal_steps"] + 1, steps_per_update))
    assert not re.search(r"(?im)\|\s*[+-]?(nan|inf)\s*\||out of memory|overflow - please increase", lines)
    assert int((run / "exit_code.txt").read_text()) == 0
    with (run / "gpu.csv").open() as stream:
        measured = [(int(used), int(free), int(util)) for _, used, free, util in list(csv.reader(stream))[1:]]
    fields, *rows = [line.split() for line in (run / "log.txt").read_text().splitlines() if line.strip()]
    stats = [dict(zip(fields, map(float, row), strict=True)) for row in rows]
    checkpoints = []
    for snapshot in stats:
        index = f"{int(snapshot['Iteration']):010d}"
        assert all(math.isfinite(value) for value in snapshot.values())
        checkpoint = run / "int_models" / f"model_{index}.pt"
        fixed = json.loads((run / f"eval_fixed_{index}.json").read_text())
        assert fixed["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checkpoints.append({"index": index, "environment_steps": int(snapshot["Samples"]), "training": snapshot,
                            "checkpoint_sha256": fixed["checkpoint_sha256"], "groups": fixed["groups"]})
    final = checkpoints[-1]["index"]
    independent = json.loads((run / f"eval_held_out_{final}.json").read_text())
    assert independent["checkpoint_sha256"] == checkpoints[-1]["checkpoint_sha256"]
    assert independent["protocol"] == protocol
    seconds = float((run / "wall_seconds.txt").read_text())
    return {
        "run": run.name, "training_code_commit": (run / "project_commit.txt").read_text().strip(),
        "upstream_commit": (run / "upstream_commit.txt").read_text().strip(),
        "variant": independent["variant"], "seed": train["seed"], "environment_steps": samples[-1],
        "updates": len(samples), "startup_to_exit_s": seconds, "env_steps_per_s": samples[-1] / seconds,
        "gpu_samples": len(measured), "gpu_peak_mib": max(x[0] for x in measured),
        "gpu_min_free_mib": min(x[1] for x in measured), "mean_gpu_utilization_pct": mean(x[2] for x in measured),
        "initial_parameters": json.loads((run / "initial_parameters.json").read_text()),
        "checkpoints": checkpoints, "held_out": independent["groups"],
        "held_out_style": independent["style_of_completed_constant_command_episodes"],
    }


def start_hashes(result):
    return {ep["id"]: ep["initial_observation_sha256"] for ep in result["episodes"]}


def check_comparison(ordinary, conditional):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Use the project's data venv") from exc
    for name in ("speed_amp.json", "motions.yaml", "audit.json", "agent_config.yaml",
                 "newton_engine.yaml", "upstream_commit.txt", "upstream.patch"):
        assert (ordinary / name).read_bytes() == (conditional / name).read_bytes(), name
    a, b = [yaml.safe_load((run / "env_config.yaml").read_text()) for run in (ordinary, conditional)]
    assert a.pop("condition_discriminator") is False
    assert b.pop("condition_discriminator") is True
    assert a == b
    result_paths = sorted(ordinary.glob("eval_fixed_*.json")) + sorted(ordinary.glob("eval_held_out_*.json"))
    reset_sets = {}
    for path in result_paths:
        a, b = [json.loads(candidate.read_text()) for candidate in (path, conditional / path.name)]
        assert a["protocol"] == b["protocol"] and a["gait"] == b["gait"]
        starts = start_hashes(a)
        assert starts == start_hashes(b)
        assert starts == reset_sets.setdefault(a["split"], starts)
    assert set(reset_sets["fixed"].values()).isdisjoint(reset_sets["held_out"].values())
    return {split: len(set(starts.values())) for split, starts in reset_sets.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ordinary", type=Path)
    parser.add_argument("conditional", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    a, b = (summarize(run) for run in (args.ordinary, args.conditional))
    for key in ("actor", "critic"):
        assert a["initial_parameters"][key] == b["initial_parameters"][key]
    assert a["environment_steps"] == b["environment_steps"]
    reset_counts = check_comparison(args.ordinary, args.conditional)
    output = {"ordinary": a, "conditional": b, "matched_unique_reset_counts": reset_counts}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(a["run"], a["startup_to_exit_s"], "s")
    print(b["run"], b["startup_to_exit_s"], "s")
    print("Unique paired physical resets:", reset_counts)


if __name__ == "__main__":
    main()
