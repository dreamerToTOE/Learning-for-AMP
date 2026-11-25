#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
    printf 'Usage: %s AMP_LOG_DIR PPO_LOG_DIR OUTPUT_DIR\n' "$0" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AMP_ROOT="$1"
PPO_ROOT="$2"
OUTPUT_ROOT="$3"
status=0

for stage in 1m:480 4m:1920 10m:4880 20m:9760 40m:19520; do
    label="${stage%%:*}"
    timestep="${stage##*:}"
    for group in amp ppo; do
        log_root="$AMP_ROOT"
        if [[ "$group" == ppo ]]; then log_root="$PPO_ROOT"; fi
        checkpoint="${log_root}/checkpoints/agent_${timestep}.pt"
        out="${OUTPUT_ROOT}/${group}/${label}"
        mkdir -p "$out"
        if [[ -s "$out/episodes.json" ]] && python3 - "$out/episodes.json" "$checkpoint" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
checkpoint_sha = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
assert data["checkpoint_sha256"] == checkpoint_sha
assert data["target_vx_mps"] == 1.5 and data["horizon_steps"] == 299
assert [e["seed"] for e in data["episodes"]] == list(range(42000, 42100))
PY
        then
            if [[ ! -f "$out/exit.txt" ]]; then
                printf 'verified_complete_output\n' > "$out/exit.txt"
            fi
            printf '%s %s already evaluated\n' "$group" "$label"
            continue
        fi
        exit_code=0
        timeout --kill-after=30s 900s "${PROJECT_ROOT}/scripts/r1_python.sh" -u "${PROJECT_ROOT}/scripts/r1_eval_walk.py" \
            --checkpoint "$checkpoint" --output "$out" --episodes 100 --seed 42000 --target-vx 1.5 \
            --motion-metrics --headless --device cuda:0 > "$out/stdout.log" 2>&1 || exit_code=$?
        printf 'exit_code=%s\n' "$exit_code" > "$out/exit.txt"
        printf '%s %s exit_code=%s\n' "$group" "$label" "$exit_code"
        if (( exit_code )); then status=1; fi
    done
done
exit "$status"
