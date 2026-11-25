#!/usr/bin/env bash
set -euo pipefail

if (( $# < 6 )); then
    printf 'Usage: %s OFFICIAL_CHECKPOINT PPO_CHECKPOINT AMP_CHECKPOINT OUTPUT SEED EPISODES [EVAL_ARGS...]\n' "$0" >&2
    exit 2
fi

checkpoints=("$1" "$2" "$3")
output="$4"
seed="$5"
episodes="$6"
shift 6
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
groups=(official ppo amp)
for index in "${!groups[@]}"; do
    group="${groups[index]}"
    run="$output/$group"
    mkdir -p "$run"
    status=0
    timeout --kill-after=20s 360s "$project_root/scripts/r1_python.sh" -u "$project_root/scripts/r1_eval_walk.py" \
        --checkpoint "${checkpoints[index]}" --output "$run" --seed "$seed" --episodes "$episodes" \
        --target-vx 1.5 --motion-metrics --reward-audit --style-reference "official=${checkpoints[0]}" \
        --style-reference "speed=${checkpoints[2]}" --headless --device cuda:0 "$@" > "$run/stdout.log" 2>&1 || status=$?
    if [[ ! -s "$run/episodes.json" ]]; then status=1; fi
    printf 'exit_code=%s\n' "$status" > "$run/exit.txt"
    printf '%s exit_code=%s\n' "$group" "$status"
    if (( status )); then exit "$status"; fi
done
