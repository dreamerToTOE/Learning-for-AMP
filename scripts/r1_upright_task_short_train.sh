#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    printf 'Usage: %s OUTPUT_DIR NUM_ENVS UPDATES\n' "$0" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="$(realpath -m "$1")"
num_envs="$2"
updates="$3"
mkdir -p "$run_root"
cd "$project_root"
run_id="$(date +%Y%m%d_%H%M%S)_$(basename "$run_root")"

run_group() {
    local group="$1" algorithm="$2"
    local run="$run_root/$group"
    mkdir "$run"
    local -a overrides=()
    if [[ "$algorithm" == AMP ]]; then
        overrides=(agent.agent.task_reward_weight=1.0 agent.agent.style_reward_weight=1.0)
        export R1_AMP_STYLE_METRICS_PATH="$run/amp_style_metrics.jsonl"
    fi
    export R1_SPEED_METRICS_PATH="$run/speed_metrics.jsonl"
    local -a command=(
        scripts/r1_python.sh -u scripts/r1_speed_tracking_train.py --upright-task
        --task Isaac-Humanoid-AMP-Walk-Direct-v0 --algorithm "$algorithm"
        --num_envs "$num_envs" --max_iterations "$updates" --seed 42 --headless --device cuda:0
        "hydra.run.dir=$run/hydra" "agent.agent.experiment.directory=r1_upright_$run_id"
        "agent.agent.experiment.experiment_name=$group"
        agent.agent.experiment.checkpoint_interval=512 agent.agent.experiment.write_interval=128
        "${overrides[@]}"
    )
    printf '%q ' "${command[@]}" > "$run/command.txt"
    printf '\n' >> "$run/command.txt"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "$run/gpu_processes_start.csv"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits -lms 250 > "$run/gpu.csv" &
    local monitor_pid=$!
    local status=0
    date +%s%N > "$run/start_ns.txt"
    timeout --kill-after=30s 600s "${command[@]}" > "$run/stdout.log" 2>&1 || status=$?
    date +%s%N > "$run/end_ns.txt"
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "$run/gpu_processes_end.csv"
    if [[ ! -s "$R1_SPEED_METRICS_PATH" ]]; then status=1; fi
    printf 'exit_code=%s\n' "$status" > "$run/exit.txt"
    printf '%s %s\n' "$group" "$status"
    return "$status"
}

printf '%s\n' "logs/skrl/r1_upright_$run_id" > "$run_root/log_dir.txt"
run_group amp AMP
unset R1_AMP_STYLE_METRICS_PATH
run_group ppo PPO
