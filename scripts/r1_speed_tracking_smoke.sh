#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${PROJECT_ROOT}/runs/r1_speed_tracking_smoke"
mkdir -p "${RUN_ROOT}"

run_group() {
    local group="$1"
    local algorithm="$2"
    local dir="${RUN_ROOT}/${group}_seed42_env2_updates50"
    local metrics="${dir}/speed_metrics.jsonl"
    mkdir -p "${dir}"
    printf '%s\n' "algorithm=${algorithm} seed=42 num_envs=2 updates=50 transitions=1600 target_vx=1.5 sigma_vx=0.5" > "${dir}/command.txt"
    date --iso-8601=seconds > "${dir}/started_at.txt"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -lms 500 > "${dir}/gpu.csv" 2>/dev/null &
    local monitor_pid=$!
    if [[ "${algorithm}" == "AMP" ]]; then
        timeout --kill-after=30s 900s env R1_SPEED_METRICS_PATH="${metrics}" R1_SPEED_METRICS_FLUSH_STEPS=16 \
            "${PROJECT_ROOT}/scripts/r1_python.sh" -u "${PROJECT_ROOT}/scripts/r1_speed_tracking_train.py" \
            --task Isaac-Humanoid-AMP-Walk-Direct-v0 --algorithm AMP --num_envs 2 --max_iterations 50 \
            --seed 42 --headless --device cuda:0 \
            agent.agent.task_reward_weight=1.0 agent.agent.style_reward_weight=1.0 \
            agent.agent.experiment.directory=r1_speed_tracking_amp_smoke \
            agent.agent.experiment.experiment_name=seed42_env2_updates50 \
            > "${dir}/stdout.log" 2>&1
    else
        timeout --kill-after=30s 900s env R1_SPEED_METRICS_PATH="${metrics}" R1_SPEED_METRICS_FLUSH_STEPS=16 \
            "${PROJECT_ROOT}/scripts/r1_python.sh" -u "${PROJECT_ROOT}/scripts/r1_speed_tracking_train.py" \
            --task Isaac-Humanoid-AMP-Walk-Direct-v0 --algorithm PPO --num_envs 2 --max_iterations 50 \
            --seed 42 --headless --device cuda:0 \
            agent.agent.experiment.directory=r1_speed_tracking_ppo_smoke \
            agent.agent.experiment.experiment_name=seed42_env2_updates50 \
            > "${dir}/stdout.log" 2>&1
    fi
    local exit_code=$?
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    printf 'exit_code=%s\n' "${exit_code}" > "${dir}/exit.txt"
    date --iso-8601=seconds > "${dir}/finished_at.txt"
    printf '%s exit_code=%s\n' "${group}" "${exit_code}"
    return 0
}

run_group amp AMP
run_group ppo PPO
