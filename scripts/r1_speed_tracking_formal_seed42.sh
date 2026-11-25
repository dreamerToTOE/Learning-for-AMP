#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1
mkdir -p runs
RUN_ROOT="$(mktemp -d "${PROJECT_ROOT}/runs/r1_speed_tracking_formal_seed42.XXXXXX")" || exit 1
printf 'Results: %s\n' "${RUN_ROOT}"
NUM_ENVS=2048
UPDATES=1221
TRANSITIONS=$((NUM_ENVS * 16 * UPDATES))

run_group() {
    local group="$1"
    local algorithm="$2"
    local dir="${RUN_ROOT}/${group}"
    mkdir -p "${dir}"
    printf '%s\n' "algorithm=${algorithm} seed=42 num_envs=${NUM_ENVS} updates=${UPDATES} transitions=${TRANSITIONS} target_vx=1.5 sigma_vx=0.5 checkpoint_interval_trainer_timesteps=80" > "${dir}/command.txt"
    date --iso-8601=seconds > "${dir}/started_at.txt"
    date +%s%N > "${dir}/started_epoch_ns.txt"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "${dir}/gpu_processes_start.csv" 2>/dev/null || true
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -lms 500 > "${dir}/gpu.csv" 2>/dev/null &
    MONITOR_PID=$!
    if [[ "${algorithm}" == "AMP" ]]; then
        timeout --kill-after=30s 1800s env R1_SPEED_METRICS_PATH="${dir}/speed_metrics.jsonl" R1_SPEED_METRICS_FLUSH_STEPS=16 \
            "${PROJECT_ROOT}/scripts/r1_python.sh" -u "${PROJECT_ROOT}/scripts/r1_speed_tracking_train.py" \
            --task Isaac-Humanoid-AMP-Walk-Direct-v0 --algorithm AMP --num_envs "${NUM_ENVS}" --max_iterations "${UPDATES}" \
            --seed 42 --headless --device cuda:0 \
            hydra.run.dir="${dir}/hydra" \
            agent.agent.task_reward_weight=1.0 agent.agent.style_reward_weight=1.0 \
            agent.agent.experiment.directory=r1_speed_tracking_formal_amp_seed42 \
            agent.agent.experiment.experiment_name=env${NUM_ENVS}_updates${UPDATES} \
            agent.agent.experiment.checkpoint_interval=80 agent.agent.experiment.write_interval=80 \
            > "${dir}/stdout.log" 2>&1
    else
        timeout --kill-after=30s 1800s env R1_SPEED_METRICS_PATH="${dir}/speed_metrics.jsonl" R1_SPEED_METRICS_FLUSH_STEPS=16 \
            "${PROJECT_ROOT}/scripts/r1_python.sh" -u "${PROJECT_ROOT}/scripts/r1_speed_tracking_train.py" \
            --task Isaac-Humanoid-AMP-Walk-Direct-v0 --algorithm PPO --num_envs "${NUM_ENVS}" --max_iterations "${UPDATES}" \
            --seed 42 --headless --device cuda:0 \
            hydra.run.dir="${dir}/hydra" \
            agent.agent.experiment.directory=r1_speed_tracking_formal_ppo_seed42 \
            agent.agent.experiment.experiment_name=env${NUM_ENVS}_updates${UPDATES} \
            agent.agent.experiment.checkpoint_interval=80 agent.agent.experiment.write_interval=80 \
            > "${dir}/stdout.log" 2>&1
    fi
    EXIT_CODE=$?
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "${dir}/gpu_processes_end.csv" 2>/dev/null || true
    date --iso-8601=seconds > "${dir}/finished_at.txt"
    date +%s%N > "${dir}/finished_epoch_ns.txt"
    printf 'exit_code=%s\n' "${EXIT_CODE}" > "${dir}/exit.txt"
    printf '%s exit_code=%s\n' "${group}" "${EXIT_CODE}"
    return "$EXIT_CODE"
}

status=0
run_group amp AMP || status=1
run_group ppo PPO || status=1
exit "$status"
