#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1
mkdir -p runs
RUN_ROOT="$(mktemp -d "${PROJECT_ROOT}/runs/r1_speed_tracking_capacity.XXXXXX")" || exit 1
printf 'Results: %s\n' "${RUN_ROOT}"
status=0
for NUM_ENVS in 256 512 1024 2048; do
    RUN_DIR="${RUN_ROOT}/amp_env${NUM_ENVS}_updates20"
    mkdir -p "${RUN_DIR}"
    printf '%s\n' "algorithm=AMP seed=42 num_envs=${NUM_ENVS} updates=20 transitions=$((NUM_ENVS * 16 * 20)) target_vx=1.5 sigma_vx=0.5" > "${RUN_DIR}/command.txt"
    date --iso-8601=seconds > "${RUN_DIR}/started_at.txt"
    date +%s%N > "${RUN_DIR}/started_epoch_ns.txt"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "${RUN_DIR}/gpu_processes_start.csv" 2>/dev/null || true
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -lms 500 > "${RUN_DIR}/gpu.csv" 2>/dev/null &
    MONITOR_PID=$!
    timeout --kill-after=30s 900s env R1_SPEED_METRICS_PATH="${RUN_DIR}/speed_metrics.jsonl" R1_SPEED_METRICS_FLUSH_STEPS=16 \
        "${PROJECT_ROOT}/scripts/r1_python.sh" -u "${PROJECT_ROOT}/scripts/r1_speed_tracking_train.py" \
        --task Isaac-Humanoid-AMP-Walk-Direct-v0 --algorithm AMP --num_envs "${NUM_ENVS}" --max_iterations 20 \
        --seed 42 --headless --device cuda:0 \
        hydra.run.dir="${RUN_DIR}/hydra" \
        agent.agent.task_reward_weight=1.0 agent.agent.style_reward_weight=1.0 \
        agent.agent.experiment.directory=r1_speed_tracking_capacity \
        agent.agent.experiment.experiment_name=amp_env${NUM_ENVS}_updates20 \
        > "${RUN_DIR}/stdout.log" 2>&1
    EXIT_CODE=$?
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "${RUN_DIR}/gpu_processes_end.csv" 2>/dev/null || true
    printf 'exit_code=%s\n' "${EXIT_CODE}" > "${RUN_DIR}/exit.txt"
    date --iso-8601=seconds > "${RUN_DIR}/finished_at.txt"
    date +%s%N > "${RUN_DIR}/finished_epoch_ns.txt"
    printf 'envs=%s exit_code=%s\n' "${NUM_ENVS}" "${EXIT_CODE}"
    if (( EXIT_CODE )); then status=1; fi
done
exit "$status"
