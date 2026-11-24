#!/usr/bin/env bash
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_TRAIN="${PROJECT_ROOT}/../r1/IsaacLab-v2.1.1/scripts/reinforcement_learning/skrl/train.py"
RUN_ROOT="${PROJECT_ROOT}/runs/r1_capacity/20261124_high_env_sweep"
LOG_ROOT="r1_capacity_sweep_high_20261124"
mkdir -p "${RUN_ROOT}"
for NUM_ENVS in 512 1024 2048; do
    RUN_DIR="${RUN_ROOT}/env${NUM_ENVS}_updates50"
    mkdir -p "${RUN_DIR}"
    COMMAND="scripts/r1_python.sh -u ${LAB_TRAIN} --task Isaac-Humanoid-AMP-Walk-Direct-v0 --algorithm AMP --num_envs ${NUM_ENVS} --max_iterations 50 --seed 42 --headless --device cuda:0 agent.agent.experiment.directory=${LOG_ROOT} agent.agent.experiment.experiment_name=seed42_env${NUM_ENVS}_updates50"
    printf '%s\n' "${COMMAND}" > "${RUN_DIR}/command.txt"
    date --iso-8601=seconds > "${RUN_DIR}/started_at.txt"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -lms 500 > "${RUN_DIR}/gpu.csv" 2>/dev/null &
    MONITOR_PID=$!
    timeout --kill-after=30s 900s bash -lc "cd '${PROJECT_ROOT}' && ${COMMAND}" > "${RUN_DIR}/stdout.log" 2>&1
    EXIT_CODE=$?
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    printf 'exit_code=%s\n' "${EXIT_CODE}" > "${RUN_DIR}/exit.txt"
    date --iso-8601=seconds > "${RUN_DIR}/finished_at.txt"
    printf 'envs=%s exit_code=%s\n' "${NUM_ENVS}" "${EXIT_CODE}"
done
