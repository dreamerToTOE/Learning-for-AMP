#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R1_WORK_ROOT="${PROJECT_ROOT}/../r1"
ISAACLAB_R1_ROOT="${R1_WORK_ROOT}/IsaacLab-v2.1.1"
R1_VENV="${R1_WORK_ROOT}/venv"
ISAACSIM_ROOT="/home/ubuntu2004/isaacsim-4.5.0"

export ISAACLAB_R1_ROOT
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="${R1_VENV}/lib/python3.10/site-packages:${ISAACLAB_R1_ROOT}/source/isaaclab:${ISAACLAB_R1_ROOT}/source/isaaclab_rl:${ISAACLAB_R1_ROOT}/source/isaaclab_tasks:${ISAACLAB_R1_ROOT}/source/isaaclab_assets"

exec "${ISAACSIM_ROOT}/python.sh" "$@"
