#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R1_WORK_ROOT="${PROJECT_ROOT}/../r1"
R1_VENV="${R1_WORK_ROOT}/venv"
LOCK_FILE="${PROJECT_ROOT}/configs/r1/requirements-venv.lock"

mkdir -p "${R1_WORK_ROOT}"
/usr/bin/python3 -m venv "${R1_VENV}"
"${R1_VENV}/bin/python" -m pip install --upgrade pip==24.3.1 wheel==0.45.1
"${R1_VENV}/bin/python" -m pip install --no-deps --no-build-isolation -r "${LOCK_FILE}"

echo "Created isolated venv at ${R1_VENV}"
echo "Use scripts/r1_python.sh to launch Isaac Sim with this venv."
