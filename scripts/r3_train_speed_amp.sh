#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
  echo "Usage: $0 ordinary|conditional smoke|formal RUN_DIR" >&2
  exit 2
fi
variant=$1
stage=$2
smoke_args=()
case "$stage" in
  smoke) smoke_args=(--smoke) ;;
  formal) ;;
  *) exit 2 ;;
esac

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
r2=$(realpath "$project/../r2")
r3=$(realpath "$project/../r3")
upstream="$r2/upstream/MimicKit"
data="$r3/assets/walk_jog"
run=$(realpath -m "$3")
mkdir "$run"
git -C "$upstream" rev-parse HEAD > "$run/upstream_commit.txt"
git -C "$upstream" diff > "$run/upstream.patch"
git -C "$project" rev-parse HEAD > "$run/project_commit.txt"
git -C "$project" status --short > "$run/project_status.txt"
cp -- "$data"/{audit.json,motions.yaml} "$project/configs/r3/speed_amp.json" \
  "$upstream/data/engines/newton_engine.yaml" "$run/"

cd "$upstream"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv > "$run/gpu_before.csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv,nounits \
  --loop-ms=500 > "$run/gpu.csv" &
sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true' EXIT

set +e
/usr/bin/time -f '%e' -o "$run/wall_seconds.txt" timeout --kill-after=10s 90m \
  env -u PYTHONHOME PYTHONPATH="$upstream/mimickit" XDG_CACHE_HOME="$r3/cache" \
  PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 \
  "$r2/mimickit-venv/bin/python" "$project/scripts/r3_train_speed_amp.py" --protocol "$run/speed_amp.json" \
  --data "$data" --variant "$variant" --output "$run" "${smoke_args[@]}" > "$run/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$run/exit_code.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv > "$run/gpu_after.csv"
exit "$status"
