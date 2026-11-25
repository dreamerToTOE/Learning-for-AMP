#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 && $# != 4 ]]; then
  echo "Usage: $0 PREPARED_DATA RUN_DIR MAX_ENVIRONMENT_STEPS [INITIAL_MODEL]" >&2
  exit 2
fi

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
r2=$(realpath "$project/../r2")
r3=$(realpath "$project/../r3")
upstream="$r2/upstream/MimicKit"
data=$(realpath -e "$1")
run=$(realpath -m "$2")
budget=$3
model_args=()
if [[ $# == 4 ]]; then
  initial_model=$(realpath -e "$4")
  model_args=(--model_file "$initial_model")
fi
mkdir "$run"

git -C "$upstream" rev-parse HEAD > "$run/upstream_commit.txt"
git -C "$upstream" diff > "$run/upstream.patch"
cp -- "$data/audit.json" "$data/motions.yaml" "$run/"
if [[ ${#model_args[@]} != 0 ]]; then
  sha256sum "$initial_model" > "$run/initial_model.sha256"
fi
cd "$upstream"

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv > "$run/gpu_before.csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv,nounits \
  --loop-ms=500 > "$run/gpu.csv" &
sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true' EXIT

set +e
/usr/bin/time -f '%e' -o "$run/wall_seconds.txt" timeout --kill-after=10s 90m \
  env -u PYTHONPATH -u PYTHONHOME XDG_CACHE_HOME="$r3/cache" PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 \
  "$r2/mimickit-venv/bin/python" mimickit/run.py --mode train --arg_file args/amp_g1_args.txt \
  --engine_config data/engines/newton_engine.yaml --env_config "$data/amp_g1_walk_jog_env.yaml" \
  --agent_config "$data/amp_g1_walk_jog_agent.yaml" --devices cuda:0 --visualize false --video false \
  --num_envs 128 --rand_seed 42 --max_samples "$budget" --save_int_models true --out_dir "$run" "${model_args[@]}" \
  > "$run/stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$run/exit_code.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv > "$run/gpu_after.csv"
exit "$status"
