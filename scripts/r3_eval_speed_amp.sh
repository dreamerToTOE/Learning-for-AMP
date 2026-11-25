#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 ]]; then
  echo "Usage: $0 ordinary|conditional fixed|held_out RUN_DIR CHECKPOINT_INDEX" >&2
  exit 2
fi
variant=$1
split=$2
run=$(realpath -e "$3")
index=$4

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
r2=$(realpath "$project/../r2")
r3=$(realpath "$project/../r3")
upstream="$r2/upstream/MimicKit"
cd "$upstream"
env -u PYTHONHOME PYTHONPATH="$upstream/mimickit" XDG_CACHE_HOME="$r3/cache" TORCH_COMPILE_DISABLE=1 \
  "$r2/mimickit-venv/bin/python" "$project/scripts/r3_eval_speed_amp.py" --data "$r3/assets/walk_jog" \
  --protocol "$run/speed_amp.json" --gait "$project/configs/r3/walk_jog_eval.json" --variant "$variant" \
  --checkpoint "$run/int_models/model_$index.pt" --split "$split" --output "$run/eval_${split}_$index.json" \
  > "$run/eval_${split}_$index.log" 2>&1
