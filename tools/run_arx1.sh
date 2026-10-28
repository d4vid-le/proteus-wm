#!/usr/bin/env bash
# Run a batch of arXiv queries sequentially with pacing to avoid rate limits.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run(){
  bash "$HERE/arx_search.sh" "$1" "${2:-15}"
  sleep 4
}

run '"residual adapter continual learning"' 15
run '"adapter fusion continual learning"' 15
run '"task adapters incremental learning"' 15
run 'LoRA continual learning sequential tasks' 15
run 'O-LoRA' 10
run 'Interpolation LoRA continual learning' 10
