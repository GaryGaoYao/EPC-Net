#!/usr/bin/env bash
# Full EPC-Net training pipeline (Linux / macOS).
#   CONFIG=configs/my.yaml bash scripts/train_epcnet.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/local_dataset777.yaml}
PY=${PYTHON:-python}

$PY -m epcnet preprocess -c "$CONFIG"
SPLIT=$($PY -c "from epcnet.config import load_config; print(load_config(['$CONFIG'])['data']['split_file'])")
[ -f "$SPLIT" ] || $PY -m epcnet split -c "$CONFIG" --n-val 5 --n-test 5
$PY -m epcnet train --stage 1 -c "$CONFIG"
$PY -m epcnet priors -c "$CONFIG"
$PY -m epcnet train --stage 2 -c "$CONFIG"
$PY -m epcnet evaluate --split test --save-predictions -c "$CONFIG"
$PY -m epcnet export -c "$CONFIG"
