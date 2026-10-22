#!/usr/bin/env bash
# Reproduce every artifact and document in this project.
#
# Usage:  ./scripts/run_all.sh
#
# Expects the datasets under data/raw/ (a symlink to the source corpus).
# Override the interpreter with PYTHON=/path/to/python if python3 on your
# PATH is not the environment holding TensorFlow.

set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> interpreter"
"$PYTHON" -c "import sys, tensorflow, sklearn; print(sys.executable); print('tensorflow', tensorflow.__version__); print('scikit-learn', sklearn.__version__)"

echo
echo "==> 1/5 exploratory analysis"
"$PYTHON" scripts/eda.py

echo
echo "==> 2/5 symptom models"
"$PYTHON" scripts/train_symptom.py --splits 5 --repeats 6

echo
echo "==> 3/5 eye image CNN"
"$PYTHON" scripts/train_eye.py --epochs 40 --augment 0.5 --subject-split

echo
echo "==> 4/5 blink CNN (slowest step, roughly 45-60 minutes on CPU)"
"$PYTHON" scripts/train_blink.py --full --epochs 16 --augment 0.3 --patience 6

echo
echo "==> 5/5 documents"
"$PYTHON" scripts/make_report.py

echo
echo "==> tests"
"$PYTHON" -m pytest

echo
echo "Done. Launch the dashboard with:  $PYTHON -m streamlit run app/app.py"