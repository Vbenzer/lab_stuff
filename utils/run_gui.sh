#!/bin/bash
cd "$(dirname "$0")"
source ../.venv1/bin/activate

# Add the project root to PYTHONPATH
export PYTHONPATH=$(realpath ..)

python3 ../gui/main_window.py