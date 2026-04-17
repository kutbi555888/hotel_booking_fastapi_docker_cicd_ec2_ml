#!/usr/bin/env bash
set -e

python -m pip install --upgrade pip
python -m pip install pipreqs
pipreqs . --force --ignore .venv,data,models,reports,.pytest_cache,__pycache__

echo "requirements.txt pipreqs orqali yangilandi."
