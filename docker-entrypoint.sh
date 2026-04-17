#!/usr/bin/env sh
set -e

if [ ! -f models/best_model.joblib ]; then
  echo "best_model.joblib topilmadi. Pipeline ishga tushirilmoqda..."
  python scripts/run_all.py
else
  echo "Tayyor best_model.joblib topildi. API ishga tushirilmoqda..."
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
