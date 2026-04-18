#!/usr/bin/env sh
set -e

if [ ! -f models/best_model.joblib ]; then
  echo "best_model.joblib topilmadi. Image ichida inference uchun model bo‘lishi shart."
  exit 1
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
