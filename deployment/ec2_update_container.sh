#!/usr/bin/env bash
set -e

if [ ! -d .git ]; then
  echo "Bu script repository root ichidan ishga tushirilishi kerak."
  exit 1
fi

git pull
chmod +x deployment/ec2_run_container.sh
./deployment/ec2_run_container.sh
