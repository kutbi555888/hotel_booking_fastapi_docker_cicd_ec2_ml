#!/usr/bin/env bash
set -e

IMAGE_NAME="hotel-booking-api:latest"
CONTAINER_NAME="hotel-booking-api"

sudo docker build -t "$IMAGE_NAME" .
sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
sudo docker run -d \
  --name "$CONTAINER_NAME" \
  --restart always \
  -p 8000:8000 \
  "$IMAGE_NAME"

echo "Container ishga tushdi."
sudo docker ps
