#!/usr/bin/env bash
set -e

: "${AWS_REGION:?AWS_REGION kerak}"
: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID kerak}"
: "${ECR_REPOSITORY:?ECR_REPOSITORY kerak}"

IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="hotel-booking-api"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker pull "$IMAGE_URI"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER_NAME" --restart always -p 80:8000 "$IMAGE_URI"
docker ps
