#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-5011}"
CONTAINER_NAME="${CONTAINER_NAME:-ds2api-structured-fail-closed-mock}"
GO_IMAGE="${GO_IMAGE:-golang:1.26}"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup

docker run -d --rm \
  --name "$CONTAINER_NAME" \
  -p "${PORT}:${PORT}" \
  -e "PORT=${PORT}" \
  -v "$ROOT_DIR:/app" \
  -w /app \
  "$GO_IMAGE" \
  go run ./scripts/structured_output_fail_closed_server.go >/dev/null

python3 "$ROOT_DIR/scripts/test_openai_sdk_langchain.py" \
  --base-url "http://127.0.0.1:${PORT}" \
  --api-key proxypal-local \
  --surface fail-closed

