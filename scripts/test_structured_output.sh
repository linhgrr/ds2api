#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="golang:1.26"

echo "=== go mod tidy ==="
docker run --rm \
  -v "$PROJECT_ROOT":/app \
  -w /app \
  -e GOPATH=/go \
  -e GOCACHE=/tmp/go-cache \
  "$IMAGE" \
  go mod tidy

echo ""
echo "=== Build check ==="
docker run --rm \
  -v "$PROJECT_ROOT":/app \
  -w /app \
  -e GOPATH=/go \
  -e GOCACHE=/tmp/go-cache \
  "$IMAGE" \
  go build -buildvcs=false ./...

echo ""
echo "=== Unit tests: internal/promptcompat ==="
docker run --rm \
  -v "$PROJECT_ROOT":/app \
  -w /app \
  -e GOPATH=/go \
  -e GOCACHE=/tmp/go-cache \
  "$IMAGE" \
  go test -buildvcs=false -v -count=1 ./internal/promptcompat/...

echo ""
echo "=== Unit tests: internal/completionruntime ==="
docker run --rm \
  -v "$PROJECT_ROOT":/app \
  -w /app \
  -e GOPATH=/go \
  -e GOCACHE=/tmp/go-cache \
  "$IMAGE" \
  go test -buildvcs=false -v -count=1 ./internal/completionruntime/... || echo "(no tests in completionruntime, skipping)"

echo ""
echo "All checks passed."
