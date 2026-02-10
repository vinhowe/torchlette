#!/bin/bash
# Script to run Torchlette tests in Docker
#
# Usage:
#   ./run-docker-tests.sh                    # Run all tests with software rendering
#   ./run-docker-tests.sh --gpu              # Run with GPU (requires nvidia-docker)
#   ./run-docker-tests.sh checkpoint         # Run checkpoint investigation tests
#   ./run-docker-tests.sh oracle             # Run PyTorch oracle comparison tests
#   ./run-docker-tests.sh <test-pattern>     # Run specific tests
#
# Examples:
#   ./run-docker-tests.sh test/amp-speed-verification.spec.ts
#   ./run-docker-tests.sh "test/*.spec.ts"

set -e

IMAGE_NAME="torchlette"
USE_GPU=false
TEST_PATTERN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            USE_GPU=true
            shift
            ;;
        checkpoint)
            TEST_PATTERN="test/checkpoint-scale-analysis.spec.ts test/checkpoint-memory-profile.spec.ts"
            shift
            ;;
        oracle)
            TEST_PATTERN="test/oracle/"
            shift
            ;;
        *)
            TEST_PATTERN="$1"
            shift
            ;;
    esac
done

echo "=== Building Docker image ==="
docker build -t "$IMAGE_NAME" .

echo ""
echo "=== Running tests ==="

if [ "$USE_GPU" = true ]; then
    echo "Mode: GPU-accelerated (nvidia-docker)"
    GPU_FLAGS="--gpus all"
else
    echo "Mode: Software rendering (Mesa llvmpipe)"
    GPU_FLAGS=""
fi

if [ -z "$TEST_PATTERN" ]; then
    echo "Tests: All tests"
    docker run --rm $GPU_FLAGS \
        -e TORCHLETTE_WEBGPU=1 \
        "$IMAGE_NAME" \
        npm test
else
    echo "Tests: $TEST_PATTERN"
    docker run --rm $GPU_FLAGS \
        -e TORCHLETTE_WEBGPU=1 \
        "$IMAGE_NAME" \
        npm test -- $TEST_PATTERN
fi
