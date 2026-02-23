# Dockerfile for running Torchlette WebGPU tests + PyTorch oracle tests
#
# Based on PyTorch nightly for oracle tests, adds Node.js for TypeScript tests.
# Includes a custom Dawn build with VulkanEnableF16OnNvidia toggle for shader-f16
# support on NVIDIA GPUs.
#
# Usage:
#   docker build -t torchlette .
#   docker run --rm torchlette npm test
#   docker run --rm torchlette npm test -- test/oracle/
#   docker run -it torchlette bash

# ============================================================================
# Stage 1: Build Dawn from source with NVIDIA f16 toggle
# ============================================================================
# Dawn's webgpu npm package (v0.3.8, Sep 2025) predates the
# VulkanEnableF16OnNvidia toggle (Oct 27, 2025). We build from latest source
# to get shader-f16 support on NVIDIA Vulkan GPUs.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS dawn-builder

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies for Dawn
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    ca-certificates \
    cmake \
    ninja-build \
    python3 \
    # Vulkan headers/loader for Dawn's Vulkan backend
    libvulkan-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Go (required by Dawn's code generators)
RUN wget -q https://go.dev/dl/go1.22.10.linux-amd64.tar.gz -O /tmp/go.tar.gz \
    && tar -C /usr/local -xzf /tmp/go.tar.gz \
    && rm /tmp/go.tar.gz
ENV PATH="/usr/local/go/bin:$PATH"

# Install Node.js headers (required for dawn.node bindings)
RUN apt-get update && apt-get install -y nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Clone Dawn source (shallow)
RUN git clone --depth=1 https://dawn.googlesource.com/dawn /tmp/dawn-src

# Fetch Dawn's third-party dependencies
RUN python3 /tmp/dawn-src/tools/fetch_dawn_dependencies.py --directory /tmp/dawn-src --shallow

# Fetch dawn_node-specific deps not covered by fetch_dawn_dependencies.py
RUN mkdir -p /tmp/dawn-src/third_party/gpuweb \
    && cd /tmp/dawn-src/third_party/gpuweb \
    && git init \
    && git remote add origin https://chromium.googlesource.com/external/github.com/gpuweb/gpuweb \
    && git fetch --depth=1 origin b4b5752ff755fe33bf6a67fb6e5964ba9d40dcdc \
    && git checkout FETCH_HEAD

RUN mkdir -p /tmp/dawn-src/third_party/node-api-headers \
    && cd /tmp/dawn-src/third_party/node-api-headers \
    && git init \
    && git remote add origin https://chromium.googlesource.com/external/github.com/nodejs/node-api-headers \
    && git fetch --depth=1 origin d5cfe19da8b974ca35764dd1c73b91d57cd3c4ce \
    && git checkout FETCH_HEAD

RUN mkdir -p /tmp/dawn-src/third_party/node-addon-api \
    && cd /tmp/dawn-src/third_party/node-addon-api \
    && git init \
    && git remote add origin https://chromium.googlesource.com/external/github.com/nodejs/node-addon-api \
    && git fetch --depth=1 origin 1e26dcb52829a74260ec262edb41fc22998669b6 \
    && git checkout FETCH_HEAD

# Configure Dawn with CMake
RUN mkdir -p /tmp/dawn-build && cd /tmp/dawn-build \
    && cmake /tmp/dawn-src -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DDAWN_BUILD_NODE_BINDINGS=1 \
        -DDAWN_ENABLE_VULKAN=1 \
        -DDAWN_USE_X11=OFF \
        -DDAWN_USE_WAYLAND=OFF

# Build dawn.node (~1097 targets, takes ~5-10 min)
RUN cd /tmp/dawn-build && ninja dawn.node -j$(nproc)

# ============================================================================
# Stage 2: Final image
# ============================================================================
# PyTorch 2.5.1 with CUDA 12.1 — compatible with host driver 545.x (CUDA 12.3).
# cu121 is the newest CUDA toolkit that driver 545 supports.
# See: https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for WebGPU (Dawn/Vulkan)
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    # Editor
    neovim \
    # Vulkan/Mesa for WebGPU (Lavapipe software fallback + loader)
    libvulkan1 \
    mesa-vulkan-drivers \
    libegl1-mesa \
    libgl1-mesa-dri \
    libgles2-mesa \
    # NVIDIA Vulkan ICD — provides nvidia_icd.json + userspace GL/Vulkan libs.
    # nvidia-container-toolkit mounts host driver libs (.so.VERSION) into the
    # container. The apt-installed version may differ from the host kernel module.
    # The entrypoint script below fixes symlinks to match the host driver version.
    libnvidia-gl-545 \
    # X11 libs that Dawn may need
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxkbcommon0 \
    libdrm2 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install pnpm
RUN npm install -g pnpm@9

# Set up working directory
WORKDIR /app

# Copy package files first (for layer caching)
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./

# Install Node.js dependencies
RUN pnpm install --frozen-lockfile

# Replace dawn.node with custom build that has VulkanEnableF16OnNvidia toggle
COPY --from=dawn-builder /tmp/dawn-build/dawn.node \
    /app/node_modules/webgpu/dist/linux-x64.dawn.node

# Copy source code
COPY . .

# Build TypeScript project
RUN pnpm build

# Environment variables
ENV TORCHLETTE_WEBGPU=1
ENV TORCH_ORACLE_PYTHON=python
# Claude Code path (install once via: curl -fsSL https://claude.ai/install.sh | bash)
ENV PATH="/root/.local/bin:$PATH"

# Fix NVIDIA Vulkan: nvidia-container-toolkit mounts host driver libs (e.g.
# libnvidia-glcore.so.545.23.08) but libnvidia-gl-545 installs a potentially
# different version (e.g. 545.29.06) and points symlinks at it. The userspace
# lib version MUST match the kernel module or ioctls to /dev/nvidiactl fail
# with EINVAL, causing Vulkan init to silently return NULL for all entry points.
# This script detects the host kernel module version and repoints symlinks.
RUN printf '#!/bin/bash\n\
HOST_VER=$(cat /proc/driver/nvidia/version 2>/dev/null | grep -oP "Module\\s+\\K[0-9.]+" | head -1)\n\
if [ -n "$HOST_VER" ] && [ -f "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.$HOST_VER" ]; then\n\
  for lib in libGLX_nvidia libEGL_nvidia; do\n\
    ln -sf "${lib}.so.${HOST_VER}" "/usr/lib/x86_64-linux-gnu/${lib}.so.0" 2>/dev/null\n\
  done\n\
fi\n\
exec "$@"\n' > /usr/local/bin/fix-nvidia-vulkan.sh && chmod +x /usr/local/bin/fix-nvidia-vulkan.sh

ENTRYPOINT ["/usr/local/bin/fix-nvidia-vulkan.sh"]

# NOTE: When running with --gpus and NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,
# Dawn uses the NVIDIA Vulkan ICD for hardware-accelerated WebGPU.
# For explicit software-only fallback, set at runtime:
#   -e LIBGL_ALWAYS_SOFTWARE=1 -e GALLIUM_DRIVER=llvmpipe

CMD ["bash"]
