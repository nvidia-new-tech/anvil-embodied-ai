# Thor-specific ROS2 Jazzy inference image.
# Keeps the upstream inference flow, but pins PyTorch to cu130 wheels that
# execute CUDA kernels on NVIDIA Thor / sm_110.
FROM ros:jazzy-ros-base

ARG LEROBOT_VERSION=0.5.1
ARG LEROBOT_EXTRAS=""
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.10.0+cu130
ARG TORCHVISION_VERSION=0.25.0+cu130

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Install Thor-compatible PyTorch first. LeRobot's dependency range accepts
# torch>=2.2.1,<2.11.0 and torchvision>=0.22.0,<0.26.0, so these cu130 wheels
# satisfy the resolver while preserving sm_110 CUDA kernel support.
RUN pip3 install --break-system-packages --no-cache-dir \
    --index-url "${PYTORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"

# Replace Debian-managed Python packaging tools with pip-managed versions.
# Without this, pip may fail while trying to uninstall Debian's packaging wheel
# because it has no RECORD file.
RUN pip3 install --break-system-packages --no-cache-dir --ignore-installed \
    "packaging==25.0" \
    "setuptools<80" \
    "numpy==2.2.6"

# Base LeRobot install. Do not ignore installed packages here; torch/torchvision
# must remain the cu130 builds installed above.
# transformers + scipy are required for VLA models (pi0, pi0.5, smolvla).
RUN pip3 install --break-system-packages --no-cache-dir \
    "lerobot==${LEROBOT_VERSION}" matplotlib \
    "transformers==5.3.0" "scipy>=1.14.0,<2.0.0"

RUN if [ -n "${LEROBOT_EXTRAS}" ]; then \
        pip3 install --break-system-packages --no-cache-dir \
            "lerobot[${LEROBOT_EXTRAS}]==${LEROBOT_VERSION}"; \
    fi

# Patch lerobot v0.5.0 bug: groot_n1.py has a malformed @dataclass that crashes
# on import. Remove groot lines from policies/__init__.py since we don't use GR00T.
RUN find /usr/local/lib -path "*/lerobot/policies/__init__.py" -exec sed -i '/groot/d' {} \;

COPY ros2/src/lerobot_control /workspace/src/lerobot_control
COPY configs /workspace/configs

RUN . /opt/ros/jazzy/setup.sh && \
    cd /workspace && \
    colcon build --symlink-install

COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["ros2", "launch", "lerobot_control", "inference.launch.py"]
