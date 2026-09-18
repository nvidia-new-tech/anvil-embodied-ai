# syntax=docker/dockerfile:1.7
# Use ROS2 Jazzy as base.
#
# Pinned by digest on purpose. `ros:jazzy-ros-base` is a moving tag — the ROS
# images are rebuilt regularly — and an unpinned FROM re-resolves on every
# build. A new digest invalidates the base layer, and because layers stack,
# EVERYTHING below it rebuilds, including the ~9 GB pip install. That is why a
# build with an untouched Dockerfile could still redownload every package.
#
# Pinning also makes the build reproducible across machines: two hosts building
# the same commit now start from the same bytes.
#
# To move to a newer base, deliberately:
#   docker buildx imagetools inspect ros:jazzy-ros-base   # read the new Digest
# then update the line below. Expect a full rebuild when you do.
FROM ros:jazzy-ros-base@sha256:386d06ec6d4188f731bae5678e07b4cb64a4e4d4152090c0bd1f881dcf7706f5

ARG LEROBOT_VERSION=0.5.1
ARG LEROBOT_EXTRAS=""

# Set environment
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create workspace
WORKDIR /workspace

# Base lerobot install — cached as long as version doesn't change
# transformers + scipy are required for VLA models (pi0, pi0.5, smolvla)
#
# The BuildKit cache mount replaces --no-cache-dir. Both keep the downloaded
# wheels out of the image, but the mount also keeps them on the host, so when
# this layer IS invalidated pip installs from disk instead of refetching ~9 GB
# over the network. The cache lives outside the image and never ships with it.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip3 install --break-system-packages --ignore-installed \
    numpy packaging "lerobot==${LEROBOT_VERSION}" matplotlib \
    "transformers==5.3.0" "scipy>=1.14.0,<2.0.0" && \
    pip3 install --break-system-packages "setuptools<80"

# Extras — separate layer, only rebuilt when LEROBOT_EXTRAS changes
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ -n "${LEROBOT_EXTRAS}" ]; then \
        pip3 install --break-system-packages \
            "lerobot[${LEROBOT_EXTRAS}]==${LEROBOT_VERSION}"; \
    fi

# Patch lerobot v0.5.0 bug: groot_n1.py has a malformed @dataclass that crashes
# on import. Remove groot lines from policies/__init__.py since we don't use GR00T.
RUN find /usr/local/lib -path "*/lerobot/policies/__init__.py" -exec sed -i '/groot/d' {} \;

# Copy ROS2 packages (changes often → late layer)
COPY ros2/src/anvil_msgs /workspace/src/anvil_msgs
COPY ros2/src/lerobot_control /workspace/src/lerobot_control

# Copy configs (lerobot_control + cyclonedds)
COPY configs /workspace/configs

# Build ROS2 workspace
RUN . /opt/ros/jazzy/setup.sh && \
    cd /workspace && \
    colcon build --symlink-install

# Set entrypoint
COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["ros2", "launch", "lerobot_control", "inference.launch.py"]
