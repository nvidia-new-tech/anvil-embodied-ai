# Setup Guide — inference on a new machine

For the agent doing the setup. Written 2026-09-11 from branch `single-pc-inference-fixes`
(`c04a384`) of `/srv/anvil/anvil-embodied-ai`.

## What is on the USB stick

| Path | What | Why it is here |
|---|---|---|
| `anvil-handoff/lerobot-inference-latest.tar.gz` | 4.8G — `docker save` of `ghcr.io/anvil-robotics/lerobot-inference:latest`, image id `sha256:70fbd34b…` | **Built locally, never pushed to ghcr.** `docker manifest inspect` on that tag returns `denied`, so `docker compose pull` will NOT work. This tarball is the only copy. |
| `smolvla_flip_pack_20260904_plus_005000/` (USB root, not under `anvil-handoff/`) | 865M SmolVLA checkpoint | Verified byte-identical to the source machine — `model.safetensors` md5 `f1d4d91ee1d6afd8fb3cd33305cf24b1`. |

Everything else — the repo, the HuggingFace cache — is re-downloadable and was deliberately
**not** copied. Don't go looking for it on the stick.

## Steps

### 1. Host prerequisites

Docker Engine + compose plugin, plus the NVIDIA container toolkit — `docker-compose.yml`
reserves `driver: nvidia, count: all`, so the stack won't start without it.

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER          # log out and back in

sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

The NVIDIA *driver* must already be on the host; the toolkit only exposes it. No host CUDA
toolkit needed — that lives inside the image.

### 2. Load the image

```bash
gunzip -c /media/<user>/U/anvil-handoff/lerobot-inference-latest.tar.gz | docker load
docker images ghcr.io/anvil-robotics/lerobot-inference
```

Confirm the id is `70fbd34b…`. Don't run `docker compose pull` — it fails with `denied` and
will make you think the image is missing.

Rebuilding instead is a valid fallback (`docker compose build`, ~15 min, needs network) —
but read the `LEROBOT_EXTRAS` warning below first.

### 3. Clone the repo

```bash
git clone https://github.com/anvil-robotics/anvil-embodied-ai.git
cd anvil-embodied-ai
```

`configs/lerobot_control/shapes/` **is** tracked in git as of 2026-09-11 — no need to copy
it. If `ls configs/lerobot_control/shapes/` comes back empty you're on an old commit.

`uv sync --all-packages` is optional — the container carries its own Python. Do it anyway,
because the preflight script in step 6 runs on the host.

### 4. Copy the checkpoint off the stick

```bash
mkdir -p ~/model_zoo
cp -r /media/<user>/U/smolvla_flip_pack_20260904_plus_005000 ~/model_zoo/
```

### 5. Write `.env`

`.env` is gitignored (`.gitignore:114`), so it doesn't come with the clone and isn't on the
stick. Build it from the template:

```bash
cp .env.example .env
```

Then set the following — these are the values the source machine ran:

```
MODEL_PATH=/home/<user>/model_zoo/smolvla_flip_pack_20260904_plus_005000
CONFIG_FILE=./configs/lerobot_control/shapes/1arm_2cam.yaml
HF_CACHE=/home/<user>/.cache/huggingface
LEROBOT_EXTRAS=pi,smolvla
HF_HUB_OFFLINE=1
ROS_DOMAIN_ID=0
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
CYCLONEDDS_URI=file:///workspace/configs/cyclonedds/single_pc.xml
```

The checkpoint is smolvla, single-arm 8-DOF, 2 cameras (chest + wrist_r), absolute actions —
hence `1arm_2cam.yaml`. **The config must match the checkpoint's shape, not its task.**

### 6. Populate the HuggingFace cache

SmolVLA loads the SmolVLM2 tokenizer at runtime, so this is required (ACT and Diffusion
wouldn't need it). On the source machine the cache held
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (2.7G) and `lerobot/smolvla_base` (1.2G).

Set `HF_HUB_OFFLINE=` (empty) for the **first** run so the download can happen, then set it
back to `1` once the cache is warm. Leaving it at `1` against an empty cache gives a
confusing stall rather than a clean error.

> If you ever hand-copy an HF cache between machines, don't copy it onto exFAT. The cache
> symlinks `snapshots/` into `blobs/`, and exFAT can't store symlinks — that's why it isn't
> on the stick. Use `tar`, or `cp -rL` to dereference.

### 7. Preflight before the arm is powered

```bash
uv run scripts/preflight_checkpoint.py "$MODEL_PATH" --config "$CONFIG_FILE"
```

**Exit code 1 means do not launch.** This matters more than it sounds: every way a checkpoint
and its inference config can disagree fails *silently* — nothing raises, nothing warns, and
the arm still moves plausibly. A camera the checkpoint declares but the config omits is
padded with a blank −1 image and mask 0, and the policy decays toward a near-constant prior
pose. The full table of failure modes is in `docs/inference.md`.

Verify DDS separately, with no GPU or model:

```bash
./scripts/run_inference.sh --echo-topic-only up
```

### 8. Run

```bash
./scripts/run_inference.sh up --build          # production
./scripts/run_inference.sh --monitor-enable up # + per-step CSV/PNG report
./scripts/run_inference.sh --fake-hardware --monitor-enable up   # no robot
```

Use `--monitor-enable` on the first real run.

## Traps specific to a fresh machine

**`LEROBOT_EXTRAS` is baked into an image layer at build time**, not installed at runtime.
The tarball was built with `pi,smolvla`. Rebuild with a different value and those policies
silently won't be there. Rebuild the image after any change to it.

**`MODEL_PATH` must be absolute or start with `./`.** A bare relative path is read by Docker
as a *named volume* and mounts empty — with no error.

**`HF_CACHE` needs write access**, not just read — HuggingFace creates lock files when
loading tokenizers.

**DDS must match anvil-loader on all three settings** — `ROS_DOMAIN_ID`,
`RMW_IMPLEMENTATION`, `CYCLONEDDS_URI`. Mixing Fast DDS and CycloneDDS means no discovery,
reported as silence rather than an error. If this box is the GPU PC with the robot on a
separate machine, switch to `two_pc_gpu.xml` and set `CYCLONEDDS_PEER_IP=<gpu_pc_ip>` on the
anvil-loader side.

**`network_mode: host` + `privileged: true`** mean a host firewall blocking DDS multicast
(udp 7400–7500) breaks discovery with no message.

**`DISPLAY` is passed through** for X11. On a headless box leave it unset rather than
pointing it somewhere invalid.

## Recommended follow-up

Push the image to ghcr so the next machine needs no USB at all:

```bash
echo $GITHUB_PAT | docker login ghcr.io -u <gh-user> --password-stdin
docker push ghcr.io/anvil-robotics/lerobot-inference:latest
```

The PAT needs `write:packages`. Tag by extras (e.g. `:pi-smolvla`) rather than relying on
`latest`, given the baked-in-extras problem above.

Don't put the image in git — GitHub rejects files over 100MB, and LFS would store another
4.8GB blob per rebuild, forever. The Dockerfile and compose file are already tracked; those
are the parts that belong in git.
