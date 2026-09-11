#!/usr/bin/env bash
# Entry point for ALL runtime inference scenarios.
# Handles directory ownership for monitor output and auto-plots CSV on exit.
#
# Usage:
#   ./scripts/run_inference.sh [OPTIONS] [COMPOSE_ARGS...]
#
# Options:
#   --fake-hardware    Use docker-compose.fake-hardware.yml (DDS bridge test, no real robot)
#   --dev-src          Mount ros2/src over the image's copy, so the working tree runs
#                      without a rebuild (see docker-compose.devsrc.yml)
#   --monitor-enable          Enable monitor profile; for production also sets MONITOR_ENABLE=true,
#                      pre-creates MONITOR_OUTPUT_DIR as current user, and plots CSV on exit
#   --echo-topic-only  Subscribe + log FPS without loading a model (sets ECHO_TOPIC_ONLY=true);
#                      useful to verify DDS connectivity on the GPU PC without a checkpoint
#   --debug            Enable debug metrics: action smoothness, queue depth, Action FPS (sets DEBUG=true)
#   --no-benchmark     Don't start the benchmark scoring web tool alongside `up`
#   -h, --help         Show this message
#
# All other arguments (e.g. up --build, down, logs) are passed directly to docker compose.
#
# Environment variables:
#   MONITOR_OUTPUT_DIR   Host dir for monitor CSV/PNG (default: ./monitor_output)
#   MODEL_PATH           Path to model checkpoint (required for production inference)
#   CONFIG_FILE          Path to inference config YAML (default: ./configs/lerobot_control/inference_default.yaml)
#   IMAGE_TAG            Docker image tag (default: latest)
#   ROS_DOMAIN_ID        ROS domain ID
#   HF_CACHE             HuggingFace cache dir (needed for VLA models)
#   BENCHMARK_PORT       Port for the benchmark scoring tool (default: 8777)
#
# Examples:
#   # Production inference (real robot), no monitor
#   MODEL_PATH=/path/to/checkpoint ./scripts/run_inference.sh up --build
#
#   # Production inference with real-time monitor + auto-plot
#   MODEL_PATH=/path/to/checkpoint ./scripts/run_inference.sh --monitor-enable up --build
#
#   # Fake-hardware DDS test (FPS monitor only, no GPU)
#   ./scripts/run_inference.sh --fake-hardware --monitor-enable up --build
#
#   # Verify DDS connectivity without a model (no MODEL_PATH needed)
#   ./scripts/run_inference.sh --echo-topic-only up --build
#
#   # Fake-hardware full inference pipeline
#   # NOTE: --profile is a docker compose global flag and must come BEFORE the subcommand.
#   MODEL_PATH=/path/to/checkpoint ./scripts/run_inference.sh --fake-hardware --profile inference up --build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
COMPOSE_FILE="${REPO_ROOT}/docker-compose.yml"
DEV_SRC=false
FAKE_HARDWARE=false
MONITOR_REQUESTED=false
ECHO_TOPIC_ONLY_REQUESTED=false
DEBUG_REQUESTED=false
BENCHMARK_REQUESTED=true
PASSTHROUGH=()

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \?//; p }' "$0"
}

# Parse our flags; collect everything else for docker compose
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev-src)
            DEV_SRC=true
            shift
            ;;
        --fake-hardware)
            FAKE_HARDWARE=true
            COMPOSE_FILE="${REPO_ROOT}/docker-compose.fake-hardware.yml"
            shift
            ;;
        --monitor-enable)
            MONITOR_REQUESTED=true
            shift
            ;;
        --echo-topic-only)
            ECHO_TOPIC_ONLY_REQUESTED=true
            shift
            ;;
        --debug)
            DEBUG_REQUESTED=true
            shift
            ;;
        --no-benchmark)
            BENCHMARK_REQUESTED=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            PASSTHROUGH+=("$@")
            break
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done

# Always inject --profile monitor when requested
if [[ "$MONITOR_REQUESTED" == true ]]; then
    PASSTHROUGH=("--profile" "monitor" "${PASSTHROUGH[@]}")
fi

# ECHO_TOPIC_ONLY: subscribe + log FPS without loading a model (DDS connectivity check)
if [[ "$ECHO_TOPIC_ONLY_REQUESTED" == true ]]; then
    export ECHO_TOPIC_ONLY=true
fi

# DEBUG: enable extra metrics inside the container
if [[ "$DEBUG_REQUESTED" == true ]]; then
    export DEBUG=true
fi

# Auto-detect ACTION_TYPE from model checkpoint's anvil_config.json.
# Checks pretrained_model/ subdirectory first, then checkpoint root, then HF snapshots/.
# Always overrides any existing ACTION_TYPE value.
if [[ -n "${MODEL_PATH:-}" ]]; then
    _model_root="${MODEL_PATH}"
    _anvil_config=""
    # 1. pretrained_model/anvil_config.json (most common)
    if [[ -f "${_model_root}/pretrained_model/anvil_config.json" ]]; then
        _anvil_config="${_model_root}/pretrained_model/anvil_config.json"
    # 2. root-level anvil_config.json
    elif [[ -f "${_model_root}/anvil_config.json" ]]; then
        _anvil_config="${_model_root}/anvil_config.json"
    # 3. HF cache snapshot: snapshots/<hash>/anvil_config.json
    elif [[ -d "${_model_root}/snapshots" ]]; then
        _anvil_config=$(find "${_model_root}/snapshots" -maxdepth 2 -name "anvil_config.json" | sort -r | head -1)
    fi

    if [[ -n "${_anvil_config}" && -f "${_anvil_config}" ]]; then
        _detected=$(python3 -c "import json; d=json.load(open('${_anvil_config}')); print(d.get('action_type','absolute'))" 2>/dev/null || true)
        if [[ -n "${_detected}" ]]; then
            export ACTION_TYPE="${_detected}"
            echo "[run_inference] ACTION_TYPE=${ACTION_TYPE} (auto-detected from ${MODEL_PATH})"
        fi
    fi
fi

# Production-only: MONITOR_ENABLE env var triggers inference_monitor_node inside the container;
# also pre-create the output dir as current user so Docker can't claim root ownership.
REAL_MONITOR=false
if [[ "$MONITOR_REQUESTED" == true && "$FAKE_HARDWARE" == false ]]; then
    REAL_MONITOR=true
    export MONITOR_ENABLE=true
    MONITOR_DIR="${MONITOR_OUTPUT_DIR:-${REPO_ROOT}/monitor_output}"
    export MONITOR_OUTPUT_DIR="$MONITOR_DIR"
    mkdir -p "$MONITOR_DIR"
    echo "[run_inference] Monitor enabled → output: $MONITOR_DIR"
fi

# Benchmark scoring tool: only worth starting for an `up`, never for down/logs/ps.
# Runs on the host (stdlib Python, no deps) and reads MODEL_PATH/CONFIG_FILE from
# our exported env, so it always scores against the checkpoint this run mounted.
BENCHMARK_PID=""
stop_benchmark() {
    if [[ -n "$BENCHMARK_PID" ]] && kill -0 "$BENCHMARK_PID" 2>/dev/null; then
        kill "$BENCHMARK_PID" 2>/dev/null || true
        wait "$BENCHMARK_PID" 2>/dev/null || true
    fi
}

if [[ "$BENCHMARK_REQUESTED" == true && " ${PASSTHROUGH[*]} " == *" up "* ]]; then
    BENCHMARK_PORT="${BENCHMARK_PORT:-8777}"
    BENCHMARK_LOG="${REPO_ROOT}/tools/benchmark_web/results/server.log"
    mkdir -p "$(dirname "$BENCHMARK_LOG")"
    # MODEL_PATH/CONFIG_FILE may come from .env (compose reads it, bash does not),
    # in which case the tool falls back to parsing .env itself.
    [[ -n "${MODEL_PATH:-}" ]] && export MODEL_PATH
    [[ -n "${CONFIG_FILE:-}" ]] && export CONFIG_FILE
    python3 "${REPO_ROOT}/tools/benchmark_web/server.py" --port "$BENCHMARK_PORT" \
        >"$BENCHMARK_LOG" 2>&1 &
    BENCHMARK_PID=$!
    trap stop_benchmark EXIT INT TERM
    sleep 1
    if kill -0 "$BENCHMARK_PID" 2>/dev/null; then
        echo "[run_inference] Benchmark tool: http://127.0.0.1:${BENCHMARK_PORT} (--no-benchmark to skip)"
    else
        # Most likely the port is already taken by an earlier run; keep inference going.
        echo "[run_inference] WARNING: benchmark tool exited at startup — see $BENCHMARK_LOG"
        BENCHMARK_PID=""
    fi
fi

echo "[run_inference] compose: $(basename "$COMPOSE_FILE") | args: ${PASSTHROUGH[*]:-<none>}"

# Run docker compose and capture exit code (don't let set -e abort before plotting)
set +e
COMPOSE_ARGS=(-f "$COMPOSE_FILE")
if [[ "$DEV_SRC" == true ]]; then
    COMPOSE_ARGS+=(-f "${REPO_ROOT}/docker-compose.devsrc.yml")
    echo "[run_inference] --dev-src: running ros2/src from the working tree, not the image"
fi

docker compose "${COMPOSE_ARGS[@]}" "${PASSTHROUGH[@]}" --remove-orphans
COMPOSE_EXIT=$?
set -e

# Auto-plot monitor CSV after production monitor run
if [[ "$REAL_MONITOR" == true ]]; then
    CSV="${MONITOR_DIR}/inference_data.csv"
    PNG="${MONITOR_DIR}/inference_report.png"
    if [[ -f "$CSV" ]]; then
        echo "[run_inference] Plotting monitor data: $CSV"
        uv run python "${REPO_ROOT}/scripts/plot_monitor_csv.py" "$CSV" -o "$PNG" \
            && echo "[run_inference] Report saved: $PNG" \
            || echo "[run_inference] WARNING: plot_monitor_csv.py failed (exit $?)"
    else
        echo "[run_inference] WARNING: monitor CSV not found at $CSV"
    fi
fi

exit "$COMPOSE_EXIT"
