#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROBODOJO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
X5_PYTHON="/home/acone/Robot_Lab/.venv/bin/python"
WRITER_PYTHON="/home/acone/micromamba/envs/arx-py310/bin/python"
MIN_FREE_GPU_MB="${ROBODOJO_MIN_FREE_GPU_MB:-12000}"

die() {
    echo "[acOne preflight][ERROR] $*" >&2
    exit 2
}

[[ "$(id -un)" == "acone" ]] || die "expected user acone"
[[ -x "${X5_PYTHON}" ]] || die "ARX SDK Python is missing: ${X5_PYTHON}"
[[ -x "${WRITER_PYTHON}" ]] || die "LeRobot writer Python is missing: ${WRITER_PYTHON}"
[[ -d "${ROBODOJO_ROOT}/Assets/Robots" ]] || die "RoboDojo Assets are not ready"

"${X5_PYTHON}" -c \
    'import arx5_interface as arx5; print("[acOne preflight] ARX SDK:", arx5.__file__)'
"${WRITER_PYTHON}" -c \
    'import av,filelock,lerobot; from lerobot.datasets.lerobot_dataset import LeRobotDataset; print("[acOne preflight] LeRobot:", lerobot.__version__)'

for can_name in can1 can3; do
    can_line="$(ip -o link show dev "${can_name}" 2>/dev/null)" \
        || die "CAN interface is missing: ${can_name}"
    grep -Eq '(<|,)UP(,|>)' <<<"${can_line}" \
        || die "CAN interface is not UP: ${can_name}"
done

DISPLAY="${DISPLAY:-:1}" xset q >/dev/null 2>&1 \
    || die "X11 display is unavailable: ${DISPLAY:-:1}"
ssh -o BatchMode=yes -o ConnectTimeout=5 hoo@10.19.127.58 true \
    || die "passwordless SSH from acOne to Hoo is unavailable"

free_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
[[ "${free_gpu_mb}" =~ ^[0-9]+$ ]] || die "could not read free memory on GPU 0"
(( free_gpu_mb >= MIN_FREE_GPU_MB )) \
    || die "GPU 0 has ${free_gpu_mb} MiB free; stop the local policy process before Isaac"

echo "[acOne preflight] READY: SDK, writer, Assets, CAN, X11, and Hoo SSH"
