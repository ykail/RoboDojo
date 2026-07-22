# Pi0.5 keyboard intervention deployment

This guide reproduces the visible RoboDojo simulation, Pi0.5 inference server,
and direct LeRobot v3 keyboard-intervention recorder on another Ubuntu
workstation. For operator controls and output semantics, see
[`KEYBOARD_INTERVENTION.md`](KEYBOARD_INTERVENTION.md).

The validated target in this guide is `OneMoreCupofCoffee`:

| Component | Validated target |
| --- | --- |
| OS | Ubuntu 22.04.5, x86_64 |
| GPU | NVIDIA RTX 3090, 24 GB |
| Driver | 580.126.09 |
| CPU / RAM | Intel i7-14700KF / 62 GiB |
| Code root | `/home/ykail/vibe_code/RoboDojo` |
| Data root | `/home/ykail/data` |
| Free space before deployment | about 400 GiB |

The source workstation (`Piper`) uses the same GPU class with driver 570 and
CUDA Toolkit 12.8. A newer NVIDIA driver can run the CUDA 12.8 binaries, but the
CUDA version printed by `nvidia-smi` is only the driver's compatibility level.
CuRobo still needs a real CUDA 12.8 Toolkit installation and `nvcc`.

## What is required

Only the RoboDojo superproject is required. It pins these submodules:

| Repository | Pinned revision |
| --- | --- |
| `XPolicyLab` | `8d6d392fd358ba65bf2382e84657ff27902f58a1` |
| `third_party/IsaacLab` | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| `third_party/curobo` | `895c6517243f8cb091c73c018c8167192d39599a` |

OpenPI is already vendored under `XPolicyLab/policy/Pi_05/openpi`. Do not clone
another OpenPI repository. Kai0 and ROS are not required for inference or
intervention recording. Kai0 is only needed later if it is
chosen as the training stack.

This branch reads `XPolicyLab` from `https://github.com/ykail/XPolicyLab.git`,
branch `feat/robodojo-pi05-runtime`, so fresh clones can fetch the pinned
Pi0.5 environment-isolation and WebSocket cold-start fixes.

Runtime storage has three separate parts:

| Content | Required for collection | Approximate size |
| --- | --- | --- |
| RoboDojo Assets | Yes | 39 GiB resolved; 66 GiB with LFS cache |
| Seed-0 inference checkpoint (`params` + `assets`) | Yes | 12.4 GB |
| Seed-0 `train_state` | No; only for resuming training | 32.3 GB |
| Original LeRobot v3 dataset | No; only for training | 120 GB / 112 GiB |

Do not copy a Conda environment or OpenPI `.venv` from another workstation.
Both contain machine-specific paths. Build the simulator and policy
environments independently as described below.

## 1. Install host prerequisites

Install the remaining host tools first:

```bash
sudo apt-get update
sudo apt-get install -y \
  git \
  git-lfs \
  rsync \
  curl \
  wget \
  cmake \
  ninja-build \
  build-essential \
  ffmpeg \
  x11-utils

git lfs install
```

`OneMoreCupofCoffee` already has a working 580.126.09 NVIDIA driver; keep it
installed. Only install a driver on a different host when `nvidia-smi` fails.
Install the CUDA 12.8 Toolkit package without replacing the working driver. Do
not install the `cuda` or `cuda-12-8` meta package, because those may also
change the driver:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

Add the toolkit to the interactive shell and verify it:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

nvidia-smi
nvcc --version
```

Persist the three exports in the user's shell configuration after the paths
have been verified.

## 2. Obtain the feature branch

The feature branch is published on the `ykail/RoboDojo` fork. On a workstation
that does not already have the repository, clone it once:

```bash
git clone --recurse-submodules \
  --branch feat/keyboard-intervention-teleop \
  https://github.com/ykail/RoboDojo.git \
  /home/ykail/vibe_code/RoboDojo
```

If `/home/ykail/vibe_code/RoboDojo` is already a Git worktree, do not clone or
rsync the code over it again. With a clean worktree, update the existing copy:

```bash
cd /home/ykail/vibe_code/RoboDojo
git fetch origin feat/keyboard-intervention-teleop
git switch feat/keyboard-intervention-teleop
git pull --ff-only origin feat/keyboard-intervention-teleop
git submodule sync --recursive
git submodule update --init --recursive
```

Verify the branch and pinned submodules:

```bash
cd /home/ykail/vibe_code/RoboDojo
git branch --show-current
git status --short
git submodule status
```

The branch must be `feat/keyboard-intervention-teleop`, the worktree must be
clean, and the three submodule hashes must match the table above. The installer
checks out the revisions recorded by the superproject; it does not update them
to the current remote `main`.

Use `rsync` later for Assets, checkpoints, and datasets only. Conda and uv
environments are rebuilt locally and are not copied between machines.

## 3. Install the simulator environment

`OneMoreCupofCoffee` already has Miniconda at `/home/ykail/miniconda3`, but a
non-interactive SSH shell does not automatically put it on `PATH`. Source it
before invoking the installer so that the existing installation is reused:

```bash
source /home/ykail/miniconda3/etc/profile.d/conda.sh

cd /home/ykail/vibe_code/RoboDojo
bash scripts/install.sh --install
```

The installer creates an independent `RoboDojo` environment with Python 3.11,
the Torch 2.7 CUDA 12.8 stack, Isaac Sim 5.1, the pinned IsaacLab fork, and the
pinned CuRobo fork. It may request `sudo` and network access. Do not install
OpenPI into this Conda environment.

Verify the source revisions again after installation:

```bash
cd /home/ykail/vibe_code/RoboDojo
git status --short
git submodule status
```

## 4. Install the separate Pi0.5 environment

The target already has `uv` 0.9.18 at `/home/ykail/.local/bin/uv`:

```bash
export PATH="/home/ykail/.local/bin:${PATH}"

cd /home/ykail/vibe_code/RoboDojo
bash XPolicyLab/policy/Pi_05/install.sh

test -x XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
```

This creates `XPolicyLab/policy/Pi_05/openpi/.venv` from its lock file. The
policy environment is intentionally separate from the simulator Conda
environment.

## 5. Transfer and relink Assets

Assets are mandatory. Copy the complete cached asset repository if future LFS
updates should remain possible:

```bash
SOURCE_HOST=piper@10.19.127.120
ROBO_ROOT=/home/ykail/vibe_code/RoboDojo

mkdir -p "${ROBO_ROOT}/.cache/robodojo_assets_repo"

rsync -aH --partial --append-verify --info=progress2 \
  "${SOURCE_HOST}:/home/piper/vibe_code/RoboDojo/.cache/robodojo_assets_repo/" \
  "${ROBO_ROOT}/.cache/robodojo_assets_repo/"

cd "${ROBO_ROOT}"
if [ -e Assets ] && [ ! -L Assets ]; then
  echo "Refusing to replace real path: ${ROBO_ROOT}/Assets" >&2
  exit 1
fi
ln -sfnT .cache/robodojo_assets_repo/Assets Assets
```

CuRobo embodiment files contain absolute asset paths. Rewrite them for the new
workstation even when Assets were copied rather than downloaded:

```bash
source /home/ykail/miniconda3/etc/profile.d/conda.sh
conda activate RoboDojo

cd /home/ykail/vibe_code/RoboDojo
python utils/update_embodiment_config_path.py
```

Alternatively, initialize Assets from Hugging Face with
`bash scripts/init_assets.sh`, then run the same path-update command.

## 6. Transfer the Pi0.5 checkpoint

The collection wrapper currently uses seed 0 and checkpoint step 59999. For
inference, copy `params`, normalization `assets`, and checkpoint metadata while
excluding the optimizer `train_state`:

```bash
SOURCE_HOST=piper@10.19.127.120
DATA_ROOT=/home/ykail/data
ROBO_ROOT=/home/ykail/vibe_code/RoboDojo
CKPT_NAME=RoboDojo-sim-arx_x5-joint-0
CKPT_ROOT="${DATA_ROOT}/RoboDojo_hf/ckpt/RoboDojo/Pi_05/${CKPT_NAME}"

mkdir -p "${CKPT_ROOT}/59999"

rsync -aH --partial --append-verify --info=progress2 \
  --exclude='/train_state/' \
  "${SOURCE_HOST}:/home/piper/data/RoboDojo_hf/ckpt/RoboDojo/Pi_05/${CKPT_NAME}/59999/" \
  "${CKPT_ROOT}/59999/"

mkdir -p "${ROBO_ROOT}/XPolicyLab/policy/Pi_05/checkpoints"
POLICY_CKPT="${ROBO_ROOT}/XPolicyLab/policy/Pi_05/checkpoints/${CKPT_NAME}"
if [ -e "${POLICY_CKPT}" ] && [ ! -L "${POLICY_CKPT}" ]; then
  echo "Refusing to replace real path: ${POLICY_CKPT}" >&2
  exit 1
fi
ln -sfnT \
  "${CKPT_ROOT}" \
  "${POLICY_CKPT}"

test -d "${CKPT_ROOT}/59999/params"
test -f "${CKPT_ROOT}/59999/assets/arx_x5_sim/norm_stats.json"
```

Rerunning the same `rsync` command resumes an interrupted transfer. Do not use
compression for model and video files. Remove the `--exclude` option only when
`train_state` is required to resume training.

## 7. Optionally transfer the original training dataset

The 120 GB LeRobot v3 dataset is not read during inference or corrective-data
collection. Transfer it only when local training is planned:

```bash
SOURCE_HOST=piper@10.19.127.120
DATA_ROOT=/home/ykail/data
ROBO_ROOT=/home/ykail/vibe_code/RoboDojo

mkdir -p "${DATA_ROOT}/RoboDojo_hf/data" "${ROBO_ROOT}/data"

rsync -aH --partial --append-verify --info=progress2 \
  "${SOURCE_HOST}:/home/piper/data/RoboDojo_hf/data/RoboDojo_lerobot_v30_video/" \
  "${DATA_ROOT}/RoboDojo_hf/data/RoboDojo_lerobot_v30_video/"

DATASET_LINK="${ROBO_ROOT}/data/RoboDojo_lerobot_v30_video"
if [ -e "${DATASET_LINK}" ] && [ ! -L "${DATASET_LINK}" ]; then
  echo "Refusing to replace real path: ${DATASET_LINK}" >&2
  exit 1
fi
ln -sfnT \
  "${DATA_ROOT}/RoboDojo_hf/data/RoboDojo_lerobot_v30_video" \
  "${DATASET_LINK}"
```

The target currently has about 400 GiB free. Minimal inference deployment has
comfortable headroom. Copying the full dataset, all training states, and
collected LeRobot videos leaves much less room for long collection sessions; mount
an additional data disk before committing to that layout.

## 8. Run a no-GPU preflight

This validates code, task configuration, Assets, the checkpoint link, and the
Conda environment without importing Isaac Sim or starting a policy server:

```bash
source /home/ykail/miniconda3/etc/profile.d/conda.sh
conda activate RoboDojo

cd /home/ykail/vibe_code/RoboDojo

CUDA_VISIBLE_DEVICES="" \
bash scripts/internal/verify_install.sh \
  --skip-isaac \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --policy-env uv \
  --env-cfg arx_x5 \
  --task stack_bowls \
  --ckpt RoboDojo-sim-arx_x5-joint-0
```

Wait until the GPU is available before rerunning the preflight without
`--skip-isaac` or starting a collection session.

## 9. Start visible collection

Keyboard collection needs a visible, focused Isaac Sim window. The safest
method is to log in to the Ubuntu desktop locally or through a GPU-capable
remote desktop and launch the command from a terminal in that session.

The observed Xorg display on `OneMoreCupofCoffee` is currently `:1`, but it can
change after reboot. A plain SSH shell has no `DISPLAY`. If SSH must launch into
the already logged-in desktop, first inspect the active display:

```bash
who
ls -l /tmp/.X11-unix
```

Only if those commands still show user `ykail`, display `:1`, and socket `X1`,
export its session variables and verify that the X server is reachable:

```bash
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus

xdpyinfo >/dev/null || {
  echo "Cannot access the active X display" >&2
  exit 1
}
```

Start an operator-driven collection session:

```bash
source /home/ykail/miniconda3/etc/profile.d/conda.sh
conda activate RoboDojo

cd /home/ykail/vibe_code/RoboDojo

bash scripts/RoboDojo/collect_pi05_keyboard.sh \
  --task stack_bowls \
  --ckpt RoboDojo-sim-arx_x5-joint-0 \
  --lerobot-repo-id robodojo_interventions_stack_bowls \
  --lerobot-root /home/ykail/data/lerobot \
  --rendering-mode quality \
  --policy-gpu 0 \
  --env-gpu 0
```

Natural task success and the configured task step limit do not end an attempt.
Use the keyboard right arrow to accept the complete candidate and advance, or
the left arrow to discard it and retry the same layout. After the last saved
layout, collection cycles to the first one. `Escape` accepts the final
candidate and exits; `Backspace` discards the final candidate and exits.

The wrapper starts both the Pi0.5 policy server and Isaac Sim. Do not start a
second policy server manually. Keep the Isaac Sim window focused for keyboard
events.

Keep `--rendering-mode quality` for normal inference and correction-data
collection. `balanced` and `performance` can improve interactivity, but they
change the camera rendering preset and may shift images away from the training
distribution.

Press plain `I` once to enter manual control and again to return to Pi0.5. Do
not use `Space`: Isaac Sim binds it to Play/Pause. Frames are written directly
to the LeRobot dataset by a CPU-only child process while Isaac runs. To append
after a later restart, run the same command with `--resume`; without that flag,
the wrapper refuses an existing dataset rather than overwriting it.

## Troubleshooting

- **`uv venv not found`**: run `bash XPolicyLab/policy/Pi_05/install.sh`; do not
  start a policy server manually.
- **`nvcc` or CUDA headers missing**: install CUDA Toolkit 12.8 and export
  `CUDA_HOME`; `nvidia-smi` alone does not prove that the Toolkit is installed.
- **Assets or URDF paths still reference `/home/piper`**: recreate the `Assets`
  link and run `python utils/update_embodiment_config_path.py` from the repo
  root.
- **The policy server starts but the checkpoint cannot load**: verify
  `59999/params` and `59999/assets/arx_x5_sim/norm_stats.json`, then check the
  checkpoint symlink with `readlink -f`.
- **No window or no keyboard response**: launch from the logged-in graphical
  desktop, verify `DISPLAY` and `XAUTHORITY`, and keep the Isaac Sim window
  focused.
- **Out of GPU memory on a single 24 GB GPU**: close other GPU applications. If
  two GPUs are available, split `--policy-gpu` and `--env-gpu`.
- **Conda dependency conflicts**: keep OpenPI in its uv `.venv`; never install
  it into the `RoboDojo` simulator environment.
