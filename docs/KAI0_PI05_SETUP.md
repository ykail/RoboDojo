# RoboDojo + Kai0 Pi0.5 配置与使用指南

本文档是 `ykail/RoboDojo` 的 `feat/kai0-pi05-runtime` 分支的标准配置
入口，适用于：

- 在一台新的 Ubuntu 工作站复现 RoboDojo + Kai0 Pi0.5；
- 使用 Kai0 server 接入 RoboDojo 仿真推理和评测；
- 使用 GUI 观察 rollout，或通过键盘介入并写入 LeRobot v3；
- 在 Kai0 的不同 branch/worktree 上开发 base、subtask、memory 等能力。

架构和协议设计见
[`KAI0_PI05_INTEGRATION.md`](KAI0_PI05_INTEGRATION.md)，键盘控制和数据语义见
[`KEYBOARD_INTERVENTION.md`](KEYBOARD_INTERVENTION.md)。

> 数据集、Assets、checkpoint、Conda 环境和 `.venv` 都不提交到 Git。
> 代码通过 Git commit 和 submodule 固定；大文件在每台机器上下载或 rsync。

## 最短部署顺序

1. 使用 `--recurse-submodules` clone `feat/kai0-pi05-runtime`。
2. 运行 `bash scripts/install.sh -i` 创建 RoboDojo/Isaac Sim 环境。
3. 运行 `bash scripts/init_assets.sh` 并更新 embodiment 绝对路径。
4. 在 `third_party/kai0` 中创建独立 uv `.venv`。
5. 准备含 `params/` 和 `assets/arx_x5_sim/norm_stats.json` 的 checkpoint step。
6. 先执行 launcher `--dry-run`，再运行一条 GUI smoke test。

后续章节给出每一步的完整命令、数据采集方式和开发流程。

## 1. 支持范围与责任边界

```text
RoboDojo / Isaac Sim                   Kai0 policy process
--------------------                  -------------------
创建 task 和 environment              加载 Pi0.5 config/checkpoint/norm
采集 camera observation + state  ---> observation adapter
执行 joint action                <--- Pi0.5 inference + action chunk
任务成功判定和结果记录                 checkpoint provenance
```

两边通过 `robodojo-policy-v1` WebSocket 协议通信。RoboDojo 主动发送观测，
Kai0 返回动作，动作最终由 RoboDojo 在 Isaac Sim 中执行。

运行 `scripts/RoboDojo/eval_kai0_pi05.sh` 时，脚本会自动启动 Kai0 server、
等待健康检查、启动 Isaac Sim，并在评测结束后回收 server。一般不需要在另一
个终端手动启动 policy server。

当前集成支持：

| 能力 | 入口 |
| --- | --- |
| Kai0 Pi0.5 仿真推理 | `scripts/RoboDojo/eval_kai0_pi05.sh` |
| GUI/无窗口单任务评测 | 同上，GUI 默认开启，`--headless` 关闭窗口 |
| 快速观察 layout | `--control-mode keyboard_observe` |
| 人工介入与 LeRobot v3 采集 | `--control-mode keyboard_intervention` |
| Base Pi0.5 训练 | Kai0 的 `scripts/train.py` |
| 真机推理 | Kai0 原生 serve/inference 入口，不经过 RoboDojo launcher |
| 官方和其他 policy | 保留 `XPolicyLab/` 路径 |

## 2. 仓库、branch 与 submodule

标准源码入口：

```text
repository: git@github.com:ykail/RoboDojo.git
branch:     feat/kai0-pi05-runtime
```

目录关系：

```text
RoboDojo/
├── XPolicyLab/             官方/原有 policy 集成
├── third_party/IsaacLab/  RoboDojo 固定的 IsaacLab
├── third_party/curobo/    RoboDojo 固定的 CuRobo
└── third_party/kai0/      Kai0 模型、训练和 policy server
```

该分支当前固定的已验证 revision 为：

| 组件 | Revision |
| --- | --- |
| `XPolicyLab` | `8d6d392fd358ba65bf2382e84657ff27902f58a1` |
| `third_party/IsaacLab` | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| `third_party/curobo` | `895c6517243f8cb091c73c018c8167192d39599a` |
| `third_party/kai0` | `ecc1a7451c3156b1e5f7533851dbb0222896206f` |

### 2.1 新电脑 clone

新电脑的 GitHub SSH key 必须有 `ykail/RoboDojo` 和
`jiran0407/kai0` 的读取权限：

```bash
git clone --recurse-submodules \
  --branch feat/kai0-pi05-runtime \
  git@github.com:ykail/RoboDojo.git

cd RoboDojo
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
```

普通使用者不要再次执行 `git submodule add`，也不需要单独 clone Kai0。
fresh clone 后 submodule 显示 detached HEAD 是正常状态：RoboDojo 固定的是
Kai0 commit，而不是一个会自动移动的 branch。

需要精确复现实验时，先检出实验记录中的 RoboDojo tag/commit，再更新
submodule：

```bash
git checkout <ROBODOJO_TAG_OR_COMMIT>
git submodule update --init --recursive
```

`git submodule status` 开头无符号表示匹配父仓库；`-` 表示尚未初始化，`+`
表示当前 submodule commit 与父仓库固定值不一致。

## 3. 主机要求

当前安装脚本配置 Python 3.11、PyTorch 2.7/cu128、Isaac Sim 5.1 和固定的
IsaacLab/CuRobo。推荐使用 Ubuntu 22.04 x86_64 和 NVIDIA GPU。

安装前检查：

```bash
nvidia-smi
git --version
git lfs version
command -v conda || echo "Conda will be installed by scripts/install.sh"
command -v uv || echo "Install uv before configuring Kai0"
```

至少需要以下系统工具：

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs rsync curl wget cmake ninja-build build-essential ffmpeg

git lfs install
```

如果还没有 `uv`，使用其官方 standalone installer，然后让当前 shell 找到
安装目录：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
uv --version
```

也可以按 [uv 官方安装文档](https://docs.astral.sh/uv/getting-started/installation/)
选择其他安装方式。

`nvidia-smi` 必须先正常工作。CuRobo 还需要真实的 CUDA 12.8 Toolkit 和
`nvcc`；`nvidia-smi` 显示的 CUDA compatibility level 不能代替 Toolkit。
不要为了安装 Toolkit 随意替换已经工作的 NVIDIA driver。更完整的主机配置
说明见 [`KEYBOARD_INTERVENTION_DEPLOYMENT.md`](KEYBOARD_INTERVENTION_DEPLOYMENT.md)。

## 4. 两套独立 Python 环境

RoboDojo/Isaac Sim 和 Kai0/JAX 必须使用独立环境。两边是不同进程，通过
socket 通信，不需要把 Kai0、JAX 或 LeRobot 安装到 Isaac Sim 的 Conda
环境中。

### 4.1 RoboDojo / Isaac Sim

```bash
cd /absolute/path/to/RoboDojo
bash scripts/install.sh -i

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate RoboDojo
```

fresh install 会把 Miniconda 放在 `${HOME}/miniconda3`。如果机器原本使用其他
Conda 安装位置，请改为 source 该安装的 `etc/profile.d/conda.sh`；也可以在
安装完成后重开 shell。

安装中断后可从指定步骤恢复：

```bash
bash scripts/install.sh --from <step>
```

可用 step 为 `system`、`conda`、`base_deps`、`submodules`、`isaacsim`、
`isaaclab` 和 `curobo`。

### 4.2 Kai0 policy 环境

```bash
cd /absolute/path/to/RoboDojo/third_party/kai0
git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

.venv/bin/python -c \
  'import jax, lerobot, openpi, websockets; print(jax.devices())'
```

不要从另一台机器复制 Conda 环境或 `.venv`。这些环境可能包含绝对路径、
editable install 和机器相关的编译产物。

## 5. Assets、数据集与 checkpoint

### 5.1 Assets

从官方 Hugging Face 仓库下载 Assets：

```bash
cd /absolute/path/to/RoboDojo
bash scripts/init_assets.sh

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate RoboDojo
python utils/update_embodiment_config_path.py
```

也可以 rsync 已验证机器上的 `.cache/robodojo_assets_repo/`，但复制后仍应运行
`update_embodiment_config_path.py`，因为 CuRobo 配置可能包含旧机器的绝对
路径。

### 5.2 数据集

仿真推理不读取训练数据集。只有训练时才需要下载，例如：

```bash
ROBO_DOJO_DATA_ROOT=/absolute/path/to/data \
bash scripts/RoboDojo/download_data.sh huggingface lerobot_v3.0
```

`lerobot_v3.0` 是 joint-only 的 LeRobot v3 数据。其他格式和大小可通过以下
命令查看：

```bash
bash scripts/RoboDojo/download_data.sh --help
```

### 5.3 Checkpoint

Kai0 launcher 的 `--checkpoint-dir` 必须指向一个实际的 JAX checkpoint step
目录，而不是实验根目录。例如：

```text
/data/checkpoints/my_experiment/5000/
├── params/
└── assets/arx_x5_sim/norm_stats.json
```

检查：

```bash
test -d /data/checkpoints/my_experiment/5000/params
test -f /data/checkpoints/my_experiment/5000/assets/arx_x5_sim/norm_stats.json
```

`--checkpoint-id` 是写入评测结果和介入数据的稳定身份，例如
`my_experiment/5000`。它不能是 `/home/...`、`file:` URL、Windows 路径或
包含 `.`/`..` path component 的本地路径。

官方 Pi0.5 checkpoint 也可以先通过 XPolicyLab downloader 获取：

```bash
bash scripts/RoboDojo/download_ckpt.sh huggingface Pi_05
```

随后将实际 step 目录传给 Kai0 launcher。数据、checkpoint 和 intervention
数据推荐放在仓库外，例如 `/data/robodojo/`，不要提交到 Git。

## 6. 标准运行配置

先设置本次实验变量；这些只是当前 shell 的变量，不会改动仓库：

```bash
export ROBODOJO_ROOT=/absolute/path/to/RoboDojo
export KAI0_ROOT="${ROBODOJO_ROOT}/third_party/kai0"
export TASK=make_toast
export CHECKPOINT_DIR=/absolute/path/to/checkpoints/my_experiment/5000
export CHECKPOINT_ID=my_experiment/5000

cd "${ROBODOJO_ROOT}"
```

### 6.1 RoboDojo 环境预检

先检查任务配置、Assets 和 Conda 环境，不验证任何 XPolicyLab policy：

```bash
bash scripts/robodojo.sh doctor --skip-policy --skip-isaac
```

移除 `--skip-isaac` 可以进一步检查 Isaac Sim/IsaacLab import。当前 doctor
尚不加载 Kai0 checkpoint；Kai0 的路径和启动参数由下一步 dry-run 检查，真实
模型/协议则由单 episode smoke test 验证。

### 6.2 无 GPU dry-run

dry-run 检查参数、路径、Kai0 Python 和待启动命令，不启动 server 或 Isaac
Sim：

```bash
bash scripts/RoboDojo/eval_kai0_pi05.sh \
  --task "${TASK}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint-id "${CHECKPOINT_ID}" \
  --kai0-root "${KAI0_ROOT}" \
  --eval-num 1 \
  --dry-run
```

### 6.3 单条 GUI smoke test

```bash
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate RoboDojo

bash scripts/RoboDojo/eval_kai0_pi05.sh \
  --task "${TASK}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint-id "${CHECKPOINT_ID}" \
  --kai0-root "${KAI0_ROOT}" \
  --eval-num 1 \
  --policy-gpu 0 \
  --env-gpu 0
```

GUI 默认开启。脚本会自动启动和停止 Kai0 server；不需要预先启动另一个
server。

### 6.4 成功率评测

```bash
bash scripts/RoboDojo/eval_kai0_pi05.sh \
  --task "${TASK}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint-id "${CHECKPOINT_ID}" \
  --kai0-root "${KAI0_ROOT}" \
  --eval-num 10 \
  --seed 0 \
  --headless \
  --policy-gpu 0 \
  --env-gpu 0
```

公平比较两个 checkpoint 时，固定 task、`--eval-num`、`--seed`、RoboDojo
commit、Kai0 commit 和 Assets。需要看画面时移除 `--headless`。

每次 Kai0 launcher 调用当前只评测一个 task，但内部仍使用 RoboDojo 的正式
`eval_policy.sh`、同一套 reset、成功判定和结果写入逻辑。现有
`scripts/robodojo.sh benchmark` sweep 只接受 XPolicyLab runtime 的
`--policy-dir/--ckpt/--policy-env`，不能直接接收 Kai0 的
`--checkpoint-dir`；因此不要将它误当作 Kai0 全任务入口。评测多个 task 时应
分别调用上述 Kai0 launcher，直到增加专用 sweep wrapper。

### 6.5 只观察 rollout，不保存数据

```bash
bash scripts/RoboDojo/eval_kai0_pi05.sh \
  --task "${TASK}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint-id "${CHECKPOINT_ID}" \
  --kai0-root "${KAI0_ROOT}" \
  --eval-num 10 \
  --control-mode keyboard_observe
```

`Left Arrow` 立即进入下一个 layout；`Escape` 或 `Backspace` 提前退出。该模式
不写 LeRobot 数据、benchmark result 或 evaluation video，且不能与
`--headless` 同时使用。

### 6.6 人工介入并写入 LeRobot v3

```bash
bash scripts/RoboDojo/eval_kai0_pi05.sh \
  --task "${TASK}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint-id "${CHECKPOINT_ID}" \
  --kai0-root "${KAI0_ROOT}" \
  --control-mode keyboard_intervention \
  --lerobot-root /absolute/path/to/lerobot \
  --lerobot-repo-id make_toast_kai0_interventions
```

继续写入已经存在且兼容的数据集时增加 `--resume`。不带 `--resume` 时脚本
不会覆盖已有数据集。

主要按键：

| Key | 作用 |
| --- | --- |
| `I` | 第一次进入人工控制，第二次返回 Pi0.5；不需要一直按住 |
| `1` / `2` | 选择左臂 / 右臂 |
| `K` | 人工控制中切换所选夹爪 |
| `Right Arrow` | 接受完整 candidate，保存并进入下一 layout |
| `Left Arrow` | 丢弃 candidate，重试同一 layout |
| `Escape` | 接受当前 candidate，保存后退出 |
| `Backspace` | 丢弃当前 candidate，然后退出 |

`Space` 没有用于介入，因为 Isaac Sim 将其绑定为 Play/Pause。完整的末端移动
和旋转按键见 [`KEYBOARD_INTERVENTION.md`](KEYBOARD_INTERVENTION.md)。

旧的 `scripts/RoboDojo/collect_pi05_keyboard.sh` 会启动 XPolicyLab Pi0.5，并
使用 `XPolicyLab/policy/Pi_05/openpi/.venv`。它是保留的 legacy 路径，不是
Kai0 采集入口；新的 Kai0 评测、观察和介入应统一使用
`eval_kai0_pi05.sh`。

## 7. GPU 与运行参数

常用参数：

| 参数/环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `--policy-gpu` | `0` | Kai0/JAX 可见 GPU |
| `--env-gpu` | `0` | Isaac Sim GPU |
| `--port` | `8000` | 本机 WebSocket/health 端口 |
| `--eval-num` | `10` | episode 数量 |
| `--seed` | `0` | RoboDojo layout seed |
| `--policy-seed` | 与 `--seed` 相同 | policy episode seed |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.3` | Kai0 JAX 显存预占比例 |
| `ROBODOJO_POLICY_READY_TIMEOUT_S` | `600` | 等待 Kai0 server ready 的秒数 |

单 GPU 可以让 `--policy-gpu` 和 `--env-gpu` 都为 `0`。多 GPU 时可将 policy
和仿真分开。只有 `policy` control mode 支持 `--headless`。

查看当前 launcher 的完整参数：

```bash
bash scripts/RoboDojo/eval_kai0_pi05.sh --help
```

## 8. 训练和不同 Kai0 功能 branch

Base Pi0.5 训练发生在 Kai0 中：

```bash
cd "${KAI0_ROOT}"
uv run scripts/train.py <CONFIG_NAME> --exp_name=<EXPERIMENT_NAME>
```

训练 config、数据路径、base checkpoint 和 norm stats 必须一起记录。具体实验
config 可以位于 Kai0 的训练 branch；它们不需要全部复制到 RoboDojo。

普通使用者只使用 RoboDojo 固定的 `third_party/kai0` commit。需要同时开发
base、subtask 或 memory 时，推荐为不同 branch 创建独立 worktree：

```bash
git -C "${KAI0_ROOT}" fetch origin

git -C "${KAI0_ROOT}" worktree add \
  -b experiment/memory-local \
  /absolute/path/to/kai0-memory \
  origin/high-level-inference

cd /absolute/path/to/kai0-memory
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

运行该版本：

```bash
cd "${ROBODOJO_ROOT}"
bash scripts/RoboDojo/eval_kai0_pi05.sh \
  --task "${TASK}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint-id "${CHECKPOINT_ID}" \
  --kai0-root /absolute/path/to/kai0-memory \
  --eval-num 1
```

每个长期使用的 worktree 都应有自己的 `.venv`。不要让多个 worktree 共用
一个 editable-install 环境。检查实际导入位置：

```bash
/absolute/path/to/kai0-memory/.venv/bin/python -c \
  'import openpi; print(openpi.__file__)'
```

## 9. 更新 Kai0 submodule 的维护流程

Kai0 修改必须先在 Kai0 仓库 commit 并 push，然后再让 RoboDojo 更新
gitlink：

```bash
cd /absolute/path/to/RoboDojo/third_party/kai0
git switch integration/robodojo-policy-v1
git pull --ff-only

# 修改、测试、commit，并先 push Kai0。

cd ../..
git add third_party/kai0
git commit -m "chore(kai0): update integration submodule"
git push
```

如果只 push RoboDojo gitlink，而对应 Kai0 commit 没有发布到其他机器可访问
的 remote，fresh clone 将无法检出该 submodule。

## 10. 常见问题

### `third_party/kai0` 为空

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### `Permission denied (publickey)`

确认当前 SSH key 有 `jiran0407/kai0` 权限：

```bash
ssh -T git@github.com
git ls-remote git@github.com:jiran0407/kai0.git HEAD
```

### `Kai0 Python is not executable` 或 `.venv` 不存在

回到实际使用的 Kai0 root 创建环境：

```bash
cd /absolute/path/to/kai0
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### `ModuleNotFoundError: numpy`、`lerobot` 或 `openpi`

检查 launcher 使用的 Python 和源码位置：

```bash
/absolute/path/to/kai0/.venv/bin/python -c \
  'import numpy, lerobot, openpi; print(openpi.__file__)'
```

不要把 XPolicyLab 的 Pi0.5 `.venv` 当作 Kai0 的标准环境。

### Server 一直等待或端口被占用

launcher 默认等待 `127.0.0.1:8000`。先查看 Kai0 server 在当前终端输出的
首个错误；常见原因是 checkpoint/norm 缺失、Python 路径错误或端口被其他
进程占用。可以通过 `--port <PORT>` 切换端口。

### `nvidia-smi` 失败

这是主机 NVIDIA driver 问题，不是 Python checkpoint 问题。先恢复 driver
和 GPU，再调试 Isaac Sim 或 JAX。

### Isaac Sim 没有窗口或收不到键盘

交互模式必须从可访问 GPU GUI 的桌面会话启动，并保持 Isaac Sim viewport
获得焦点。普通 SSH shell 通常没有正确的 `DISPLAY`、`XAUTHORITY` 和 Vulkan
上下文。详细远程桌面说明见
[`KEYBOARD_INTERVENTION_DEPLOYMENT.md`](KEYBOARD_INTERVENTION_DEPLOYMENT.md)。

### Checkpoint action dimension 或 norm 不匹配

当前 RoboDojo Pi0.5 profile 是 ARX-X5 双臂 joint action。checkpoint 必须带
与训练一致的 `arx_x5_sim` norm stats。不要用名称相似但动作/夹爪标度不同的
generic ARX norm 替换。

### `git submodule status` 出现 `+`

当前 submodule checkout 与 RoboDojo 固定版本不同。若只是运行标准版本：

```bash
git submodule update --init --recursive
```

如果其中有未提交开发内容，先保存或提交，不要直接覆盖。

## 11. 完成标准与复现记录

部署完成后至少检查：

```bash
cd /absolute/path/to/RoboDojo
git status --short
git submodule status --recursive
nvidia-smi

third_party/kai0/.venv/bin/python -c \
  'import jax, lerobot, openpi, websockets; print(openpi.__file__)'

bash scripts/RoboDojo/eval_kai0_pi05.sh --help
```

随后运行一次 GUI、单 episode smoke test，确认：

1. Kai0 server 加载的是指定 checkpoint 和 norm；
2. policy-v1 握手、observation schema 和 action dimension 校验通过；
3. Isaac Sim 能创建环境并执行 action；
4. episode 能正常结束，或在交互模式下按预期退出；
5. 评测结果记录了 RoboDojo/Kai0 revision 和 checkpoint identity。

每次正式实验至少保存：

```text
RoboDojo commit
Kai0 commit 和其他 submodule commits
Kai0 dirty/clean 状态
训练 config 与 checkpoint step
norm stats 来源
dataset 版本和 episode selection
task、eval_num、seed 与 control mode
GPU、driver、Isaac Sim 和 IsaacLab 版本
```

这样 checkpoint、代码、数据和评测结果才能在另一台机器上对应起来。
