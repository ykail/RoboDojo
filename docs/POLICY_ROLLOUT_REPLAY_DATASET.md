# Kai0 policy rollout + 可恢复状态数据集

本功能用于无人值守地 rollout 一个 Kai0 Pi0.5 checkpoint，并把每条轨迹同时保存为：

1. 标准 LeRobot v3 数据（joint state、policy action、三路相机视频）；
2. 与每个 LeRobot frame 严格对齐的 RoboDojo 仿真状态 sidecar；
3. layout、checkpoint、eval seed、policy seed、成功/失败、RoboDojo/Kai0/Assets/IsaacLab commit 和 Isaac 版本 provenance。

当前已经实现的是“采集前半段”。普通 LeRobot 视频可以直接读取；sidecar 已经包含未来从第 N 帧恢复场景所需的显式物理状态，但“选择视频时间点 -> 恢复 -> 人工介入”的恢复工具还没有实现。

## Coffee：官方 59999 checkpoint 采集 100 条 make_toast

先确认 GPU 空闲，再运行：

```bash
ssh OneMoreCupofCoffee
cd /home/ykail/vibe_code/RoboDojo
source /home/ykail/miniconda3/etc/profile.d/conda.sh
conda activate RoboDojo

XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
bash scripts/RoboDojo/collect_kai0_rollouts.sh \
  --task make_toast \
  --checkpoint-dir /home/ykail/data/RoboDojo_hf/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999 \
  --checkpoint-id RoboDojo-sim-arx_x5-joint-0/59999 \
  --episodes 100 \
  --layout-plan '0:0-69,1:0-29' \
  --lerobot-root /home/ykail/data/lerobot \
  --lerobot-repo-id make_toast_pi05_official_59999_rollout100 \
  --kai0-root /home/ykail/vibe_code/RoboDojo/third_party/kai0 \
  --kai0-python /home/ykail/vibe_code/RoboDojo/third_party/kai0/.venv/bin/python \
  --headless
```

这个 plan 使用 100 个互不重复的复合身份：

- eval seed 0 的 layout 0–69；
- eval seed 1 的 layout 0–29。

每条 `make_toast` 最多运行任务官方的 1400 control steps（25 Hz，即最多约 56 秒）；成功时仍会提前结束。成功和失败 rollout 都会保存，因为该数据集的用途是定位 corner case，而不是只构造 demonstration。

这不是官方 benchmark 的计数协议。官方单个 task/seed 默认评测 25 条；这里的 70+30 是专门用于数据采集的 100-layout protocol，不会改变普通评测命令的 25 条上限。

如需观看 Isaac Sim，把最后的 `--headless` 改为 `--gui`。自动录制没有键盘控制，也不需要窗口焦点。

## 中断后续采

重复完全相同的命令，并增加 `--resume`：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
bash scripts/RoboDojo/collect_kai0_rollouts.sh \
  --task make_toast \
  --checkpoint-dir /home/ykail/data/RoboDojo_hf/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999 \
  --checkpoint-id RoboDojo-sim-arx_x5-joint-0/59999 \
  --episodes 100 \
  --layout-plan '0:0-69,1:0-29' \
  --lerobot-root /home/ykail/data/lerobot \
  --lerobot-repo-id make_toast_pi05_official_59999_rollout100 \
  --kai0-root /home/ykail/vibe_code/RoboDojo/third_party/kai0 \
  --kai0-python /home/ykail/vibe_code/RoboDojo/third_party/kai0/.venv/bin/python \
  --headless \
  --resume
```

续采进度只认已经原子提交的 episode metadata。启动前会核对：

- LeRobot `total_episodes` 与 RoboDojo episode metadata 是同一连续前缀；
- collection ID、plan hash、checkpoint、layout、eval seed 和 policy seed 一致；
- state NPZ、layout JSON 和 schema 文件都存在，且 SHA256 正确；
- 同一个 plan index 没有重复。

默认 policy seed 等于当前 eval seed，保持 `eval_kai0_pi05.sh` 的现有语义。若显式传 `--policy-seed N`，该固定值也会进入签名；续采时不能悄悄换成另一 seed。

如果进程在 LeRobot commit 的极短窗口内被 `SIGKILL` 或机器掉电，可能留下无法自动回滚的半提交。续采会 fail closed 并报告 episode/metadata 不一致，不会把同一个 layout 再跑一次而静默产生 101 条。

## 输出结构

```text
/home/ykail/data/lerobot/make_toast_pi05_official_59999_rollout100/
├── data/
│   ├── chunk-000/                # 标准 LeRobot v3 frame/parquet
│   └── robodojo_replay/
│       └── chunk-000/
│           └── episode_0000000.npz  # 每帧显式仿真状态 + terminal state
├── videos/                       # 标准 LeRobot v3 三路视频
└── meta/
    ├── info.json
    └── robodojo/
        ├── collections/          # 签名后的 100 条 plan + policy provenance
        ├── episodes/             # 每条成功/失败、seed、layout、checkpoint、sidecar hash
        └── replay/
            ├── schema.json
            └── layouts/          # 原始 saved layout + 对象/关节 inventory
```

一条 frame 的时间语义是：

```text
observation_t + simulator_state_t + policy_action_t
                         |
                         v
                 RoboDojo.take_action()
```

NPZ 中保存两台 ARX-X5 的 root/joint/target 状态、4 片面包的 pose 和线/角速度、toaster 的 root/joint/velocity 状态，以及任务计数。静态 shelf、桌面、材质和初始颜色由完整 saved layout 重建。另有一个 action 执行后的 terminal snapshot。

当前 replay profile 对 `make_toast` 已做完整 inventory 验证。若其他 task 含 Dynamic、Garment 或 Fluid 移动物体，录制会明确拒绝启动，而不会生成一个错误标记为 complete 的 sidecar。

## “重放”目前能做到什么

- 现在：用标准 LeRobot 工具读取 frame、action 和三路视频；读取 NPZ 检查任意视频帧对应的机械臂/面包/toaster 状态。
- 下一阶段：实现 restore runner，按 episode + frame 写回 layout 和显式物理状态，渲染校验后切到人工控制。
- 不承诺：接触瞬间 bit-for-bit 的确定性复现。PhysX 的 contact warm-start/cache、policy server 的 action-chunk 内部状态和 RNG 没有被序列化。实际人工分支应优先恢复到故障前约 0.2–0.5 秒。

## 关键代码

| 作用 | 文件 |
| --- | --- |
| 100 条计划、签名、断点续采 | `scripts/RoboDojo/collect_kai0_rollouts.py`、`src/eval_client/rollout_collection.py` |
| Kai0 + Isaac 生命周期 | `scripts/RoboDojo/eval_kai0_pi05.sh` |
| policy pre-action 录制 hook | `src/eval_client/policy_runtime/eval_loop.py` |
| Isaac 状态 inventory/capture | `src/eval_client/sim_state_snapshot.py` |
| CPU LeRobot/NPZ 原子提交 | `src/eval_client/lerobot_stream_recorder.py`、`scripts/RoboDojo/lerobot_stream_writer.py` |
