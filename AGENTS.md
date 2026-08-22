# Repository Guidelines

## Project Structure & Ownership

- `env/` contains the simulator backbone and managers; `env_cfg/` holds shared robot, scene, camera, and simulator YAML.
- `task/RoboDojo/tasks/` implements benchmark tasks and `task/RoboDojo/config/` contains the matching task YAML. `task_registry.py` loads task modules dynamically.
- `src/eval_client/` is the Isaac Sim evaluation client, `utils/` contains shared helpers, and `scripts/` provides public commands. Docker support is in `docker/`.
- `XPolicyLab/` and `third_party/` are submodules; do not edit their contents. Downloaded assets and evaluation output live in `Assets/` and `eval_result/`.

## Environment & Development Commands

Use the existing `RoboDojo` Conda environment and its RTX 3090 GPU: `conda activate RoboDojo`. Use it for Python, Isaac Sim, and generation scripts; do not substitute another environment. Consult `CLAUDE.md` for task patterns, validation loops, and project pitfalls.

When a task requires Python changes, edit only the specific Python files needed for that task. Do not modify unrelated Python files, apply repository-wide replacements, or run repository-wide formatters, linters, or pre-commit hooks. Run only focused checks relevant to the files or behavior changed:

```bash
python scripts/internal/task_inventory.py --format json --check
bash scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy
```

## Isaac Sim Validation

When you complete relevant code modifications, considering raise a explicit request to run the GPU Isaac Sim validation program specifying the exact command. Wait for the user to confirm/click "Yes" before proceeding with execution. Validation guidelines:
- Always request to run a small explicit subset first (for example, `--max-episodes 1`).
- Inspect the output and request confirmation before scaling up on the RTX 3090.
- Keep generated artifacts under `./tmp`; do not add large generated data to Git unless requested.

## Style, Naming & Tasks

Use four-space Python indentation, lowercase `snake_case` modules and directories, and `PascalCase` manager/helper classes. Follow the existing 120-character line length and import-order conventions. Do not run `ruff check .`, `ruff format .`, or equivalent repository-wide commands unless the user explicitly requests them. Avoid `print`, `breakpoint`, and commented-out debug code.

A task’s module, YAML filename, and exported environment class must share its task name: `tasks/stack_bowls.py` and `config/stack_bowls.yml`. Preserve the few asset-driven casing exceptions (for example, `push_T`). Task classes must inherit `TaskEnv`; keep YAML labels identical to those referenced in task logic and make `run_reward()` call a meaningful reward/success check.

Do not add the following sentence at the beginning of every newly generated python script/file.

```python
from __future__ import annotations
```
