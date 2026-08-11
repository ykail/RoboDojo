from collections.abc import Iterable, Mapping
from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any, Dict, List

from env.global_configs import ASSETS_PATH, BENCHMARK
from utils.load_file import *


class SeedManager:
    def __init__(self, config: Mapping[str, Any]):
        self.config: Mapping[str, Any] = config
        self.num_envs: int = int(self.config["num_envs"])

        # config fields used for directory layout
        self.task_name: str = str(self.config["task_name"])
        self.config_name: str = str(self.config["config_name"])

        self.st_idx: int
        self.ed_idx: int
        self.type: str

        self._current_batch_seeds: List[int] | None = None
        # Operator-driven collection is intentionally unbounded.  Once every
        # saved layout has been visited, it rewinds the finite evaluation list
        # and increments this value so recorders can distinguish repeated
        # passes over the same layout id.
        self.cycle_index: int = 0

    def init_eval(
        self,
        completed_layout_ids: Iterable[int] | None = None,
        abandoned_layout_ids: Iterable[int] | None = None,
    ):
        self.eval_seed = self.config.get("seed", 0)
        layout_dir = Path(ASSETS_PATH, "Eval_Layout", BENCHMARK, self.config_name, str(self.eval_seed))
        pattern = re.compile(rf"{re.escape(self.task_name)}_\d+\.json")
        matching_files = sorted(
            [p for p in layout_dir.iterdir() if pattern.fullmatch(p.name)],
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )

        matching_files = [str(p) for p in matching_files]
        self.seed_info = {}
        for idx, file_path in enumerate(matching_files):
            self.seed_info[idx] = {"scene_layout": file_path}

        all_layout_ids = list(range(len(matching_files)))
        excluded = set(int(s) for s in (completed_layout_ids or [])) | set(int(s) for s in (abandoned_layout_ids or []))
        if excluded:
            self.seed_list: List[int] = [s for s in all_layout_ids if s not in excluded]
            print(
                f"[SeedManager] init_eval resume filter: excluded={len(excluded)} "
                f"remaining={len(self.seed_list)}/{len(all_layout_ids)}"
            )
        else:
            self.seed_list = all_layout_ids
        self.st_idx = 0
        self.ed_idx = len(self.seed_list)

        self.type = "eval"
        self.idx = 0
        self.cycle_index = 0
        self._current_batch_seeds = None

    def get_seeds(self, max_count: int | None = None) -> List[int] | None:
        """Return a list of seeds for the next `reset()` call.

        Returns None when enough episodes have been successfully collected.
        """

        if self.idx >= self.ed_idx:
            return None
        if max_count is not None:
            batch_size = min(self.num_envs, max(0, int(max_count)))
            if batch_size == 0:
                return None
            batch = self.seed_list[self.idx : min(self.idx + batch_size, self.ed_idx)]
            self.idx += len(batch)
            self._current_batch_seeds = batch
            return batch
        if self.idx + self.num_envs > self.ed_idx:
            batch = self.seed_list[self.idx : self.ed_idx]
            result = deepcopy(batch)
            for _ in range(self.num_envs - len(result)):
                batch.append(self.seed_list[self.ed_idx - 1])  # pad with last seed if not enough remaining
        else:
            batch = self.seed_list[self.idx : self.idx + self.num_envs]
        self.idx += self.num_envs
        self._current_batch_seeds = batch
        return batch

    def get_seed_scene_info(self, seed: int) -> Dict[str, Any]:
        seed_info = self.seed_info.get(seed)
        if seed_info is None:
            raise ValueError(f"Seed {seed} not found in seed list.")
        file_path = seed_info.get("scene_layout")
        if file_path is None or not os.path.exists(file_path):
            raise ValueError(f"Scene layout file not found for seed {seed} at expected path {file_path}.")
        data = load_json(file_path)
        return data

    def get_cyclic_seeds(self, max_count: int | None = None) -> List[int] | None:
        """Return the next batch, rewinding after the last saved layout.

        This is only for operator-driven data collection.  Benchmark
        evaluation must continue to use :meth:`get_seeds`, whose ``None``
        result is the normal finite-evaluation termination signal.
        """

        seeds = self.get_seeds(max_count=max_count)
        if seeds is not None:
            return seeds
        if self.ed_idx <= 0:
            return None
        self.idx = 0
        self.cycle_index += 1
        return self.get_seeds(max_count=max_count)

    def resume_cyclic_after(self, layout_id: int, cycle_index: int) -> tuple[int, int]:
        """Place the operator cursor immediately after one durable layout.

        Raw DAgger bundles store both the layout id and cycle at commit time.
        Restoring from that durable boundary avoids repeating layout zero after
        a process restart and remains correct when an unstable layout was
        skipped without producing a bundle.
        """

        if isinstance(layout_id, bool) or not isinstance(layout_id, int):
            raise ValueError("resume layout_id must be an integer")
        if (
            isinstance(cycle_index, bool)
            or not isinstance(cycle_index, int)
            or cycle_index < 0
        ):
            raise ValueError("resume cycle_index must be a non-negative integer")
        if self.ed_idx <= 0 or not self.seed_list:
            raise ValueError("cannot resume an empty cyclic layout list")
        try:
            position = self.seed_list.index(layout_id)
        except ValueError as exc:
            raise ValueError(
                f"resume layout {layout_id} is absent from the current seed list"
            ) from exc
        next_position = position + 1
        if next_position >= self.ed_idx:
            self.idx = 0
            self.cycle_index = cycle_index + 1
        else:
            self.idx = next_position
            self.cycle_index = cycle_index
        self._current_batch_seeds = None
        return self.seed_list[self.idx], self.cycle_index

    def eval_step(self):
        self._current_batch_seeds = None
