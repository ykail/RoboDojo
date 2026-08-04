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
        selected_layout_ids: Iterable[int] | None = None,
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
        if selected_layout_ids is None:
            # Preserve the benchmark's historical dense 0..N-1 indexing.
            # Collection-only selection below uses the filename suffix so an
            # explicit plan cannot silently point at the wrong sparse file.
            for idx, file_path in enumerate(matching_files):
                self.seed_info[idx] = {"scene_layout": file_path}
            all_layout_ids = list(range(len(matching_files)))
        else:
            selected = [int(layout_id) for layout_id in selected_layout_ids]
            if any(layout_id < 0 for layout_id in selected):
                raise ValueError("selected layout ids must be non-negative")
            if len(selected) != len(set(selected)):
                raise ValueError(f"selected layout ids contain duplicates: {selected}")
            file_by_suffix = {
                int(Path(file_path).stem.rsplit("_", 1)[-1]): file_path
                for file_path in matching_files
            }
            missing = [layout_id for layout_id in selected if layout_id not in file_by_suffix]
            if missing:
                raise ValueError(
                    f"Requested {self.task_name} layout id(s) do not exist for "
                    f"eval seed {self.eval_seed}: {missing}"
                )
            for layout_id in selected:
                self.seed_info[layout_id] = {
                    "scene_layout": file_by_suffix[layout_id]
                }
            all_layout_ids = selected
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

    def eval_step(self):
        self._current_batch_seeds = None
