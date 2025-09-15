from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


class DatasetLogger:
    """Append interaction samples to disk (JSONL) for offline training.

    Structure:
      <root>/dataset/<YYYYMMDD>/<user_id>.jsonl

    Each line is a JSON record with minimal fields required to reconstruct
    training sequences while keeping storage cheap.
    """

    def __init__(self, root_dir: str, subdir: Optional[str] = None) -> None:
        self.root = root_dir
        self.dataset_dir = os.path.join(self.root, subdir or "dataset")
        os.makedirs(self.dataset_dir, exist_ok=True)

    def _path_for(self, user_id: str) -> str:
        day = time.strftime("%Y%m%d")
        d = os.path.join(self.dataset_dir, day)
        os.makedirs(d, exist_ok=True)
        safe_uid = "u_" + "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
        return os.path.join(d, f"{safe_uid}.jsonl")

    def log(self, user_id: str, record: Dict[str, Any]) -> None:
        try:
            record = dict(record)
            record.setdefault("ts", int(time.time() * 1000))
            p = self._path_for(user_id)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # best-effort logging: never crash the game loop
            pass
