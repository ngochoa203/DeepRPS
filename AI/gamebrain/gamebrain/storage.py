from __future__ import annotations

import json
import os
from typing import Dict, Optional
import numpy as np
import asyncio
from .database import get_database_adapter, DatabaseAdapter


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class StateStorage:
    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self.db_adapter: Optional[DatabaseAdapter] = None
        try:
            self.db_adapter = get_database_adapter()
        except Exception as e:
            print(f"Database adapter failed, using file storage: {e}")
            self.db_adapter = None

    def _global_path(self) -> str:
        return os.path.join(self.state_dir, "global_state.json")

    def _user_path(self, uid: str) -> str:
        safe = uid.replace("/", "_")
        return os.path.join(self.state_dir, f"user_{safe}.json")

    def save_global(self, state: Dict) -> None:
        with open(self._global_path(), "w", encoding="utf-8") as f:
            json.dump(state, f, cls=NumpyEncoder)

    def load_global(self) -> Optional[Dict]:
        p = self._global_path()
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            # Convert arrays back if present
            if "W" in d:
                d["W"] = np.array(d["W"], dtype=np.float32)
            if "b" in d:
                d["b"] = np.array(d["b"], dtype=np.float32)
            if "fp_index" in d:
                d["fp_index"] = {k: np.array(v, dtype=np.float32) for k, v in d["fp_index"].items()}
            return d

    def save_user(self, uid: str, state: Dict) -> None:
        with open(self._user_path(uid), "w", encoding="utf-8") as f:
            json.dump(state, f, cls=NumpyEncoder)

    def load_user(self, uid: str) -> Optional[Dict]:
        p = self._user_path(uid)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_users(self) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for name in os.listdir(self.state_dir):
            if name.startswith("user_") and name.endswith(".json"):
                p = os.path.join(self.state_dir, name)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        d = json.load(f)
                        uid = d.get("user_id") or name[len("user_"):-len(".json")]
                        out[uid] = d
                except Exception:
                    continue
        return out
