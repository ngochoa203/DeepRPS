from __future__ import annotations

import json
import os
from typing import Dict, Optional
import numpy as np

try:
    import redis  # type: ignore
except Exception:  # redis is optional
    redis = None  # type: ignore


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class StateStorage:
    """
    Storage with optional Redis backend.
    - If REDIS_URL is set and redis client is available, use Redis for persistence.
    - Otherwise, fall back to filesystem at state_dir.
    Keys:
      rps:global -> JSON string
      rps:users (set) -> user ids
      rps:user:<uid> -> JSON string
    """

    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        self._redis = None
        url = os.getenv("REDIS_URL")
        if url and redis is not None:
            try:
                self._redis = redis.from_url(url, decode_responses=True)  # str <-> str
            except Exception:
                self._redis = None
        if self._redis is None:
            os.makedirs(self.state_dir, exist_ok=True)

    # ---------------- Filesystem helpers ----------------
    def _global_path(self) -> str:
        return os.path.join(self.state_dir, "global_state.json")

    def _user_path(self, uid: str) -> str:
        safe = uid.replace("/", "_")
        return os.path.join(self.state_dir, f"user_{safe}.json")

    # ---------------- Redis helpers ----------------
    def _k_global(self) -> str:
        return "rps:global"

    def _k_user(self, uid: str) -> str:
        return f"rps:user:{uid}"

    def _k_users_set(self) -> str:
        return "rps:users"

    # ---------------- Public API ----------------
    def save_global(self, state: Dict) -> None:
        if self._redis is not None:
            try:
                self._redis.set(self._k_global(), json.dumps(state, cls=NumpyEncoder))
                return
            except Exception:
                pass
        with open(self._global_path(), "w", encoding="utf-8") as f:
            json.dump(state, f, cls=NumpyEncoder)

    def load_global(self) -> Optional[Dict]:
        if self._redis is not None:
            try:
                s = self._redis.get(self._k_global())
                if s is None:
                    return None
                d = json.loads(s)
                # Convert arrays back if present
                if "W" in d:
                    d["W"] = np.array(d["W"], dtype=np.float32)
                if "b" in d:
                    d["b"] = np.array(d["b"], dtype=np.float32)
                if "fp_index" in d:
                    d["fp_index"] = {k: np.array(v, dtype=np.float32) for k, v in d["fp_index"].items()}
                return d
            except Exception:
                pass
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
        if self._redis is not None:
            try:
                self._redis.set(self._k_user(uid), json.dumps(state, cls=NumpyEncoder))
                self._redis.sadd(self._k_users_set(), uid)
                return
            except Exception:
                pass
        with open(self._user_path(uid), "w", encoding="utf-8") as f:
            json.dump(state, f, cls=NumpyEncoder)

    def load_user(self, uid: str) -> Optional[Dict]:
        if self._redis is not None:
            try:
                s = self._redis.get(self._k_user(uid))
                if s is None:
                    return None
                return json.loads(s)
            except Exception:
                pass
        p = self._user_path(uid)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_users(self) -> Dict[str, Dict]:
        if self._redis is not None:
            out: Dict[str, Dict] = {}
            try:
                uids = list(self._redis.smembers(self._k_users_set()) or [])
                for uid in uids:
                    s = self._redis.get(self._k_user(uid))
                    if s is None:
                        continue
                    try:
                        out[uid] = json.loads(s)
                    except Exception:
                        continue
                return out
            except Exception:
                return {}
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
