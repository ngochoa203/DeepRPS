from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


class LinUCB:
    def __init__(self, d_action: int, d_feature: int, alpha: float = 1.0):
        self.d_action = d_action
        self.d_feature = d_feature
        self.alpha = alpha
        # A = I_d, b = 0_d per action
        self.A = np.array([np.eye(d_feature, dtype=np.float32) for _ in range(d_action)])
        self.b = np.zeros((d_action, d_feature), dtype=np.float32)

    def select_action(self, x: np.ndarray) -> Tuple[int, Dict]:
        # x shape (d_feature,)
        x = x.astype(np.float32)
        p = []
        for a in range(self.d_action):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mu = float(theta @ x)
            ucb = self.alpha * float(np.sqrt(x @ A_inv @ x))
            p.append(mu + ucb)
        a = int(np.argmax(p))
        return a, {"scores": p}

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        x = x.astype(np.float32)
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x

    def to_dict(self) -> Dict:
        return {
            "d_action": self.d_action,
            "d_feature": self.d_feature,
            "alpha": self.alpha,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
        }

    @staticmethod
    def from_dict(d: Dict) -> "LinUCB":
        obj = LinUCB(d_action=int(d["d_action"]), d_feature=int(d["d_feature"]), alpha=float(d.get("alpha", 1.0)))
        obj.A = np.array(d["A"], dtype=np.float32)
        obj.b = np.array(d["b"], dtype=np.float32)
        return obj
