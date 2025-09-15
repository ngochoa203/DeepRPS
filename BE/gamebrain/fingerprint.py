from __future__ import annotations

from typing import Dict, List
import numpy as np


def _markov_matrix(events: List[Dict]) -> np.ndarray:
    # 3x3 transition of user moves
    M = np.ones((3, 3), dtype=np.float32)  # Laplace smoothing
    prev = None
    for e in events:
        u = int(e.get("u_move", 0))
        if prev is not None:
            M[prev, u] += 1.0
        prev = u
    # row-normalize
    M = M / M.sum(axis=1, keepdims=True)
    return M


def _wsls_bias(events: List[Dict]) -> np.ndarray:
    # win-stay, lose-shift tendencies
    # returns [p_stay_win, p_shift_lose, p_stay_draw]
    if not events:
        return np.zeros(3, dtype=np.float32)
    stay_win = []
    shift_lose = []
    stay_draw = []
    prev_u = None
    prev_res = None
    for e in events:
        u = int(e.get("u_move", 0))
        if prev_u is not None:
            stay = 1.0 if u == prev_u else 0.0
            if prev_res == "win":
                stay_win.append(stay)
            elif prev_res == "lose":
                shift_lose.append(1.0 - stay)
            else:
                stay_draw.append(stay)
        prev_u = u
        prev_res = e.get("result", "draw")
    def avg(x):
        return float(np.mean(x)) if x else 0.0
    return np.array([avg(stay_win), avg(shift_lose), avg(stay_draw)], dtype=np.float32)


def _tempo_stats(events: List[Dict]) -> np.ndarray:
    dts = [int(e.get("dt_ms", 400)) for e in events]
    if not dts:
        return np.zeros(4, dtype=np.float32)
    return np.array([
        float(np.mean(dts)), float(np.std(dts)), float(np.min(dts)), float(np.max(dts))
    ], dtype=np.float32)


def _bias_entropy(events: List[Dict]) -> np.ndarray:
    # bias over moves and its entropy
    counts = np.ones(3, dtype=np.float32)
    for e in events:
        counts[int(e.get("u_move", 0))] += 1.0
    p = counts / counts.sum()
    ent = -np.sum(p * np.log(p))
    return np.concatenate([p, np.array([ent], dtype=np.float32)])


def behavior_fingerprint(events: List[Dict]) -> np.ndarray:
    # 128-dim fingerprint: packed features
    M = _markov_matrix(events).reshape(-1)
    wsls = _wsls_bias(events)
    tempo = _tempo_stats(events)
    bias_ent = _bias_entropy(events)
    # Pad/stack to 128 dims deterministically
    vec = np.concatenate([M, wsls, tempo, bias_ent])  # 9 + 3 + 4 + 4 = 20
    # Repeat and truncate to 128
    reps = int(np.ceil(128 / len(vec)))
    full = np.tile(vec, reps)[:128]
    # Normalize
    norm = np.linalg.norm(full) + 1e-8
    return (full / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))
