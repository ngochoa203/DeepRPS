from __future__ import annotations

import numpy as np

MOVES = [0, 1, 2]  # 0=Rock,1=Paper,2=Scissors


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = x.astype(np.float32) / max(1e-6, temperature)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= i < n:
        v[i] = 1.0
    return v


def best_response_move(p_opp: np.ndarray) -> int:
    # If opp plays Rock/Paper/Scissors with probs p0,p1,p2, best response is
    # choose move that beats the most likely opponent move
    # Rock beats Scissors, Paper beats Rock, Scissors beats Paper
    # indices: 0=Rock,1=Paper,2=Scissors
    # We can compute expected value of each of our moves and pick argmax
    # Payoff matrix A[i, j] = result of playing i against j (1 win, 0 draw, -1 lose)
    A = np.array([
        [0, -1, 1],   # our Rock vs [R,P,S]
        [1, 0, -1],   # our Paper
        [-1, 1, 0],   # our Scissors
    ], dtype=np.float32)
    ev = A @ p_opp.astype(np.float32)
    return int(np.argmax(ev))


def entropy(p: np.ndarray) -> float:
    p = p.astype(np.float32)
    p = np.clip(p, 1e-8, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p)))
