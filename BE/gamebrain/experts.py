from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Move encoding: 0=Rock, 1=Paper, 2=Scissors


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    s = float(np.sum(p))
    if s <= 0 or not np.isfinite(s):
        return np.ones(3, dtype=np.float32) / 3.0
    return p / s


def freq_expert(ngram_counts_1: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Dirichlet-smoothed marginal frequency of user moves."""
    p = np.asarray(ngram_counts_1, dtype=np.float32) + float(alpha)
    return _normalize(p)


def markov1_expert(ngram_counts_2: np.ndarray, last_move: Optional[int], alpha: float = 0.5) -> np.ndarray:
    """p(next | last) with Laplace smoothing; if last unknown, average rows."""
    if last_move is None:
        row = np.mean(ngram_counts_2, axis=0)
    else:
        row = ngram_counts_2[int(last_move)]
    p = np.asarray(row, dtype=np.float32) + float(alpha)
    return _normalize(p)


def wsls_expert(history: List[Dict], default: Optional[np.ndarray] = None) -> np.ndarray:
    """Win-Stay/Lose-Shift heuristic: if user just won (AI lose), predict repeat;
    if user just lost (AI win), predict counter to AI last; draw -> slight repeat bias.
    """
    if not history:
        return np.ones(3, dtype=np.float32) / 3.0 if default is None else default
    last = history[-1]
    ai = last.get("ai_move")
    u = last.get("u_move")
    if u is None:
        return np.ones(3, dtype=np.float32) / 3.0 if default is None else default
    res = last.get("result")
    p = np.ones(3, dtype=np.float32) * 0.001
    if res == "lose":  # AI lost, user won => likely repeat
        p[int(u)] += 2.0
    elif res == "win":  # AI won, user lost => likely counter to our move
        if ai is not None:
            p[(int(ai) + 1) % 3] += 2.2
    else:  # draw
        p[int(u)] += 1.0
        if ai is not None:
            p[(int(ai) + 1) % 3] += 0.6
    return _normalize(p)


def cycle_expert(history: List[Dict]) -> np.ndarray:
    """Detect short cycles (length 3) and project next user move probability."""
    if len(history) < 3:
        return np.ones(3, dtype=np.float32) / 3.0
    u_moves = [h.get("u_move") for h in history[-6:] if h.get("u_move") is not None]
    counts = np.zeros(3, dtype=np.float32)
    # count matches with lag 3
    for i in range(3, len(u_moves)):
        if u_moves[i] == u_moves[i - 3]:
            prev = u_moves[i - 3]
            if prev is not None:
                counts[int(prev)] += 1.0
    if float(np.sum(counts)) <= 0:
        return np.ones(3, dtype=np.float32) / 3.0
    return _normalize(counts)


def alternating2_expert(history: List[Dict]) -> np.ndarray:
    """Detect 2-move alternation A-B-A-B and project next user move."""
    if len(history) < 3:
        return np.ones(3, dtype=np.float32) / 3.0
    u = [h.get("u_move") for h in history[-6:] if h.get("u_move") is not None]
    if len(u) < 3:
        return np.ones(3, dtype=np.float32) / 3.0
    score = 0.0
    for i in range(2, len(u)):
        if u[i] == u[i - 2] and u[i] != u[i - 1]:
            score += 1.0
    if score <= 0:
        return np.ones(3, dtype=np.float32) / 3.0
    # Next expected user move equals move from two steps ago
    p = np.ones(3, dtype=np.float32) * 1e-3
    if u[-2] is None:
        return np.ones(3, dtype=np.float32) / 3.0
    p[int(u[-2])] += 1.5 * score
    return _normalize(p)


def recent_expert(history: List[Dict], k: int = 5) -> np.ndarray:
    """Distribution over the last k user moves (forgetting)."""
    if not history:
        return np.ones(3, dtype=np.float32) / 3.0
    recent = history[-k:] if len(history) >= k else history
    w = np.array([0.6 ** (len(recent) - 1 - i) for i in range(len(recent))], dtype=np.float32)
    counts = np.zeros(3, dtype=np.float32)
    for wi, h in zip(w, recent):
        um = h.get("u_move")
        if um is None:
            continue
        counts[int(um)] += wi
    return _normalize(counts)


def population_prior_expert(first_move_counts: np.ndarray) -> np.ndarray:
    """Global prior over user moves (useful for round 1-2)."""
    return _normalize(np.asarray(first_move_counts, dtype=np.float32))


def antibot_counter_expert(history: List[Dict]) -> np.ndarray:
    """If user often plays counter to our last move, bias towards that next user move."""
    if len(history) < 2:
        return np.ones(3, dtype=np.float32) / 3.0
    cnt = np.zeros(3, dtype=np.float32)
    total = 0.0
    for i in range(1, len(history)):
        prev_ai = history[i - 1]["ai_move"]
        u_now = history[i]["u_move"]
        if prev_ai is None:
            continue
        total += 1.0
        if u_now == (int(prev_ai) + 1) % 3:
            cnt[u_now] += 1.0
    if total <= 0:
        return np.ones(3, dtype=np.float32) / 3.0
    # Slight smoothing
    cnt += 0.5
    return _normalize(cnt)


def ai_policy_from_user_dist(p_user: np.ndarray) -> np.ndarray:
    """Map user move distribution to our move distribution by 'beating' transformation."""
    p_user = _normalize(p_user)
    p_ai = np.zeros(3, dtype=np.float32)
    for j in range(3):
        p_ai[(j + 1) % 3] += float(p_user[j])
    return _normalize(p_ai)


EXPERT_REGISTRY = (
    "PopulationPrior",
    "Freq",
    "Markov1",
    "WSLS",
    "Recent",
    "Alternating2",
    "Cycle3",
    "AntiBotCounter",
)


def compute_all_experts(
    *,
    history: List[Dict],
    ngram_counts_1: np.ndarray,
    ngram_counts_2: np.ndarray,
    last_move: Optional[int],
    first_move_counts: np.ndarray,
) -> Tuple[List[str], List[np.ndarray]]:
    """Return expert names and their user-move distributions p(R,P,S)."""
    names: List[str] = list(EXPERT_REGISTRY)
    preds: List[np.ndarray] = []
    preds.append(population_prior_expert(first_move_counts))
    preds.append(freq_expert(ngram_counts_1, alpha=1.0))
    preds.append(markov1_expert(ngram_counts_2, last_move, alpha=0.5))
    preds.append(wsls_expert(history))
    preds.append(recent_expert(history, k=5))
    preds.append(alternating2_expert(history))
    preds.append(cycle_expert(history))
    preds.append(antibot_counter_expert(history))
    return names, preds
