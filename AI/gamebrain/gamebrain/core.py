from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .bandit import LinUCB
from .fingerprint import behavior_fingerprint, cosine_similarity
from .storage import StateStorage
from .utils import softmax, one_hot, best_response_move, entropy


@dataclass
class UserState:
    user_id: str
    ue: np.ndarray  # 64-dim user embedding
    bandit: LinUCB
    ngram_counts_1: np.ndarray  # 3
    ngram_counts_2: np.ndarray  # 3x3 (prev->next)
    ema_freq: np.ndarray  # 3, EMA of user's moves
    last_move: Optional[int] = None
    ema_logloss: float = 1.1  # start slightly worse than uniform logloss (~1.0986)
    ema_decay: float = 0.98
    drift_cooldown: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    # provenance and behavior flags
    seeded_from: Optional[str] = None
    seed_weight: float = 0.0
    adversarial_score: float = 0.0
    # New: user reaction model and strategy-family stats
    ai_to_user_counts: np.ndarray = field(default_factory=lambda: np.ones((3, 3), dtype=np.float32))
    family_stats: Dict[str, Any] = field(default_factory=dict)  # {family: {n, mean}}
    last_family: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "ue": self.ue.tolist(),
            "bandit": self.bandit.to_dict(),
            "ngram_counts_1": self.ngram_counts_1.tolist(),
            "ngram_counts_2": self.ngram_counts_2.tolist(),
            "ema_freq": self.ema_freq.tolist(),
            "last_move": self.last_move,
            "ema_logloss": self.ema_logloss,
            "ema_decay": self.ema_decay,
            "drift_cooldown": self.drift_cooldown,
            "seeded_from": self.seeded_from,
            "seed_weight": self.seed_weight,
            "adversarial_score": self.adversarial_score,
            "ai_to_user_counts": self.ai_to_user_counts.tolist(),
            "family_stats": self.family_stats,
            "last_family": self.last_family,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "UserState":
        return UserState(
            user_id=d["user_id"],
            ue=np.array(d["ue"], dtype=np.float32),
            bandit=LinUCB.from_dict(d["bandit"]),
            ngram_counts_1=np.array(d["ngram_counts_1"], dtype=np.float32),
            ngram_counts_2=np.array(d["ngram_counts_2"], dtype=np.float32),
            ema_freq=np.array(d["ema_freq"], dtype=np.float32),
            last_move=d.get("last_move"),
            ema_logloss=float(d.get("ema_logloss", 1.1)),
            ema_decay=float(d.get("ema_decay", 0.98)),
            drift_cooldown=int(d.get("drift_cooldown", 0)),
            seeded_from=d.get("seeded_from"),
            seed_weight=float(d.get("seed_weight", 0.0)),
            adversarial_score=float(d.get("adversarial_score", 0.0)),
            ai_to_user_counts=np.array(d.get("ai_to_user_counts", np.ones((3, 3)).tolist()), dtype=np.float32),
            family_stats=d.get("family_stats", {}),
            last_family=d.get("last_family"),
        )


class GameBrain:
    """
    GameBrain for Rock-Paper-Scissors.

    - Predicts opponent move distribution p(rock,paper,scissors)
    - Combines n-gram (1-2) + light logistic head with user embedding (64-dim)
    - Contextual bandit (LinUCB) on short-term features
    - Change-point detection via log-loss EMA; adjusts entropy and updates
    - Behavioral re-ID using 128-dim fingerprint and cosine similarity
    """

    def __init__(
        self,
        state_dir: str = "./rps_state",
        remember_history: bool = True,
        bandit_alpha: float = 0.8,
        random_seed: int = 42,
    ):
        self.state_dir = state_dir
        self.remember_history = remember_history
        self.rng = random.Random(random_seed)
        np.random.seed(random_seed)

        os.makedirs(self.state_dir, exist_ok=True)
        self.storage = StateStorage(self.state_dir)

        # Global model params (logistic head). Simple linear on features -> 3 logits.
        # Feature dim: ngram1(3) + ngram2_row(3) + ema(3) + last(3) + ue(64) + short(7) = 83
        self.feature_dim = 3 + 3 + 3 + 3 + 64 + 7
        self.W = np.zeros((3, self.feature_dim), dtype=np.float32)
        self.b = np.zeros(3, dtype=np.float32)

        # Per-user states
        self.users: Dict[str, UserState] = {}

        # Fingerprint index: map user_id -> fingerprint vector (128)
        self.fp_index: Dict[str, np.ndarray] = {}
        # Global prior: distribution of users' first moves (aggregated)
        self.first_move_counts = np.ones(3, dtype=np.float32)

        # Load existing state if any
        if self.remember_history:
            state = self.storage.load_global()
            if state is not None:
                self.W = state.get("W", self.W)
                self.b = state.get("b", self.b)
                self.fp_index = state.get("fp_index", {})
                if "first_move_counts" in state:
                    try:
                        self.first_move_counts = np.array(state["first_move_counts"], dtype=np.float32)
                    except Exception:
                        pass
            # load per-user
            for uid, ud in self.storage.load_all_users().items():
                try:
                    self.users[uid] = UserState.from_dict(ud)
                except Exception:
                    continue

    # ---------------------- Public API ----------------------
    def predict(self, user_hint: Optional[str], ctx: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        uid = self._resolve_user(user_hint, ctx)
        us = self._get_or_create_user(uid)

        # Soft re-ID seeding if ctx suggests similar player (but not strict)
        soft_info = None
        if (ctx and "events" in ctx and ctx["events"]) and (len(us.history) == 0) and (us.seeded_from is None):
            fp = behavior_fingerprint(ctx["events"])
            best_uid, best_sim = None, -1.0
            for kid, vec in self.fp_index.items():
                sim = cosine_similarity(fp, vec)
                if sim > best_sim:
                    best_uid, best_sim = kid, sim
            if best_uid is not None and 0.75 <= best_sim < 0.9 and best_uid in self.users:
                src = self.users[best_uid]
                wseed = float(max(0.0, min(0.85, (best_sim - 0.75) / (0.9 - 0.75))))
                self._warm_start_from(us, src, wseed)
                us.seeded_from, us.seed_weight = best_uid, wseed
                soft_info = {"seeded_from": best_uid, "sim": best_sim, "w": wseed}

        features, short_ctx = self._build_features(us)
        logits = self.W @ features + self.b
        p_ngram = softmax(logits)

        # Bandit proposes an action (our move) given short-term context features
        bandit_action, bandit_meta = us.bandit.select_action(short_ctx)

        # Best response with non-repeat bias on R3
        p_adj = self._apply_nonrepeat_bias(p_ngram, us)
        br_move = best_response_move(p_adj)

        # Short-horizon lookahead
        look_scores = self._lookahead_scores(us, p_adj)
        m_look = int(np.argmax(look_scores))

        # Fuse bandit suggestion with best-response using data-dependent weight
        n_obs = float(us.ngram_counts_1.sum() + us.ngram_counts_2.sum())
        w = min(1.0, max(0.2, n_obs / (n_obs + 20.0)))
        base_choice = br_move if self.rng.random() < w else bandit_action

        # Style-based prediction from user's own transitions
        style_probs = self._user_style_probs(us)
        style_move: Optional[int] = None
        style_conf = 0.0
        if style_probs is not None and us.last_move is not None:
            p_style = self._predict_next_from_style(us.last_move, style_probs)
            style_move = int(best_response_move(p_style))
            style_conf = float(np.max(style_probs))

        # KNN over fingerprint index to infer likely style when data is sparse
        knn_move, knn_info = self._knn_style(uid, us, k=3)

        # Early-round meta rules and low-data blending
        meta_move, meta_policy = self._early_meta(us)
        confidence = min(1.0, (us.ngram_counts_1.sum() + us.ngram_counts_2.sum()) / 50.0)
        if us.seed_weight > 0.0:
            confidence *= (1.0 - 0.4 * us.seed_weight)
        choices = [base_choice, m_look]
        weights = [0.6 * confidence + 0.1, 0.3 + 0.3 * (1 - confidence)]
        # Anti-exploit detection and counter-response
        anti_info = None
        anti_s, anti_move = self._user_counters_last_ai_prob(us)
        is_exploiting = self._exploitation_alert(us)
        
        # Trigger on ANY pattern detection OR exploitation alert
        if (anti_s is not None and anti_s >= 0.10) or is_exploiting:
            if anti_move is not None:
                choices.append(anti_move)
            
            # Aggressive weighting for pattern counter
            base_weight = 2.0 if anti_s is not None and anti_s >= 0.10 else 1.5
            # Scale 2.0..8.0 as anti_s goes 0.10..1.0 (much more aggressive)
            if anti_s is not None:
                weight = base_weight + 6.0 * min(1.0, (anti_s - 0.10) / 0.90)
            else:
                weight = base_weight
                
            # Apply exploitation multipliers
            if is_exploiting:
                weight *= 6.0  # 6x multiplier for exploitation
            if len(us.history) <= 8:  # Extended early rounds
                weight *= 3.0  # 3x early game multiplier
            if len(us.history) >= 2 and us.history[-1]["result"] == "lose":
                weight *= 2.0  # 2x after immediate loss
            
            weights.append(weight)
            anti_info = {"p_counter": anti_s, "anti_move": int(anti_move) if anti_move is not None else None}
            
            # Multi-variant counter-strategies
            variant_threshold = 0.08 if is_exploiting else 0.12  # Even lower threshold
            if anti_s is not None and anti_s >= variant_threshold:
                
                # Variant 1: Triple counter (3rd level thinking)
                if len(us.history) >= 1:
                    last_ai = us.history[-1]["ai_move"]
                    if last_ai is not None:
                        expected_user = (last_ai + 1) % 3  # They counter us
                        counter_counter = (expected_user + 1) % 3  # We counter their counter
                        triple_counter = (counter_counter + 1) % 3  # They counter our counter-counter
                        our_response = (triple_counter + 1) % 3  # We beat their triple thinking
                        choices.append(our_response)
                        weights.append(weight * 0.9)  # High variant weight
                
                # Variant 2: Pattern-breaking randomized strategy
                alt_moves = [m for m in range(3) if m != (anti_move if anti_move is not None else 0)]
                if alt_moves:
                    random_alt = self.rng.choice(alt_moves)
                    choices.append(random_alt)
                    weights.append(weight * 0.8)
                
                # Variant 3: Historical pattern counter
                if len(us.history) >= 3:
                    # Look at user's last 3 moves to predict pattern
                    recent_moves = [h["u_move"] for h in us.history[-3:]]
                    if len(set(recent_moves)) == 1:  # All same
                        expected_next = recent_moves[-1]  # Continue pattern
                        our_counter = (expected_next + 1) % 3
                        choices.append(our_counter)
                        weights.append(weight * 0.85)
                    else:
                        # Mixed pattern - use most common recent move
                        from collections import Counter
                        most_common = Counter(recent_moves).most_common(1)[0][0]
                        our_counter = (most_common + 1) % 3
                        choices.append(our_counter)
                        weights.append(weight * 0.75)
                
                # Variant 4: Meta-level strategy (assume they adapt)
                if len(us.history) >= 1:
                    last_ai = us.history[-1]["ai_move"]
                    if last_ai is not None:
                        # They know we know they counter, so they might switch
                        # Predict they'll try to beat our counter-counter
                        expected_user = (last_ai + 1) % 3
                        they_expect_our_counter = (expected_user + 1) % 3
                        they_might_play = (they_expect_our_counter + 2) % 3  # Beat our expected move
                        our_meta_response = (they_might_play + 1) % 3
                        choices.append(our_meta_response)
                        weights.append(weight * 0.95)  # Highest variant weight
                
                # Variant 5: Anti-anti strategy (reverse psychology)
                if anti_move is not None:
                    # Instead of playing anti_move, play what would lose to their expected counter
                    reverse_move = (anti_move + 2) % 3  # Lose to anti_move
                    choices.append(reverse_move)
                    weights.append(weight * 0.7)  # Medium variant weight
            
            # ALWAYS add pure mathematical counter as ultimate backup
            if len(us.history) >= 1:
                last_ai = us.history[-1]["ai_move"]
                if last_ai is not None:
                    expected_user_move = (last_ai + 1) % 3
                    pure_counter = (expected_user_move + 1) % 3
                    choices.append(pure_counter)
                    weights.append(weight * 1.0)  # Full weight backup
        if meta_move is not None:
            choices.append(meta_move)
            # MASSIVE boost for meta moves when exploitation detected - IMMEDIATE PRIORITY
            base_meta_weight = 0.3 * (1 - confidence) + (0.1 if len(us.history) >= 2 else 0.4)
            if is_exploiting or (anti_s is not None and anti_s >= 0.10):
                # MAXIMUM priority for early meta rules when being exploited
                if len(us.history) <= 4:  # First 4 rounds get extreme priority
                    base_meta_weight *= 8.0  # 8x boost for early meta under exploitation
                else:
                    base_meta_weight *= 4.0  # 4x boost for later meta
            weights.append(base_meta_weight)
        # AI-conditioned user model (how user responds to our move)
        ai_cond = self._ai_conditioned_move(us)
        if ai_cond is not None:
            ai_move2, conf2 = ai_cond
            choices.append(int(ai_move2))
            weights.append(0.2 + 0.4 * conf2)
        # Style-based choice (if confident)
        if style_move is not None:
            choices.append(style_move)
            weights.append(0.2 + 0.3 * min(1.0, style_conf))
        # KNN style-based choice for cold start
        if knn_move is not None and knn_info is not None and knn_info.get("best_sim", 0.0) >= 0.8:
            choices.append(int(knn_move))
            weights.append(0.15 + 0.25 * (1.0 - confidence))

        # EXTREME RESPONSE when exploitation detected - ELIMINATE predictable strategies
        if is_exploiting or (anti_s is not None and anti_s >= 0.10):
            # DESTROY basic strategies completely
            if len(weights) >= 1:
                weights[0] *= 0.02  # Almost eliminate base_choice (BR) - 2%
            if len(weights) >= 2:
                weights[1] *= 0.5   # Reduce lookahead significantly
            
            # MASSIVELY boost all anti-exploitation strategies
            for i in range(2, len(weights)):
                if i < len(choices):
                    # Check if this is one of our anti-exploitation moves
                    current_move = choices[i]
                    if (anti_move is not None and current_move == anti_move) or i >= len(choices) - 6:
                        # This is likely an anti-exploitation strategy
                        weights[i] *= 5.0  # 5x boost for anti-strategies
                    else:
                        # Other strategic moves get moderate boost
                        weights[i] *= 2.5
        tallies = np.zeros(3, dtype=np.float32)
        for c, wgt in zip(choices, weights):
            tallies[int(c)] += wgt
        fused_move = int(np.argmax(tallies))

        # EXTENDED early meta precedence - first 4 rounds when exploited, first 2 normally  
        early_meta_rounds = 4 if (is_exploiting or (anti_s is not None and anti_s >= 0.10)) else 2
        if len(us.history) < early_meta_rounds and meta_move is not None:
            ai_move = int(meta_move)
            policy = meta_policy
            us.last_family = "early"
            eps = 0.0
            meta = {
                "uid": uid,
                "p_opp": p_ngram.tolist(),
                "br_move": int(br_move),
                "bandit": bandit_meta,
                "lookahead": look_scores.tolist(),
                "fused_move": int(fused_move),
                "eps": float(eps),
                "policy": policy,
                "soft_seed": soft_info,
                "anti_exploit": anti_info,
                "first_counts": self.first_move_counts.tolist(),
                "style_probs": style_probs.tolist() if style_probs is not None else None,
                "knn": knn_info,
                "family": us.last_family,
            }
            return int(ai_move), meta

        # Entropy control + adversarial anti-tracking (enhanced)
        eps = self._dynamic_entropy_eps(us)
        
        # MAXIMUM randomness when being exploited for unpredictability
        if is_exploiting or (anti_s is not None and anti_s >= 0.10):
            eps = max(eps, 0.6)  # Minimum 60% randomness when exploited
            # Extra randomness in early rounds for faster learning
            if len(us.history) <= 5:
                eps = max(eps, 0.8)  # 80% randomness in early exploitation
            
        if self._is_adversarial(us) or is_exploiting:
            # OVERRIDE: If we have strong pattern detection (>90% confidence), ALWAYS use it
            if anti_info is not None and anti_move is not None:
                pattern_confidence = anti_info.get("p_counter", 0.0)
                if pattern_confidence >= 0.9:  # 90%+ confidence - FORCE use pattern counter
                    ai_move = anti_move
                    policy = "pattern-override-high-conf"
                elif pattern_confidence >= 0.7:  # 70%+ confidence - likely use pattern counter
                    if self.rng.random() < 0.9:  # 90% chance to use pattern
                        ai_move = anti_move
                        policy = "pattern-override-med-conf"
                    else:
                        ai_move = self.rng.randrange(3)
                        policy = "anti-track-mix"
                else:
                    # Lower confidence - use standard anti-tracking
                    if self.rng.random() < 0.2:  # Reduced predictable BR
                        ai_move = br_move
                        policy = "anti-track-br"
                    elif self.rng.random() < 0.6:
                        ai_move = anti_move  # Use our best counter-strategy
                        policy = "anti-track-counter"
                    else:
                        ai_move = self.rng.randrange(3)  # Pure random
                        policy = "anti-track-mix"
            else:
                # No pattern detected - use standard anti-tracking
                if self.rng.random() < 0.2:
                    ai_move = br_move
                    policy = "anti-track-br"
                else:
                    ai_move = self.rng.randrange(3)
                    policy = "anti-track-mix"
        else:
            # Choose a strategy family with a small UCB bandit to avoid being caught
            family_candidates: Dict[str, Optional[int]] = {
                "br": int(br_move),
                "look": int(m_look),
                "style": int(style_move) if 'style_move' in locals() and style_move is not None else None,
                "knn": int(knn_move) if knn_move is not None else None,
            }
            if anti_info is not None and anti_move is not None:
                family_candidates["anti"] = int(anti_move)
            if ai_cond is not None:
                family_candidates["ai_cond"] = int(ai_cond[0])

            # Build UCB scores with heavy exploitation penalty
            fam_stats = us.family_stats or {}
            total_n = 1.0 + sum(int(fam_stats.get(f, {}).get("n", 0)) for f in fam_stats)
            c = 1.2  # Higher exploration constant
            scores: Dict[str, float] = {}
            exploitation_detected = self._exploitation_alert(us)
            
            for f, mv in family_candidates.items():
                if mv is None:
                    continue
                s = fam_stats.get(f, {})
                n = float(s.get("n", 0.0))
                mean = float(s.get("mean", 0.5))
                ucb = mean + c * math.sqrt(math.log(total_n) / (n + 1.0))
                
                # Heavy penalty for repeating failed strategies
                if exploitation_detected:
                    if us.last_family == f:
                        ucb *= 0.2  # Even more severe penalty for repeating
                    elif f in ["br", "style"]:  # Penalize predictable families
                        ucb *= 0.4  # Even lower penalty
                    elif f in ["anti", "ai_cond"]:  # Massive boost for counter-strategies
                        # Extra boost if we have high confidence pattern
                        pattern_boost = 3.0
                        if anti_info is not None:
                            pattern_confidence = anti_info.get("p_counter", 0.0)
                            if pattern_confidence >= 0.9:
                                pattern_boost = 5.0  # 5x boost for 90%+ confidence
                            elif pattern_confidence >= 0.7:
                                pattern_boost = 4.0  # 4x boost for 70%+ confidence
                        ucb *= pattern_boost
                        
                scores[f] = float(ucb)
                
            # Boost exploration when being exploited
            explore_boost = 2.0 if exploitation_detected else 1.0
            scores["mix"] = 0.5 * explore_boost + c * math.sqrt(math.log(total_n) / 1.0)

            # pick best family available
            fam_choice = max(scores.items(), key=lambda x: x[1])[0]
            if fam_choice == "mix" or family_candidates.get(fam_choice) is None:
                if self.rng.random() < eps:
                    ai_move = self.rng.randrange(3)
                    policy = "explore"
                else:
                    ai_move = fused_move
                    policy = "exploit"
            else:
                mv_sel = family_candidates.get(fam_choice)
                if mv_sel is None:
                    ai_move = fused_move
                    policy = "exploit"
                else:
                    ai_move = int(mv_sel)
                    policy = f"family:{fam_choice}"
            us.last_family = policy

        meta = {
            "uid": uid,
            "p_opp": p_ngram.tolist(),
            "br_move": int(br_move),
            "bandit": bandit_meta,
            "lookahead": look_scores.tolist(),
            "fused_move": int(fused_move),
            "eps": float(eps),
            "policy": policy,
            "soft_seed": soft_info,
            "anti_exploit": anti_info,
            "first_counts": self.first_move_counts.tolist(),
            "style_probs": style_probs.tolist() if style_probs is not None else None,
            "knn": knn_info,
            "family": us.last_family,
        }
        return int(ai_move), meta

    def feedback(
        self,
        user_hint: Optional[str],
        ai_move: int,
        user_move: int,
        dt_ms: int,
        result: str,
    ) -> None:
        """Update models after a round."""
        uid = user_hint if user_hint is not None else "anon"
        us = self._get_or_create_user(uid)

        # Update n-gram counts
        us.ngram_counts_1[user_move] += 1.0
        if us.last_move is not None:
            us.ngram_counts_2[us.last_move, user_move] += 1.0
        us.last_move = user_move

        # Update EMA frequency
        us.ema_freq = 0.9 * us.ema_freq + 0.1 * one_hot(user_move, 3)

        # Update bandit with reward: win=1, draw=0.5, lose=0
        reward = {"win": 1.0, "draw": 0.5, "lose": 0.0}.get(result, 0.5)
        _, short_ctx = self._build_features(us)
        us.bandit.update(short_ctx, ai_move, reward)

        # Update strategy family stats with simple running mean
        if us.last_family:
            fs = us.family_stats.get(us.last_family, {"n": 0, "mean": 0.5})
            n = float(fs.get("n", 0.0))
            mean = float(fs.get("mean", 0.5))
            new_mean = (mean * n + reward) / (n + 1.0)
            us.family_stats[us.last_family] = {"n": n + 1.0, "mean": float(new_mean)}
            # Reset last family after accounting
            us.last_family = None

        # Train logistic head with one SGD step
        features, _ = self._build_features(us)
        logits = self.W @ features + self.b
        p = softmax(logits)
        y = one_hot(user_move, 3)
        # cross-entropy grad
        grad_logits = p - y  # shape (3,)
        # simple learning rate schedule
        lr = 0.1
        self.W -= lr * np.outer(grad_logits, features)
        self.b -= lr * grad_logits

        # Online update of user embedding (subset of features) unless drift locking
        # Feature layout: [n1(3), n2(3), ema(3), last(3), ue(64), short(7)]
        if us.drift_cooldown == 0:
            grad_feat = self.W.T @ grad_logits  # shape (feature_dim,)
            ue_grad = grad_feat[12:12 + 64]
            us.ue -= 0.05 * ue_grad.astype(np.float32)

        # Change-point detection via log-loss EMA
        logloss = -float(np.log(max(1e-8, p[user_move])))
        us.ema_logloss = us.ema_decay * us.ema_logloss + (1 - us.ema_decay) * logloss
        # If logloss exceeds threshold, increase cooldown (drift)
        if us.ema_logloss > 1.2:  # worse than uniform for a bit
            us.drift_cooldown = 5
            # temporarily reduce effect of past
            us.ema_decay = 0.95
        else:
            us.ema_decay = min(0.99, us.ema_decay + 0.001)
            if us.drift_cooldown > 0:
                us.drift_cooldown -= 1

    # Update adversarial detector
        self._update_adversarial(us)

        # Update global first-move stats if this is the user's first recorded round
        if len(us.history) == 0:
            try:
                self.first_move_counts[int(user_move)] += 1.0
            except Exception:
                pass

        # Append to history for fingerprinting (bounded)
        us.history.append({
            "u_move": int(user_move),
            "ai_move": int(ai_move),
            "result": result,
            "dt_ms": int(dt_ms),
        })
        # Update ai->user transition counts (how user responds to our move)
        try:
            self._update_ai_cond_counts(us, ai_move, user_move)
        except Exception:
            pass
        if len(us.history) > 500:
            us.history = us.history[-500:]

        # Update fingerprint index
        fp = behavior_fingerprint(us.history)
        self.fp_index[uid] = fp

        # Save incrementally if allowed
        if self.remember_history:
            self.storage.save_user(uid, us.to_dict())
            self.storage.save_global({
                "W": self.W,
                "b": self.b,
                "fp_index": self.fp_index,
                "first_move_counts": self.first_move_counts,
            })

    # --------- AI-conditioned user model ---------
    def _ai_conditioned_move(self, us: UserState) -> Optional[Tuple[int, float]]:
        # Estimate user's response to our chosen move and pick counter.
        # Instead of using last AI move, predict what we might play and counter their likely response.
        if len(us.history) < 4:
            return None
        
        # Build empirical conditional from ai_to_user_counts
        M = us.ai_to_user_counts.astype(np.float32)
        
        # Check all three possible AI moves and their expected user responses
        best_move = None
        best_conf = 0.0
        
        for potential_ai_move in range(3):
            row = M[potential_ai_move]
            if float(np.sum(row)) <= 3.0:  # Need some data
                continue
                
            p_user = row / float(np.sum(row))
            # Find what beats the most likely user response
            likely_user_move = int(np.argmax(p_user))
            our_counter = (likely_user_move + 1) % 3  # beats their likely response
            conf = float(np.max(p_user))
            
            # Prefer moves with higher confidence and avoid predictable patterns
            if conf > best_conf and conf > 0.4:
                best_move = our_counter
                best_conf = conf
        
        if best_move is not None:
            return int(best_move), best_conf
        return None

    def _update_ai_cond_counts(self, us: UserState, ai_move: int, user_move: int) -> None:
        # Update counts of how user responded to our move in previous round
        if len(us.history) < 2:
            return
        prev_ai = us.history[-2]["ai_move"]
        if prev_ai is None:
            return
        us.ai_to_user_counts[int(prev_ai), int(user_move)] += 1.0

    def save(self) -> None:
        if not self.remember_history:
            return
        self.storage.save_global({
            "W": self.W,
            "b": self.b,
            "fp_index": self.fp_index,
            "first_move_counts": self.first_move_counts,
        })
        for uid, us in self.users.items():
            self.storage.save_user(uid, us.to_dict())

    # ---------------------- Internal helpers ----------------------
    def _resolve_user(self, user_hint: Optional[str], ctx: Optional[Dict[str, Any]]) -> str:
        if user_hint is not None:
            return user_hint
        # Try behavioral re-ID from ctx events
        if ctx and "events" in ctx and ctx["events"]:
            fp = behavior_fingerprint(ctx["events"])
            # Find best match in index
            best_uid, best_sim = None, -1.0
            for uid, vec in self.fp_index.items():
                sim = cosine_similarity(fp, vec)
                if sim > best_sim:
                    best_uid, best_sim = uid, sim
            if best_uid is not None and best_sim >= 0.88:
                # soft/strict handling
                return best_uid
        return "anon"

    def _get_or_create_user(self, uid: str) -> UserState:
        if uid in self.users:
            return self.users[uid]
        # create new user state
        us = UserState(
            user_id=uid,
            ue=np.random.normal(scale=0.05, size=(64,)).astype(np.float32),
            bandit=LinUCB(d_action=3, d_feature=7, alpha=0.8),
            ngram_counts_1=np.ones(3, dtype=np.float32),
            ngram_counts_2=np.ones((3, 3), dtype=np.float32),
            ema_freq=np.ones(3, dtype=np.float32) / 3.0,
            last_move=None,
        )
        # Load if persisted
        if self.remember_history:
            d = self.storage.load_user(uid)
            if d is not None:
                us = UserState.from_dict(d)
        self.users[uid] = us
        return us

    def _build_features(self, us: UserState) -> Tuple[np.ndarray, np.ndarray]:
        # n-gram 1
        n1 = us.ngram_counts_1 / max(1.0, us.ngram_counts_1.sum())
        # n-gram 2: if last move known, use that row; otherwise average
        if us.last_move is None:
            n2_row = us.ngram_counts_2.mean(axis=0)
        else:
            n2_row = us.ngram_counts_2[us.last_move]
        n2 = n2_row / max(1.0, n2_row.sum())
        # ema freq
        ema = us.ema_freq / max(1e-6, us.ema_freq.sum())
        # last move one-hot
        last = one_hot(us.last_move if us.last_move is not None else -1, 3)

        # short-term features for bandit: last 3 outcomes, tempo stats
        short = self._short_term_features(us.history[-5:])

        feat = np.concatenate([n1, n2, ema, last, us.ue, short]).astype(np.float32)
        return feat, short

    def _short_term_features(self, recent: List[Dict[str, Any]]) -> np.ndarray:
        # outcomes encoded as reward
        rewards = [{"win": 1.0, "draw": 0.5, "lose": 0.0}[h["result"]] for h in recent] if recent else []
        if len(rewards) == 0:
            mu, sg = 0.5, 0.0
        else:
            mu = float(np.mean(rewards))
            sg = float(np.std(rewards))
        # tempo: mean/std of dt_ms
        dts = [h.get("dt_ms", 400) for h in recent]
        tmu = float(np.mean(dts)) if dts else 400.0
        tsg = float(np.std(dts)) if len(dts) > 1 else 0.0
        # last user move one-hot (3) to bias bandit exploration
        last_u = one_hot(recent[-1]["u_move"] if recent else -1, 3)
        return np.array([mu, sg, tmu / 1000.0, tsg / 1000.0] + last_u.tolist(), dtype=np.float32)

    def _dynamic_entropy_eps(self, us: UserState) -> float:
        base = 0.05
        if us.drift_cooldown > 0:
            return min(0.25, base + 0.05 * us.drift_cooldown)
        # higher entropy when low data or high entropy of user
        n_obs = us.ngram_counts_1.sum()
        user_ent = entropy(us.ema_freq / max(1e-6, us.ema_freq.sum()))
        scale = 0.15 * (1.0 / (1.0 + n_obs / 30.0)) + 0.05 * (user_ent / math.log(3))
        return float(min(0.25, base + scale))

    # ---------------------- New Logic: meta, lookahead, transfer, adversarial ----------------------
    def _early_meta(self, us: UserState) -> Tuple[Optional[int], str]:
        n = len(us.history)
        if n == 0:
            # Round 1: Strategic first move avoiding Paper, prioritize Rock for strength
            m0 = int(np.argmax(self.first_move_counts))
            
            # Be more aggressive in first move selection
            if m0 == 1:  # If users commonly play Paper first
                move = 2  # Play Scissors to beat Paper
            else:
                beat = (m0 + 1) % 3  # Beat the most common first move
                move = beat if beat != 1 else 0  # Avoid Paper, default to Rock
            
            return int(move), "meta-r1-aggressive-beat"
            
        if n == 1:
            last = us.history[-1]
            last_ai, last_u, res = last["ai_move"], last["u_move"], last["result"]
            
            # INSTANT EXPLOITATION RESPONSE - Round 2 is critical
            if res == "lose":  # We lost round 1 - IMMEDIATE COUNTER-STRATEGY
                # Assume they'll repeat their winning move OR counter our last move
                if last_u == (last_ai + 1) % 3:  # They countered us
                    # They're likely to counter again - use double counter
                    expected_user = (last_ai + 1) % 3  # They'll counter our R1 move again
                    move = (expected_user + 1) % 3  # We counter their expected counter
                    return move, "meta-r2-anti-counter"
                else:
                    # They played something else and won - assume they'll repeat
                    move = (last_u + 1) % 3  # Counter their likely repeat
                    return move, "meta-r2-anti-repeat"
                    
            elif res == "draw":  # Draw means they might have a pattern
                # Assume they'll either repeat or counter our R1 move
                repeat_counter = (last_u + 1) % 3
                ai_counter = (last_ai + 1) % 3
                our_response = (ai_counter + 1) % 3
                # Choose the more aggressive response
                move = repeat_counter if self.rng.random() < 0.6 else our_response
                return move, "meta-r2-draw-aggressive"
                
            else:  # res == "win" - We won but still be cautious
                # They might try to counter our winning move
                counter = (last_ai + 1) % 3  # They might counter our R1 winner
                move = (counter + 1) % 3    # We counter their expected counter
                return move, "meta-r2-defend-win"
                
        if n == 2:
            # Round 3 - Look for immediate patterns
            u_last2 = [us.history[-2]["u_move"], us.history[-1]["u_move"]]
            results = [us.history[-2]["result"], us.history[-1]["result"]]
            
            # If they won or drew either of the last 2 rounds, assume continuation
            if "lose" in results or "draw" in results:
                if u_last2[0] == u_last2[1]:  # Same move twice
                    # Definitely counter their pattern
                    move = (u_last2[1] + 1) % 3  # Counter their repeated move
                    return move, "meta-r3-pattern-break"
                else:
                    # Mixed moves but they had success - counter their most recent
                    recent_winner = u_last2[1] if results[1] in ["lose", "draw"] else u_last2[0]
                    move = (recent_winner + 1) % 3
                    return move, "meta-r3-counter-success"
            else:
                # Standard pattern detection
                if u_last2[0] == u_last2[1]:
                    move = (u_last2[1] + 2) % 3  # Beat expected continuation
                    return move, "meta-r3-beat-repeat"
                    
        if n == 3:
            # Round 4 - Extended early pattern detection
            recent_moves = [h["u_move"] for h in us.history[-3:]]
            recent_results = [h["result"] for h in us.history[-3:]]
            
            # If they had any success in last 3 rounds, predict their next move
            if "lose" in recent_results or "draw" in recent_results:
                # Find their most successful recent move
                success_moves = []
                for i, result in enumerate(recent_results):
                    if result in ["lose", "draw"]:
                        success_moves.append(recent_moves[i])
                
                if success_moves:
                    # Counter their most common successful move
                    from collections import Counter
                    most_common = Counter(success_moves).most_common(1)[0][0]
                    move = (most_common + 1) % 3
                    return move, "meta-r4-counter-success-pattern"
                    
        return None, "meta-none"

    def _apply_nonrepeat_bias(self, p: np.ndarray, us: UserState) -> np.ndarray:
        p = p.astype(np.float32).copy()
        if len(us.history) >= 2:
            u_last2 = [us.history[-2]["u_move"], us.history[-1]["u_move"]]
            if u_last2[0] == u_last2[1]:
                p[u_last2[1]] *= 0.85
                s = float(np.sum(p));
                if s > 0:
                    p /= s
        return p

    def _lookahead_scores(self, us: UserState, p_opp: np.ndarray) -> np.ndarray:
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float32)
        scores = np.zeros(3, dtype=np.float32)
        beta = 0.6
        for m in range(3):
            ev0 = float(A[m] @ p_opp)
            p_win = float(p_opp[(m + 2) % 3])
            p_draw = float(p_opp[m])
            p_lose = float(p_opp[(m + 1) % 3])

            def next_p(cond: str) -> np.ndarray:
                base = p_opp.copy()
                if cond == "win":
                    t = np.zeros(3, dtype=np.float32); t[(m + 1) % 3] = 1.0
                    return 0.6 * t + 0.4 * base
                if cond == "lose":
                    t = np.zeros(3, dtype=np.float32); t[(m + 1) % 3] = 1.0
                    return 0.55 * base + 0.45 * t
                t = base.copy(); t[m] *= 0.3; t = t / np.sum(t)
                return t

            p1 = next_p("win"); br1 = best_response_move(p1); ev1 = float(A[br1] @ p1)
            p2 = next_p("lose"); br2 = best_response_move(p2); ev2 = float(A[br2] @ p2)
            p3 = next_p("draw"); br3 = best_response_move(p3); ev3 = float(A[br3] @ p3)
            ev_future = p_win * ev1 + p_lose * ev2 + p_draw * ev3
            scores[m] = ev0 + beta * ev_future
        return scores

    def _warm_start_from(self, target: UserState, src: UserState, w: float) -> None:
        target.ngram_counts_1 = (1 - w) * target.ngram_counts_1 + w * src.ngram_counts_1
        target.ngram_counts_2 = (1 - w) * target.ngram_counts_2 + w * src.ngram_counts_2
        target.ema_freq = (1 - w) * target.ema_freq + w * src.ema_freq
        target.ue = ((1 - w) * target.ue + w * src.ue).astype(np.float32)

    def _update_adversarial(self, us: UserState) -> None:
        recent = us.history[-12:]
        if len(recent) < 6:
            return
        counts = np.zeros(3, dtype=np.float32)
        switches = 0
        last = None
        for h in recent:
            m = h["u_move"]; counts[m] += 1
            if last is not None and last != m:
                switches += 1
            last = m
        p = counts / max(1.0, counts.sum())
        uniformity = float(1.0 - np.std(p) / (1/np.sqrt(12)))
        switch_rate = switches / max(1, len(recent) - 1)
        score = 0.0
        if uniformity > 0.9 and 0.5 < switch_rate < 0.95:
            score += 0.5
        if entropy(p) > 1.05:
            score += 0.3
        us.adversarial_score = max(0.0, min(1.0, 0.8 * us.adversarial_score + 0.2 * score))

    def _is_adversarial(self, us: UserState) -> bool:
        return us.adversarial_score > 0.6

    def _user_counters_last_ai_prob(self, us: UserState, k: int = 20) -> Tuple[Optional[float], Optional[int]]:
        # IMMEDIATE FIRST-MOVE LEARNING SYSTEM
        if len(us.history) < 1:
            return None, None
        
        # INSTANT RESPONSE AFTER ROUND 1 - Assume any successful move will be repeated
        if len(us.history) == 1:
            first_round = us.history[0]
            ai_move = first_round.get("ai_move")
            user_move = first_round.get("u_move") 
            result = first_round.get("result")
            
            if ai_move is not None and user_move is not None:
                # If user won or drew round 1, IMMEDIATELY assume they'll repeat their strategy
                if result in ["lose", "draw"]:  # User won or drew
                    if user_move == (ai_move + 1) % 3:  # They countered us
                        # MAXIMUM confidence - they'll definitely try countering again
                        return 0.95, (user_move + 1) % 3  # Counter their expected counter
                    else:
                        # They played something else and succeeded - assume repeat  
                        return 0.90, (user_move + 1) % 3  # Counter their likely repeat
                
        # IMMEDIATE PATTERN DETECTION - React to ANY hint of exploitation
        recent = us.history[-min(k, 8):]  # Look at fewer moves for faster response
        
        # Track patterns with EXTREME weighting for recent success
        lag1_cnt = lag2_cnt = copy_cnt = beat_cnt = repeat_cnt = total = 0
        
        for i in range(1, len(recent)):
            prev_ai = recent[i-1]["ai_move"]
            user_now = recent[i]["u_move"]
            result = recent[i]["result"]
            if prev_ai is None:
                continue
            
            # MASSIVE weight for recent successful moves (wins/draws for user)
            base_weight = 3.0 if result in ["lose", "draw"] else 1.0
            # Exponential decay with success bonus
            age_weight = (0.8 ** (len(recent) - 1 - i))
            weight = base_weight * age_weight * 4.0  # Overall 4x boost
            
            total += weight
            
            # Detect all pattern types
            if user_now == (prev_ai + 1) % 3:  # Counter to our last move
                lag1_cnt += weight
            elif user_now == prev_ai:  # Copy our last move  
                copy_cnt += weight
            elif user_now == (prev_ai + 2) % 3:  # Beat our last move
                beat_cnt += weight
                
            # Check for repeating their last successful move
            if i > 0:
                prev_user = recent[i-1]["u_move"]
                if user_now == prev_user and result in ["lose", "draw"]:
                    repeat_cnt += weight * 2.0  # Extra weight for repeating winners
                    
            # Lag-2 pattern (counter to 2 moves ago)
            if i >= 2:
                prev2_ai = recent[i-2]["ai_move"]  
                if prev2_ai is not None and user_now == (prev2_ai + 1) % 3:
                    lag2_cnt += weight
        
        if total < 0.5:  # Trigger almost immediately
            return None, None
        
        # ALTERNATING & CYCLE PATTERN DETECTION
        alt_2_cnt = alt_3_cnt = cycle_3_cnt = 0.0
        
        if len(recent) >= 4:
            user_moves = [h["u_move"] for h in recent]
            results = [h["result"] for h in recent]
            
            # 2-move alternating pattern (A-B-A-B like Scissors-Paper-Scissors-Paper)
            for i in range(2, len(user_moves)):
                if user_moves[i] == user_moves[i-2]:  # Same as 2 moves ago
                    weight = 6.0 if results[i] in ["lose", "draw"] else 2.0  # Higher weight for alternating
                    alt_2_cnt += weight
            
            # 3-move alternating pattern (A-B-C-A-B-C)
            if len(user_moves) >= 6:
                for i in range(3, len(user_moves)):
                    if user_moves[i] == user_moves[i-3]:  # Same as 3 moves ago
                        weight = 6.0 if results[i] in ["lose", "draw"] else 2.0
                        cycle_3_cnt += weight
        
        # Calculate probabilities including alternating patterns    
        lag1_prob = lag1_cnt / total
        lag2_prob = lag2_cnt / total
        copy_prob = copy_cnt / total  
        beat_prob = beat_cnt / total
        repeat_prob = repeat_cnt / total
        alt_2_prob = alt_2_cnt / max(total, 1.0)  # Prevent division by zero
        cycle_3_prob = cycle_3_cnt / max(total, 1.0)
        
        # Find strongest pattern including alternating
        max_prob = max(lag1_prob, lag2_prob, copy_prob, beat_prob, repeat_prob, alt_2_prob, cycle_3_prob)
        
        # ULTRA-LOW threshold - activate on ANY hint of pattern (even 1 occurrence)
        min_threshold = 0.08 if len(us.history) <= 3 else 0.05  # Even lower for early rounds
        
        if max_prob < min_threshold:
            return None, None
            
        prev_ai = us.history[-1]["ai_move"] if us.history else None
        last_user = us.history[-1]["u_move"] if us.history else None
        
        if prev_ai is None:
            return None, None
        
        # INSTANT MAXIMUM COUNTER-STRATEGY INCLUDING ALTERNATING PATTERNS
        confidence_boost = 2.0  # Double reported confidence
        
        if alt_2_prob == max_prob and len(us.history) >= 2:
            # 2-move alternating pattern detected (e.g., Scissors-Paper-Scissors-Paper)
            user_moves = [h["u_move"] for h in us.history[-4:]]  # Look at last 4 moves
            if len(user_moves) >= 2:
                # Predict next move in alternating pattern
                if len(user_moves) % 2 == 0:  # Even position, expect first move of pattern
                    expected_user = user_moves[-2]  # Same as 2 moves ago
                else:  # Odd position, expect second move of pattern  
                    expected_user = user_moves[-1] if len(user_moves) >= 1 else 0
                anti_move = (expected_user + 1) % 3  # Counter their expected move
                return min(alt_2_prob * confidence_boost, 1.0), anti_move
                
        elif cycle_3_prob == max_prob and len(us.history) >= 3:
            # 3-move cycle pattern detected (A-B-C-A-B-C)
            user_moves = [h["u_move"] for h in us.history[-6:]]  # Look at last 6 moves
            if len(user_moves) >= 3:
                # Predict next move in 3-move cycle
                cycle_pos = len(user_moves) % 3
                expected_user = user_moves[-(3-cycle_pos)] if (3-cycle_pos) <= len(user_moves) else user_moves[-1]
                anti_move = (expected_user + 1) % 3
                return min(cycle_3_prob * confidence_boost, 1.0), anti_move
                
        elif repeat_prob == max_prob and last_user is not None:
            # They're repeating successful moves - counter directly
            return min(repeat_prob * confidence_boost, 1.0), (last_user + 1) % 3
            
        elif lag1_prob == max_prob:
            # Counter-to-AI pattern - use multi-level counter
            expected_user = (prev_ai + 1) % 3
            if len(us.history) >= 2 and lag1_prob >= 0.2:  # Quick 3rd level escalation
                counter_counter = (expected_user + 1) % 3
                anti_move = (counter_counter + 1) % 3  # 3rd level
            else:
                anti_move = (expected_user + 1) % 3  # Counter-counter
            return min(lag1_prob * confidence_boost, 1.0), anti_move
            
        elif beat_prob == max_prob:
            # Beat-AI pattern
            expected_user = (prev_ai + 2) % 3
            anti_move = (expected_user + 1) % 3  
            return min(beat_prob * confidence_boost, 1.0), anti_move
            
        elif copy_prob == max_prob:
            # Copy-AI pattern  
            anti_move = (prev_ai + 1) % 3
            return min(copy_prob * confidence_boost, 1.0), anti_move
            
        elif lag2_prob == max_prob and len(us.history) >= 2:
            # Lag-2 counter pattern
            prev2_ai = us.history[-2]["ai_move"]
            if prev2_ai is not None:
                expected_user = (prev2_ai + 1) % 3
                anti_move = (expected_user + 1) % 3
                return min(lag2_prob * confidence_boost, 1.0), anti_move
                
        return None, None

    # --------- Style detection & KNN inference ---------
    def _user_style_probs(self, us: UserState, k: int = 12) -> Optional[np.ndarray]:
        """Estimate user's style relative to their previous move.
        Returns probs for [follow, backward, forward, sticky].
        """
        recent = us.history[-k:]
        if len(recent) < 4:
            return None
        follow = back = fwd = sticky = 0
        for i in range(1, len(recent)):
            u_prev = recent[i-1]["u_move"]
            u_now = recent[i]["u_move"]
            res_prev = recent[i-1]["result"]
            if u_now == u_prev:
                follow += 1
                if res_prev != "lose":
                    sticky += 1
            if u_now == (u_prev + 2) % 3:
                back += 1
            if u_now == (u_prev + 1) % 3:
                fwd += 1
        tot = follow + back + fwd + sticky
        if tot <= 0:
            return None
        p = np.array([follow, back, fwd, sticky], dtype=np.float32)
        p = p / float(np.sum(p))
        return p

    def _predict_next_from_style(self, last_user_move: int, style_probs: np.ndarray) -> np.ndarray:
        """Map style probabilities to predicted next-move distribution."""
        p = np.zeros(3, dtype=np.float32)
        p[last_user_move] += float(style_probs[0])  # follow
        p[(last_user_move + 2) % 3] += float(style_probs[1])  # backward
        p[(last_user_move + 1) % 3] += float(style_probs[2])  # forward
        p[last_user_move] += float(style_probs[3])  # sticky ~ follow
        s = float(np.sum(p))
        if s <= 0:
            return np.ones(3, dtype=np.float32) / 3.0
        return p / s

    def _knn_style(self, uid: str, us: UserState, k: int = 3) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        if not self.fp_index:
            return None, None
        # build fingerprint for current user if has history
        if len(us.history) == 0:
            return None, None
        fp = behavior_fingerprint(us.history)
        sims: List[Tuple[str, float]] = []
        for nid, vec in self.fp_index.items():
            if nid == uid:
                continue
            sim = float(cosine_similarity(fp, vec))
            sims.append((nid, sim))
        if not sims:
            return None, None
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:k]
        agg = np.zeros(4, dtype=np.float32)
        used = 0
        best_sim = 0.0
        for nid, sim in top:
            best_sim = max(best_sim, sim)
            if nid not in self.users or sim < 0.8:
                continue
            nv = self.users[nid]
            sp = self._user_style_probs(nv)
            if sp is None:
                continue
            agg += float(sim) * sp
            used += 1
        if used == 0:
            return None, {"used": 0, "best_sim": best_sim}
        s = float(np.sum(agg))
        if s > 0:
            agg /= s
        if us.last_move is None:
            return None, {"used": used, "best_sim": best_sim}
        p_next = self._predict_next_from_style(us.last_move, agg)
        move = int(best_response_move(p_next))
        return move, {"used": used, "best_sim": best_sim, "agg": agg.tolist()}

    def _exploitation_alert(self, us: UserState) -> bool:
        # INSTANT EXPLOITATION DETECTION - React immediately from round 1
        history = us.history
        
        # FIRST ROUND ALERT - If we lose or draw round 1, assume exploitation
        if len(history) == 1:
            result = history[0]["result"]
            if result in ["lose", "draw"]:
                return True  # Immediate alert on any non-win
        
        # IMMEDIATE ALERT - Any pattern of non-wins
        if len(history) >= 2:
            # Check last 2-6 rounds with extreme sensitivity
            recent = history[-min(6, len(history)):]
            loses = sum(1 for h in recent if h["result"] == "lose")
            draws = sum(1 for h in recent if h["result"] == "draw")
            
            # ULTRA-sensitive thresholds
            lose_rate = loses / len(recent) 
            non_win_rate = (loses + draws) / len(recent)
            
            # Trigger on ANY significant non-winning
            if lose_rate >= 0.2 or non_win_rate >= 0.3:  # 20% losses OR 30% non-wins
                return True
            
            # Special trigger for early rounds - even 1 loss after 2+ rounds
            if len(recent) <= 4 and loses >= 1:
                return True
        
        # ADVANCED PATTERN DETECTION - Including alternating and cycle patterns
        if len(history) >= 2:
            recent = history[-min(8, len(history)):]  # Look at more history for cycles
            
            patterns = {"counter": 0.0, "copy": 0.0, "beat": 0.0, "repeat": 0.0, "alternating": 0.0, "cycle": 0.0}
            
            # Standard pattern detection
            for i in range(1, len(recent)):
                prev_ai = recent[i-1]["ai_move"]
                user_now = recent[i]["u_move"]
                result = recent[i]["result"]
                
                if prev_ai is not None:
                    # Extra weight for successful user moves
                    weight = 3.0 if result in ["lose", "draw"] else 1.0
                    
                    if user_now == (prev_ai + 1) % 3:  # counter
                        patterns["counter"] += weight
                    elif user_now == prev_ai:  # copy
                        patterns["copy"] += weight  
                    elif user_now == (prev_ai + 2) % 3:  # beat
                        patterns["beat"] += weight
                
                # Check for repeating successful moves
                if i > 0:
                    prev_user = recent[i-1]["u_move"]
                    if user_now == prev_user and result in ["lose", "draw"]:
                        patterns["repeat"] += weight
            
            # ALTERNATING PATTERN DETECTION (A-B-A-B or A-B-C-A-B-C)
            if len(recent) >= 4:
                user_moves = [h["u_move"] for h in recent]
                results = [h["result"] for h in recent]
                
                # Check 2-move alternating (A-B-A-B)
                alt_2_matches = 0
                for i in range(2, len(user_moves)):
                    if user_moves[i] == user_moves[i-2]:  # Same as 2 moves ago
                        weight = 4.0 if results[i] in ["lose", "draw"] else 1.0
                        alt_2_matches += weight
                
                if alt_2_matches >= 2.0:  # At least 2 alternations
                    patterns["alternating"] += alt_2_matches
                
                # Check 3-move cycle (A-B-C-A-B-C)
                if len(user_moves) >= 6:
                    cycle_3_matches = 0
                    for i in range(3, len(user_moves)):
                        if user_moves[i] == user_moves[i-3]:  # Same as 3 moves ago
                            weight = 4.0 if results[i] in ["lose", "draw"] else 1.0
                            cycle_3_matches += weight
                    
                    if cycle_3_matches >= 2.0:  # At least 2 cycle matches
                        patterns["cycle"] += cycle_3_matches
            
            # Trigger on ANY weighted pattern >= 1.5 (even single successful occurrence)
            if any(score >= 1.5 for score in patterns.values()):
                return True
        
        # LOGLOSS EXTREME SENSITIVITY
        if us.ema_logloss > 1.08:  # Even lower threshold
            return True
        
        # WIN RATE MONITORING - Should be dominating
        if len(history) >= 3:
            recent = history[-min(8, len(history)):]
            wins = sum(1 for h in recent if h["result"] == "win")
            win_rate = wins / len(recent)
            
            # Different thresholds based on game count
            if len(recent) <= 3 and win_rate < 0.67:  # 67% in first 3
                return True
            elif len(recent) <= 5 and win_rate < 0.6:   # 60% in first 5  
                return True
            elif win_rate < 0.7:  # 70% after that
                return True
                
        return False
