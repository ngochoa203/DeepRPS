from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .bandit import LinUCB
from .experts import (
    compute_all_experts,
    ai_policy_from_user_dist,
    EXPERT_REGISTRY,
)
from .fingerprint import behavior_fingerprint, cosine_similarity
from .storage import StateStorage
from .utils import softmax, one_hot, best_response_move, MOVES, entropy


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
    # EXPERT MIXTURE (per-user posterior)
    expert_w: np.ndarray = field(default_factory=lambda: np.ones(len(EXPERT_REGISTRY), dtype=np.float32) / float(len(EXPERT_REGISTRY)))
    expert_eta: float = 0.5
    expert_gamma: float = 0.1
    safe_mode_cooldown: int = 0
    
    # ADVANCED BEHAVIORAL MODELING - Post-result predictions
    post_draw_transitions: np.ndarray = field(default_factory=lambda: np.ones((3, 3), dtype=np.float32))  # (ai_draw_move, next_user_move)
    post_loss_transitions: np.ndarray = field(default_factory=lambda: np.ones((3, 3), dtype=np.float32))  # (ai_win_move, next_user_move) 
    post_win_transitions: np.ndarray = field(default_factory=lambda: np.ones((3, 3), dtype=np.float32))   # (ai_loss_move, next_user_move)
    contextual_transitions: np.ndarray = field(default_factory=lambda: np.ones((3, 3, 3), dtype=np.float32))  # (result_encoded, prev_user, next_user)
    
    # DEEP NEURAL PREDICTION SYSTEM
    sequence_memory: List[Tuple[int, int, str]] = field(default_factory=list)  # (ai_move, user_move, result) sequence
    user_psychology_profile: Dict[str, float] = field(default_factory=lambda: {
        "counter_tendency": 0.5,     # Tendency to counter AI moves
        "repeat_tendency": 0.5,      # Tendency to repeat own moves  
        "beat_tendency": 0.5,        # Tendency to beat AI moves
        "meta_awareness": 0.5,       # Awareness of AI patterns
        "emotional_reaction": 0.5,   # Changes behavior after loss/win
        "pattern_complexity": 0.5    # Uses complex vs simple patterns
    })
    neural_weights: Dict[str, np.ndarray] = field(default_factory=lambda: {
        "sequence_weights": np.random.normal(0, 0.1, (64, 3)),  # Sequence → prediction
        "psychology_weights": np.random.normal(0, 0.1, (6, 3)), # Psychology → prediction  
        "context_weights": np.random.normal(0, 0.1, (9, 3))     # Context → prediction
    })

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
            "expert_w": self.expert_w.tolist(),
            "expert_eta": self.expert_eta,
            "expert_gamma": self.expert_gamma,
            "safe_mode_cooldown": self.safe_mode_cooldown,
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
            # experts
            **({
                "expert_w": np.array(d.get("expert_w", (np.ones(len(EXPERT_REGISTRY)) / float(len(EXPERT_REGISTRY))).tolist()), dtype=np.float32),
                "expert_eta": float(d.get("expert_eta", 0.5)),
                "expert_gamma": float(d.get("expert_gamma", 0.1)),
                "safe_mode_cooldown": int(d.get("safe_mode_cooldown", 0)),
            }),
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
        # Global ai->user response model aggregated across all users
        self.global_ai_to_user_counts = np.ones((3, 3), dtype=np.float32)

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
                if "global_ai_to_user_counts" in state:
                    try:
                        self.global_ai_to_user_counts = np.array(state["global_ai_to_user_counts"], dtype=np.float32)
                    except Exception:
                        pass
                if "global_expert_w" in state:
                    try:
                        self.global_expert_w = np.array(state["global_expert_w"], dtype=np.float32)
                    except Exception:
                        pass
            # load per-user
            for uid, ud in self.storage.load_all_users().items():
                try:
                    self.users[uid] = UserState.from_dict(ud)
                except Exception:
                    continue
        # Initialize global expert prior if not loaded
        if not hasattr(self, "global_expert_w"):
            self.global_expert_w = np.ones(len(EXPERT_REGISTRY), dtype=np.float32) / float(len(EXPERT_REGISTRY))

        # ephemeral cache for last expert predictions per user
        self._last_expert_preds: Dict[str, List[np.ndarray]] = {}

    # ---------------------- Public API ----------------------
    def predict(self, user_hint: Optional[str], ctx: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        uid = self._resolve_user(user_hint, ctx)
        us = self._get_or_create_user(uid)
        
        # Soft Round-2 suggestion (no hard return): reduces exploitability in first 3 moves
        early_override_move: Optional[int] = None
        early_override_policy: Optional[str] = None
        if len(us.history) == 1:
            last = us.history[-1]
            last_ai = last.get("ai_move")
            last_u = last.get("u_move")
            res = last.get("result")
            if last_ai is not None and last_u is not None:
                repeat_counter = (last_u + 1) % 3
                defend_vs_counter = (last_ai + 2) % 3
                r = self.rng.random()
                if res in ["lose", "draw"]:
                    # Prefer to counter repeat but occasionally defend vs their counter
                    early_override_move = repeat_counter if r < 0.65 else defend_vs_counter
                    early_override_policy = "r2-soft-anti-repeat-mix" if r < 0.65 else "r2-soft-anti-counter"
                else:
                    # We won: defend vs their counter more often
                    early_override_move = defend_vs_counter if r < 0.6 else repeat_counter
                    early_override_policy = "r2-soft-defend-counter" if r < 0.6 else "r2-soft-defend-repeat"

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

        # Mid-session transfer: if early rounds look like a known player, warm start from them
        if us.seeded_from is None and 1 <= len(us.history) <= 4 and len(self.fp_index) > 0:
            try:
                fp = behavior_fingerprint(us.history)
                best_uid, best_sim = None, -1.0
                for kid, vec in self.fp_index.items():
                    sim = cosine_similarity(fp, vec)
                    if sim > best_sim:
                        best_uid, best_sim = kid, sim
                if best_uid is not None and best_sim >= 0.88 and best_uid in self.users:
                    src = self.users[best_uid]
                    # Stronger seed early based on similarity
                    wseed = float(max(0.2, min(0.9, (best_sim - 0.85) / (1.0 - 0.85))))
                    self._warm_start_from(us, src, wseed)
                    us.seeded_from, us.seed_weight = best_uid, wseed
                    soft_info = {"seeded_from": best_uid, "sim": best_sim, "w": wseed, "mode": "mid-session"}
            except Exception:
                pass

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

        # Robust mixed-response against meta exploitation (entropy-regularized BR)
        meta_exploit = self._meta_exploit_score(us)
        robust_move, robust_meta = self._robust_response(p_adj, us, meta_exploit)

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
        if 'early_override_move' in locals() and early_override_move is not None:
            choices.append(int(early_override_move))
            # Strong in round 2, then fades fast
            early_w = 0.8 if len(us.history) == 1 else 0.2
            weights.append(early_w)

        # Mixture-of-Experts: compute global+user hierarchical prediction
        try:
            names, expert_user_preds = compute_all_experts(
                history=us.history,
                ngram_counts_1=us.ngram_counts_1,
                ngram_counts_2=us.ngram_counts_2,
                last_move=us.last_move,
                first_move_counts=self.first_move_counts,
            )
            # Initialize/repair expert weights if dimension mismatch
            if not isinstance(us.expert_w, np.ndarray) or us.expert_w.shape[0] != len(names):
                # seed from global prior
                gw = getattr(self, "global_expert_w", None)
                if gw is None or len(gw) != len(names):
                    gw = np.ones(len(names), dtype=np.float32) / float(len(names))
                us.expert_w = np.asarray(gw, dtype=np.float32).copy()
            # Small exploration on the weights to avoid collapse
            nE = float(len(names))
            w_bar = (1.0 - float(us.expert_gamma)) * us.expert_w + float(us.expert_gamma) * (np.ones_like(us.expert_w) / nE)
            w_bar = w_bar / max(1e-8, float(np.sum(w_bar)))
            # Combine user-move distributions, then map to our best response
            p_user_star = np.zeros(3, dtype=np.float32)
            for wi, pu in zip(w_bar, expert_user_preds):
                p_user_star += float(wi) * np.asarray(pu, dtype=np.float32)
            s = float(np.sum(p_user_star)); p_user_star = p_user_star / s if s > 0 else (np.ones(3, dtype=np.float32) / 3.0)
            # Our action distribution induced by beating theirs; use as an expert candidate
            p_ai_star = ai_policy_from_user_dist(p_user_star)
            experts_move = int(np.argmax(p_ai_star))
            choices.append(experts_move)
            # Weight experts higher when we have little data or are in safe mode
            experts_weight = 0.25 + 0.35 * (1.0 - confidence)
            weights.append(experts_weight)
            # Cache predictions for online weight updates in feedback
            self._last_expert_preds[uid] = [np.asarray(p, dtype=np.float32) for p in expert_user_preds]
        except Exception:
            pass

        # Prefer robust move when exploitation likely
        if robust_move is not None:
            choices.append(int(robust_move))
            # Weight scales with exploitation strength and prediction confidence
            robust_w = 0.25 + 0.75 * float(meta_exploit)
            weights.append(robust_w)
        # ULTIMATE DOMINATION SYSTEM: INSTANT MAXIMUM RESPONSE
        anti_info = None
        anti_s, anti_move = self._user_counters_last_ai_prob(us)
        is_exploiting = self._exploitation_alert(us)
        # Detect copy-exploit tendency: user copies our last move; prepare anti-copy
        anti_copy_move: Optional[int] = None
        copy_suspect: float = 0.0
        if len(us.history) >= 1 and us.history[-1]["ai_move"] is not None:
            last_ai_mv = int(us.history[-1]["ai_move"])
            anti_copy_move = (last_ai_mv + 1) % 3  # beats a copy of our last move
            copy_suspect = self._copy_suspect_score(us)
        
        # Trigger on MINIMAL pattern detection OR exploitation alert
        if (anti_s is not None and anti_s >= 0.01) or is_exploiting:  # 1% threshold!
            if anti_move is not None:
                choices.append(anti_move)
            
            # MAXIMUM DOMINATION WEIGHTING SYSTEM
            base_weight = 5.0 if anti_s is not None and anti_s >= 0.01 else 3.0
            # Scale 5.0..20.0 as anti_s goes 0.01..1.0 (extreme aggression)
            if anti_s is not None:
                weight = base_weight + 15.0 * min(1.0, (anti_s - 0.01) / 0.99)
            else:
                weight = base_weight
                
            # EXTREME multipliers for guaranteed dominance
            if is_exploiting:
                weight *= 10.0  # 10x multiplier for exploitation
            if len(us.history) <= 10:  # Extended early learning
                weight *= 5.0  # 5x early game multiplier
            if len(us.history) >= 2 and us.history[-1]["result"] == "lose":
                weight *= 4.0  # 4x after immediate loss
            if len(us.history) >= 2 and us.history[-1]["result"] == "draw":
                weight *= 3.0  # 3x after draw (should never draw)
            
            weights.append(weight)
            anti_info = {"p_counter": anti_s, "anti_move": int(anti_move) if anti_move is not None else None}
            
            # ANTI-PREDICTABILITY CHECK - Don't let user counter-predict AI
            current_win_rate = self._calculate_recent_win_rate(us)
            if current_win_rate < 0.65 and len(us.history) >= 4:
                # AI is being counter-predicted - add chaos variants
                chaos_variants = []
                
                # Add 3 unpredictable moves
                for _ in range(3):
                    random_move = self.rng.choice([0, 1, 2])
                    chaos_variants.append(random_move)
                    weights.append(weight * 0.6)  # Medium weight for chaos
                choices.extend(chaos_variants)
                
                # Add counter-intuitive moves
                if len(us.history) >= 1:
                    last_user = us.history[-1]["u_move"]
                    # Instead of countering user, play what they played (psychological effect)
                    mirror_move = last_user  
                    choices.append(mirror_move)
                    weights.append(weight * 0.8)
                    
                    # Play what would lose to user's last move (reverse psychology)
                    lose_move = (last_user + 2) % 3
                    choices.append(lose_move) 
                    weights.append(weight * 0.5)

            # MAXIMUM VARIANT SYSTEM - Add 5 different counter-strategies  
            variant_threshold = 0.08 if is_exploiting else 0.12  # Even lower threshold
            if anti_s is not None and anti_s >= variant_threshold:                # Variant 1: Triple counter (3rd level thinking)
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
            # MAXIMUM boost for meta moves - ALWAYS PRIORITIZE WINNING
            base_meta_weight = 0.3 * (1 - confidence) + (0.1 if len(us.history) >= 2 else 0.4)
            if is_exploiting or (anti_s is not None and anti_s >= 0.01):  # Lower threshold
                # EXTREME priority for meta rules when ANY pattern detected
                if len(us.history) <= 6:  # First 6 rounds get maximum priority
                    base_meta_weight *= 15.0  # 15x boost for early meta
                else:
                    base_meta_weight *= 10.0  # 10x boost for later meta
            else:
                # Even without exploitation, boost meta learning
                base_meta_weight *= 3.0  # 3x default boost
            weights.append(base_meta_weight)
        # AI-conditioned user model (how user responds to our move) — gated to avoid over-reliance
        ai_cond = self._ai_conditioned_move(us)
        if ai_cond is not None and not is_exploiting:
            ai_move2, conf2 = ai_cond
            # Only use when confident and not under exploitation
            if conf2 >= 0.6:
                choices.append(int(ai_move2))
                weights.append(0.15 + 0.35 * conf2)
        # Style-based choice (if confident)
        if style_move is not None:
            choices.append(style_move)
            weights.append(0.2 + 0.3 * min(1.0, style_conf))
        # Dedicated 2-move alternating pattern candidate
        alt_conf, alt_expected = self._detect_two_move_alternating(us)
        if alt_expected is not None:
            alt_counter = (int(alt_expected) + 1) % 3
            choices.append(alt_counter)
            # Strong weight early; scale with confidence and early rounds
            early_boost = 1.8 if len(us.history) <= 8 else 1.0
            weights.append(0.6 * early_boost + 2.0 * alt_conf * early_boost)
        # Anti-copy candidate (more aggressive and earlier)
        if anti_copy_move is not None and copy_suspect > 0.08:
            choices.append(int(anti_copy_move))
            # Stronger base weight and scale with suspicion
            weights.append(0.2 + 0.9 * copy_suspect + (0.3 if is_exploiting else 0.0))
        # KNN style-based choice for cold start
        if knn_move is not None and knn_info is not None and knn_info.get("best_sim", 0.0) >= 0.8:
            choices.append(int(knn_move))
            weights.append(0.15 + 0.25 * (1.0 - confidence))

        # HARD-LOCK: If user is spamming a single move, counter deterministically
        hard_lock_repeat = False
        repeated_user_move: Optional[int] = None
        if len(us.history) >= 3:
            last_k = [h["u_move"] for h in us.history[-4:]]
            if len(last_k) >= 3 and len(set(last_k)) == 1:
                repeated_user_move = int(last_k[-1])
                hard_lock_repeat = True

        if hard_lock_repeat and repeated_user_move is not None:
            ai_move = int((repeated_user_move + 1) % 3)
            policy = "hard-lock-repeat"
            meta = {
                "uid": uid,
                "p_opp": p_ngram.tolist(),
                "br_move": int(br_move),
                "bandit": bandit_meta,
                "lookahead": look_scores.tolist(),
                "fused_move": int(ai_move),
                "eps": 0.0,
                "policy": policy,
                "soft_seed": soft_info,
                "anti_exploit": anti_info,
                "first_counts": self.first_move_counts.tolist(),
                "style_probs": style_probs.tolist() if style_probs is not None else None,
                "knn": knn_info,
                "family": "anti",
            }
            return int(ai_move), meta

        # TOTAL ANNIHILATION when exploitation detected - ELIMINATE ALL predictable strategies
        if is_exploiting or (anti_s is not None and anti_s >= 0.01):  # Lower threshold
            # OBLITERATE basic strategies completely
            if len(weights) >= 1:
                weights[0] *= 0.001  # Virtually eliminate base_choice (BR) - 0.1%
            if len(weights) >= 2:
                weights[1] *= 0.1   # Severely reduce lookahead
            
            # MASSIVELY boost ALL anti-exploitation strategies
            for i in range(2, len(weights)):
                if i < len(choices):
                    # Check if this is one of our anti-exploitation moves
                    current_move = choices[i]
                    if (anti_move is not None and current_move == anti_move) or i >= len(choices) - 10:
                        # This is likely an anti-exploitation strategy
                        weights[i] *= 20.0  # 20x boost for anti-strategies
                    else:
                        # Other strategic moves get major boost
                        weights[i] *= 8.0
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
        # SAFE MODE: if rolling win-rate dips below baseline, raise randomness for a few rounds
        if len(us.history) >= 6:
            recent = us.history[-min(30, len(us.history)):]
            wins_recent = sum(1 for h in recent if h["result"] == "win")
            wr_recent = wins_recent / max(1, len(recent))
            if wr_recent < 0.33:
                us.safe_mode_cooldown = max(us.safe_mode_cooldown, 8)
        safe_mode_active = us.safe_mode_cooldown > 0
        if safe_mode_active:
            eps = max(eps, 0.33)
        
        # MAXIMUM randomness when being exploited for unpredictability — but NOT when we have a strong pattern
        strong_pattern = (anti_info is not None and float(anti_info.get("p_counter", 0.0)) >= 0.5)
        if not strong_pattern:
            if is_exploiting or (anti_s is not None and anti_s >= 0.10):
                eps = max(eps, 0.6)  # Minimum 60% randomness when exploited
                # Extra randomness in early rounds for faster learning
                if len(us.history) <= 5:
                    eps = max(eps, 0.8)  # 80% randomness in early exploitation
            
        if self._is_adversarial(us) or is_exploiting or safe_mode_active:
            # ABSOLUTE OVERRIDE: If we have ANY pattern detection, ALWAYS use it
            if anti_info is not None and anti_move is not None:
                pattern_confidence = anti_info.get("p_counter", 0.0)
                if pattern_confidence >= 0.5:  # 50%+ confidence - FORCE use pattern counter
                    ai_move = anti_move
                    policy = "pattern-override-high-conf"
                elif pattern_confidence >= 0.2:  # 20%+ confidence - almost always use pattern counter
                    if self.rng.random() < 0.95:  # 95% chance to use pattern
                        ai_move = anti_move
                        policy = "pattern-override-med-conf"
                    else:
                        ai_move = self.rng.randrange(3)
                        policy = "anti-track-mix"
                elif pattern_confidence >= 0.01:  # Even tiny confidence - prefer pattern
                    if self.rng.random() < 0.8:  # 80% chance to use pattern
                        ai_move = anti_move
                        policy = "pattern-override-low-conf"
                    else:
                        ai_move = self.rng.randrange(3)
                        policy = "anti-track-mix"
                else:
                    # Very low confidence - still favor pattern over random
                    if self.rng.random() < 0.6:
                        ai_move = anti_move if anti_move is not None else self.rng.randrange(3)
                        policy = "anti-track-counter"
                    else:
                        ai_move = self.rng.randrange(3)
                        policy = "anti-track-mix"
            else:
                # No pattern detected - minimal randomness
                if self.rng.random() < (0.6 if safe_mode_active else 0.7):  # Prefer BR over random
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
            if robust_move is not None:
                family_candidates["robust"] = int(robust_move)
            if anti_copy_move is not None and copy_suspect > 0.15:
                family_candidates["anti_copy"] = int(anti_copy_move)
            # Include experts family if available
            if uid in self._last_expert_preds:
                family_candidates["experts"] = int(experts_move)

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
                    elif f in ["anti", "ai_cond", "robust", "anti_copy"]:  # Massive boost for counter-strategies
                        # MAXIMUM boost for any pattern confidence
                        pattern_boost = 8.0  # Base 8x boost
                        if anti_info is not None:
                            pattern_confidence = anti_info.get("p_counter", 0.0)
                            if pattern_confidence >= 0.5:
                                pattern_boost = 15.0  # 15x boost for 50%+ confidence
                            elif pattern_confidence >= 0.2:
                                pattern_boost = 12.0  # 12x boost for 20%+ confidence
                            elif pattern_confidence >= 0.01:
                                pattern_boost = 10.0  # 10x boost for any confidence
                        # Also scale with meta exploitation intensity
                        ucb *= pattern_boost * (1.0 + 1.5 * float(meta_exploit))
                        
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

        # FINAL ANTI-PREDICTABILITY LAYER - Prevent AI from being too predictable
        # Skip if we have a strong detected pattern (e.g., user spamming one move)
        recent_win_rate = self._calculate_recent_win_rate(us)
        if (anti_info is None or float(anti_info.get("p_counter", 0.0)) < 0.5) and (not hard_lock_repeat):
            if recent_win_rate < 0.7 and len(us.history) >= 4:
                # Check if AI has been too predictable
                recent_ai_moves = [h["ai_move"] for h in us.history[-4:] if h["ai_move"] is not None]
                
                if len(recent_ai_moves) >= 3:
                    # If AI is showing patterns, add randomness
                    move_entropy = len(set(recent_ai_moves)) / len(recent_ai_moves)
                    
                    if move_entropy < 0.7:  # Low entropy = predictable
                        # Force unpredictable choice 30% of the time
                        if self.rng.random() < 0.3:
                            # Avoid recently used moves
                            unused_moves = [m for m in [0, 1, 2] if m not in recent_ai_moves[-2:]]
                            if unused_moves:
                                ai_move = int(self.rng.choice(unused_moves))
                                policy = "anti-predict-chaos"
                            else:
                                # If all moves used, pick least recent
                                move_counts = {m: recent_ai_moves.count(m) for m in [0, 1, 2]}
                                ai_move = min(move_counts.keys(), key=lambda k: move_counts[k])
                                policy = "anti-predict-balance"
        # Immediate anti-copy guard: if user just copied our last move, preemptively counter
        try:
            if len(us.history) >= 2 and us.history[-1]["ai_move"] is not None:
                prev_ai = int(us.history[-1]["ai_move"])  # AI move last round
                if us.history[-1]["u_move"] == prev_ai:
                    # They copied last round; choose the move that beats a copy again
                    preempt = (prev_ai + 1) % 3
                    ai_move = preempt
                    policy = "anti-copy-instant"
        except Exception:
            pass

        # FINAL ALT-2 GUARD: if a strong two-move alternation is detected, hard-counter it
        try:
            alt_conf2, alt_expected2 = self._detect_two_move_alternating(us)
            if alt_expected2 is not None and alt_conf2 >= 0.25:  # low threshold for fast lock
                alt_counter2 = (int(alt_expected2) + 1) % 3
                if ai_move != alt_counter2:
                    ai_move = alt_counter2
                    policy = "alt2-guard"
        except Exception:
            pass

        # NO-DRAW GUARD: if user likely to repeat their last move, avoid mirroring
        # This reduces long draw streaks against one-move or sticky players.
        if len(us.history) >= 1:
            try:
                rp = self._repeat_probability(us)
                last_user_move = us.history[-1]["u_move"]
                if rp >= 0.6:
                    # If our current choice would draw, switch to the counter
                    if ai_move == last_user_move:
                        ai_move = (last_user_move + 1) % 3
                        policy = "no-draw-guard-counter-repeat"
                    # If we've drawn multiple times recently, proactively counter
                    recent = us.history[-4:] if len(us.history) >= 4 else us.history
                    draw_streak = sum(1 for h in recent if h["result"] == "draw")
                    if draw_streak >= 2 and self.rng.random() < 0.8:
                        ai_move = (last_user_move + 1) % 3
                        policy = "no-draw-guard-break-streak"
                # Anti-copy final guard: if user likely copies our last move, avoid vulnerable move
                if us.history[-1]["ai_move"] is not None:
                    last_ai_mv = int(us.history[-1]["ai_move"])
                    copy_sus = self._copy_suspect_score(us)
                    vulnerable_move = (last_ai_mv + 2) % 3  # loses to copy(last_ai)
                    if copy_sus >= 0.35 and ai_move == vulnerable_move:
                        if self.rng.random() < (0.6 + 0.3 * copy_sus):
                            ai_move = (last_ai_mv + 1) % 3
                            policy = "anti-copy-guard"
            except Exception:
                pass

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
            "experts": {
                "names": names if 'names' in locals() else None,
                "w": us.expert_w.tolist() if isinstance(us.expert_w, np.ndarray) else None,
            },
            "safe_mode": bool(safe_mode_active),
            "win_rate_recent": float(self._calculate_recent_win_rate(us)),
        }
        # decrement safe-mode cooldown after deciding
        if us.safe_mode_cooldown > 0:
            us.safe_mode_cooldown -= 1
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
            
        # UPDATE ADVANCED BEHAVIORAL TRANSITIONS
        # If this is not the first round, update post-result behavior
        if len(us.history) >= 1:  # Have previous round data
            prev_result = us.history[-1]["result"]
            prev_ai_move = us.history[-1]["ai_move"]
            
            # Update transition matrices based on previous result
            if prev_result == "draw":
                # User's move after a draw
                us.post_draw_transitions[prev_ai_move, user_move] += 2.0  # Strong learning
            elif prev_result == "win":  # AI won, user lost
                # User's move after losing
                us.post_loss_transitions[prev_ai_move, user_move] += 2.5  # Extra strong learning
            elif prev_result == "lose":  # AI lost, user won
                # User's move after winning  
                us.post_win_transitions[prev_ai_move, user_move] += 1.8
                
            # Update contextual transitions: (prev_result, prev_user_move) -> current_user_move
            prev_user_move = us.history[-1]["u_move"]
            result_encoding = {"win": 0, "draw": 1, "lose": 2}  # AI perspective
            result_idx = result_encoding.get(prev_result, 1)
            us.contextual_transitions[result_idx, prev_user_move, user_move] += 2.0
            
            # UPDATE PSYCHOLOGY PROFILE - Learn user tendencies
            prev_ai = us.history[-1]["ai_move"]
            if prev_ai is not None:
                # Counter tendency
                if user_move == (prev_ai + 1) % 3:
                    us.user_psychology_profile["counter_tendency"] += 0.1
                else:
                    us.user_psychology_profile["counter_tendency"] -= 0.05
                    
                # Beat tendency  
                if user_move == (prev_ai + 2) % 3:
                    us.user_psychology_profile["beat_tendency"] += 0.1
                else:
                    us.user_psychology_profile["beat_tendency"] -= 0.05
                    
                # Repeat tendency
                if user_move == prev_user_move:
                    us.user_psychology_profile["repeat_tendency"] += 0.1
                else:
                    us.user_psychology_profile["repeat_tendency"] -= 0.05
                    
                # Emotional reaction - changes after loss
                if prev_result == "win" and user_move != prev_user_move:  # AI won, user changed
                    us.user_psychology_profile["emotional_reaction"] += 0.15
                elif prev_result == "win" and user_move == prev_user_move:  # AI won, user didn't change
                    us.user_psychology_profile["emotional_reaction"] -= 0.05
                    
                # Meta awareness - if user breaks their own patterns
                if len(us.history) >= 3:
                    # Check if user broke their established pattern
                    prev_prev_user = us.history[-2]["u_move"] if len(us.history) >= 2 else None
                    if prev_prev_user is not None:
                        expected_pattern = prev_prev_user  # Simple pattern expectation
                        if user_move != expected_pattern and user_move != prev_user_move:
                            us.user_psychology_profile["meta_awareness"] += 0.12
                        else:
                            us.user_psychology_profile["meta_awareness"] -= 0.03
                            
                # Pattern complexity - varies move choice
                recent_moves = [h["u_move"] for h in us.history[-5:]]
                if len(set(recent_moves)) >= 3:  # Uses all 3 moves recently
                    us.user_psychology_profile["pattern_complexity"] += 0.08
                elif len(set(recent_moves)) == 1:  # Very predictable
                    us.user_psychology_profile["pattern_complexity"] -= 0.1
            
            # Clamp psychology values to [0, 1]
            for key in us.user_psychology_profile:
                us.user_psychology_profile[key] = max(0.0, min(1.0, us.user_psychology_profile[key]))
                
        # UPDATE SEQUENCE MEMORY for neural learning
        us.sequence_memory.append((ai_move, user_move, result))
        if len(us.sequence_memory) > 50:  # Keep last 50 moves
            us.sequence_memory.pop(0)
            
        # NEURAL WEIGHTS UPDATE - Simple gradient update
        if len(us.history) >= 2:
            # Update neural weights based on prediction accuracy
            features = self._build_neural_features(us)
            if features is not None:
                target = np.zeros(3)
                target[user_move] = 1.0
                
                # Simple gradient update for each weight matrix
                for weight_name in us.neural_weights:
                    if weight_name == "sequence_weights":
                        feature_vec = features[:64]  # First 64 features
                    elif weight_name == "psychology_weights":
                        feature_vec = features[64:70]  # Next 6 features
                    else:  # context_weights
                        feature_vec = features[70:79]  # Next 9 features
                    
                    if len(feature_vec) > 0 and len(feature_vec) == us.neural_weights[weight_name].shape[0]:
                        pred = self._softmax(us.neural_weights[weight_name].T @ feature_vec)
                        error = target - pred
                        
                        # Gradient update
                        lr = 0.01  # Learning rate
                        us.neural_weights[weight_name] += lr * np.outer(feature_vec, error)
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

        # Online update of experts (EXP4.P/Hedge-like)
        try:
            if us.user_id in self._last_expert_preds:
                preds = self._last_expert_preds.pop(us.user_id)
                nE = len(preds)
                if isinstance(us.expert_w, np.ndarray) and us.expert_w.shape[0] == nE:
                    # Use log-loss for smoother updates
                    losses = np.zeros(nE, dtype=np.float32)
                    for i, pu in enumerate(preds):
                        pu = pu.astype(np.float32)
                        pu = pu / max(1e-8, float(np.sum(pu)))
                        losses[i] = -math.log(float(pu[int(user_move)]) + 1e-8)
                    # Hedge update
                    eta = float(us.expert_eta)
                    new_w = us.expert_w * np.exp(-eta * losses)
                    # Exploration mix
                    gamma = float(us.expert_gamma)
                    new_w = (1.0 - gamma) * new_w + gamma * (np.ones_like(new_w) / float(nE))
                    s = float(np.sum(new_w));
                    if s > 0:
                        new_w = new_w / s
                        us.expert_w = new_w.astype(np.float32)
                        # Lightly update global prior
                        if hasattr(self, "global_expert_w") and len(self.global_expert_w) == nE:
                            self.global_expert_w = (0.98 * self.global_expert_w + 0.02 * new_w).astype(np.float32)
        except Exception:
            pass

        # Save incrementally if allowed
        if self.remember_history:
            self.storage.save_user(uid, us.to_dict())
            self.storage.save_global({
                "W": self.W,
                "b": self.b,
                "fp_index": self.fp_index,
                "first_move_counts": self.first_move_counts,
                "global_ai_to_user_counts": self.global_ai_to_user_counts,
                "global_expert_w": getattr(self, "global_expert_w", None),
            })

    # --------- AI-conditioned user model ---------
    def _ai_conditioned_move(self, us: UserState) -> Optional[Tuple[int, float]]:
        """Pick a counter move based on how this user typically responds to OUR move.
        Now gated by support and entropy to avoid over-reliance and easy exploitation.
        """
        if len(us.history) < 6:
            # Cold-start: try global model if sufficiently informative
            M = getattr(self, "global_ai_to_user_counts", None)
            if M is None:
                return None
            try:
                M = np.asarray(M, dtype=np.float32)
            except Exception:
                return None
            best_move = None
            best_conf = 0.0
            for potential_ai_move in range(3):
                row = M[potential_ai_move]
                support = float(np.sum(row))
                if support < 50.0:
                    continue
                p_user = row / support
                eps = 1e-8
                ent = -float(np.sum(p_user * np.log(p_user + eps))) / float(np.log(3))
                conf = float(np.max(p_user))
                if conf < 0.55 or ent > 0.95:
                    continue
                likely_user_move = int(np.argmax(p_user))
                our_counter = (likely_user_move + 1) % 3
                if conf > best_conf:
                    best_move, best_conf = our_counter, conf
            if best_move is not None:
                return int(best_move), float(best_conf)
            return None

        # If we're being exploited, don't rely on this model
        try:
            if self._exploitation_alert(us):
                return None
        except Exception:
            pass

        M = us.ai_to_user_counts.astype(np.float32)
        best_move: Optional[int] = None
        best_conf: float = 0.0

        for potential_ai_move in range(3):
            row = M[potential_ai_move]
            support = float(np.sum(row))
            if support < 8.0:  # require enough data
                continue

            p_user = row / support
            # row entropy (normalized) — low entropy => strong tendency
            eps = 1e-8
            ent = -float(np.sum(p_user * np.log(p_user + eps))) / float(np.log(3))
            conf = float(np.max(p_user))

            # Require strong signal and reasonably low entropy
            if conf < 0.6 or ent > 0.9:
                continue

            likely_user_move = int(np.argmax(p_user))
            our_counter = (likely_user_move + 1) % 3

            if conf > best_conf:
                best_move = our_counter
                best_conf = conf

        if best_move is not None:
            return int(best_move), float(best_conf)
        return None

    

    def _update_ai_cond_counts(self, us: UserState, ai_move: int, user_move: int) -> None:
        # Update counts of how user responded to our move in previous round
        if len(us.history) < 2:
            return
        prev_ai = us.history[-2]["ai_move"]
        if prev_ai is None:
            return
        us.ai_to_user_counts[int(prev_ai), int(user_move)] += 1.0
        # Update global model as well
        if hasattr(self, "global_ai_to_user_counts"):
            self.global_ai_to_user_counts[int(prev_ai), int(user_move)] += 1.0

    def save(self) -> None:
        if not self.remember_history:
            return
        self.storage.save_global({
            "W": self.W,
            "b": self.b,
            "fp_index": self.fp_index,
            "first_move_counts": self.first_move_counts,
            "global_ai_to_user_counts": self.global_ai_to_user_counts,
            "global_expert_w": getattr(self, "global_expert_w", None),
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

    def _meta_exploit_score(self, us: UserState) -> float:
        """Compute how likely the user is actively exploiting our policy [0..1]."""
        score = 0.0
        # Immediate signal from alert
        try:
            if self._exploitation_alert(us):
                score += 0.5
        except Exception:
            pass
        # Recent win rate deficit
        try:
            wr = self._calculate_recent_win_rate(us)
        except Exception:
            wr = 0.5
        score += max(0.0, 0.75 - float(wr)) * 1.2  # stronger push when we're under 75%
        # Logloss pressure
        if hasattr(us, "ema_logloss"):
            score += max(0.0, float(us.ema_logloss) - 1.0) * 0.4
        # Adversarial behavior tracker
        if hasattr(us, "adversarial_score"):
            score += 0.5 * float(us.adversarial_score)
        # Detected anti pattern confidence
        try:
            anti_s, _ = self._user_counters_last_ai_prob(us)
            if anti_s is not None:
                score += min(0.6, float(anti_s) * 0.6)
        except Exception:
            pass
        # Clamp to [0,1]
        return float(max(0.0, min(1.0, score)))

    def _robust_response(self, p_adj: np.ndarray, us: UserState, meta_exploit: float) -> Tuple[Optional[int], Dict[str, Any]]:
        """Hedge our response against exploitation by smoothing the opponent model and adding entropy.
        Returns (move, meta). Move may be None if inputs are invalid.
        """
        try:
            p = np.array(p_adj, dtype=np.float32).copy()
        except Exception:
            return None, {"ok": False, "reason": "bad_p"}
        s = float(np.sum(p))
        if s <= 0 or not np.isfinite(s):
            p = np.ones(3, dtype=np.float32) / 3.0
        else:
            p /= s
        # Hedge with uniform to avoid being overfit to adversary shaping
        rho = float(min(0.75, max(0.2, 0.25 + 0.5 * float(meta_exploit))))
        p_hedge = (1.0 - rho) * p + rho * (np.ones(3, dtype=np.float32) / 3.0)
        # Payoff matrix A[m, u] where rows are our moves, columns user moves
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float32)
        ev = A @ p_hedge  # expected payoff per our move
        # Temperature for entropy-regularization increases when exploitation rises
        temp = float(1.0 + 1.5 * max(0.0, min(1.0, meta_exploit)))
        # Softmax over expected values to produce mixed robust strategy
        logits = ev / max(1e-6, temp)
        probs = np.exp(logits - np.max(logits)); probs /= float(np.sum(probs))
        # Choose: sample under high exploitation to be less predictable
        move = int(np.argmax(ev))
        sample_prob = 0.15 + 0.55 * float(meta_exploit)  # 15%..70%
        if self.rng.random() < sample_prob:
            move = int(np.random.choice([0, 1, 2], p=probs))
        meta = {
            "rho": rho,
            "p_hedge": p_hedge.tolist(),
            "ev": ev.tolist(),
            "temp": temp,
            "probs": probs.tolist(),
            "meta_exploit": float(meta_exploit),
        }
        return move, meta

    # ---------------------- New Logic: meta, lookahead, transfer, adversarial ----------------------
    def _early_meta(self, us: UserState) -> Tuple[Optional[int], str]:
        n = len(us.history)
        if n == 0:
            # Round 1: mixed opening derived from population prior (minimax-safe)
            # Build a user-first distribution then map to our beating distribution
            p_first = self.first_move_counts.astype(np.float32)
            p_first = p_first / max(1e-6, float(np.sum(p_first)))
            p_ai_from_prior = np.zeros(3, dtype=np.float32)
            for j in range(3):
                p_ai_from_prior[(j + 1) % 3] += float(p_first[j])
            p_ai_from_prior = p_ai_from_prior / max(1e-6, float(np.sum(p_ai_from_prior)))
            # Entropy-mix with uniform to avoid opening traps
            mixed = 0.7 * p_ai_from_prior + 0.3 * (np.ones(3, dtype=np.float32) / 3.0)
            mixed = mixed / max(1e-6, float(np.sum(mixed)))
            move = int(np.random.choice([0, 1, 2], p=mixed))
            return int(move), "meta-r1-mixed-opening"
            
        if n == 1:
            last = us.history[-1]
            last_ai, last_u, res = last["ai_move"], last["u_move"], last["result"]
            
            # Round 2: probabilistic defense to avoid being baited
            counter_repeat = (last_u + 1) % 3
            counter_their_counter = (last_ai + 2) % 3 if last_ai is not None else counter_repeat
            rnd = self.rng.random()
            if res in ["lose", "draw"]:
                # 60% counter repeat, 35% counter their counter, 5% random
                if rnd < 0.60:
                    return counter_repeat, "meta-r2-anti-repeat-mix"
                elif rnd < 0.95:
                    return counter_their_counter, "meta-r2-anti-counter-mix"
                else:
                    return self.rng.randrange(3), "meta-r2-explore"
            else:  # we won
                # 55% defend vs their counter, 35% still counter repeat, 10% random
                if rnd < 0.55:
                    return counter_their_counter, "meta-r2-defend-counter-mix"
                elif rnd < 0.90:
                    return counter_repeat, "meta-r2-defend-repeat-mix"
                else:
                    return self.rng.randrange(3), "meta-r2-explore"
                
        if n == 2:
            # Round 3 - Look for immediate patterns
            u_last2 = [us.history[-2]["u_move"], us.history[-1]["u_move"]]
            results = [us.history[-2]["result"], us.history[-1]["result"]]
            ai_last = us.history[-1]["ai_move"]
            # Escalate if we lost both early rounds
            if results[0] == "lose" and results[1] in ["lose", "draw"]:
                # They might be countering our adaptations; go double-level more often
                defend_vs_counter = (ai_last + 2) % 3 if ai_last is not None else (u_last2[1] + 1) % 3
                counter_repeat = (u_last2[1] + 1) % 3
                rnd = self.rng.random()
                if rnd < 0.65:
                    return defend_vs_counter, "meta-r3-escalate-anti-counter"
                elif rnd < 0.90:
                    return counter_repeat, "meta-r3-escalate-repeat"
                else:
                    return self.rng.randrange(3), "meta-r3-explore"
            
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
        
        # PRIORITY 0: MIXED STRATEGY - Avoid being predictable
        win_rate = self._calculate_recent_win_rate(us)
        
        # If AI is losing too much, activate chaos mode
        if win_rate < 0.6 and len(us.history) >= 5:
            # CHAOS MODE - Break patterns to avoid counter-prediction
            chaos_confidence = 0.85
            chaos_moves = list(range(3))
            
            # Remove obviously bad moves but stay unpredictable
            if len(us.history) >= 1:
                last_user = us.history[-1]["u_move"]
                # Don't play what user just beat
                losing_move = (last_user + 2) % 3
                if losing_move in chaos_moves:
                    chaos_moves.remove(losing_move)
            
            # Pick random from remaining good moves
            chaos_move = self.rng.choice(chaos_moves) if chaos_moves else self.rng.choice([0, 1, 2])
            return chaos_confidence, chaos_move
            
        # PRIORITY 1: NEURAL DEEP LEARNING PREDICTION with randomization
        neural_confidence, neural_move = self._neural_prediction(us)
        if neural_confidence is not None and neural_move is not None and neural_confidence >= 0.5:
            # Don't always use the obvious counter - mix strategies
            counter_move = (neural_move + 1) % 3
            
            # META-LEVEL THINKING: Sometimes user expects us to counter
            # 80% time: Use direct counter
            # 15% time: Use counter-counter (assume they predict our counter)
            # 5% time: Use beat (completely different)
            
            strategy_roll = self.rng.random()
            if strategy_roll < 0.8:
                # Direct counter (most of the time)
                final_move = counter_move
                confidence_multiplier = 1.8
            elif strategy_roll < 0.95:
                # Counter-counter (they expect us to counter, so we counter their counter)
                expected_user_counter = (counter_move + 1) % 3  # They might counter our counter
                final_move = (expected_user_counter + 1) % 3   # We counter their counter
                confidence_multiplier = 1.6
            else:
                # Beat strategy (unexpected move)
                final_move = (neural_move + 2) % 3  # Beat their predicted move directly
                confidence_multiplier = 1.4
                
            ultra_confidence = min(0.94, neural_confidence * confidence_multiplier)
            return ultra_confidence, final_move

        # ULTIMATE BEHAVIORAL PREDICTION - Predict based on previous result context
        if len(us.history) >= 1:
            last_result = us.history[-1]["result"] 
            last_user_move = us.history[-1]["u_move"]
            last_ai_move = us.history[-1]["ai_move"]
            
            # POST-DRAW BEHAVIOR PREDICTION using learned Markov transitions
            if last_result == "draw":
                # Use learned post-draw transition matrix
                draw_probs = us.post_draw_transitions[last_ai_move, :] / us.post_draw_transitions[last_ai_move, :].sum()
                
                # Get most likely user move after draw
                expected_user = int(np.argmax(draw_probs))
                confidence = float(draw_probs[expected_user])
                
                # If we have strong learned pattern (>50% probability)
                if confidence >= 0.35:  # Lower threshold for faster learning
                    counter_move = (expected_user + 1) % 3
                    # Scale confidence to be more aggressive
                    boosted_confidence = min(0.92, confidence * 2.2)
                    return boosted_confidence, counter_move
                    
                # Fallback to historical pattern analysis if Markov isn't strong enough
                draw_follow_pattern = {"repeat": 0.0, "counter": 0.0, "beat": 0.0, "random": 0.0}
                
                for i in range(1, len(us.history)):
                    if us.history[i-1]["result"] == "draw":
                        prev_user = us.history[i-1]["u_move"]  
                        prev_ai = us.history[i-1]["ai_move"]
                        next_user = us.history[i]["u_move"]
                        
                        if next_user == prev_user:  # Repeated same move
                            draw_follow_pattern["repeat"] += 2.0
                        elif next_user == (prev_ai + 1) % 3:  # Counter AI's move
                            draw_follow_pattern["counter"] += 2.0  
                        elif next_user == (prev_ai + 2) % 3:  # Beat AI's move
                            draw_follow_pattern["beat"] += 2.0
                        else:
                            draw_follow_pattern["random"] += 1.0
                
                # If we have strong pattern, predict with high confidence
                max_pattern = max(draw_follow_pattern.values())
                if max_pattern >= 1.5:  # Lower threshold
                    pattern_type = max(draw_follow_pattern.keys(), key=lambda k: draw_follow_pattern[k])
                    
                    if pattern_type == "repeat":
                        expected_user = last_user_move
                        confidence = 0.88
                    elif pattern_type == "counter":
                        expected_user = (last_ai_move + 1) % 3
                        confidence = 0.92
                    elif pattern_type == "beat":
                        expected_user = (last_ai_move + 2) % 3  
                        confidence = 0.90
                    else:
                        expected_user = last_user_move  # Default fallback
                        confidence = 0.7
                    
                    counter_move = (expected_user + 1) % 3
                    return confidence, counter_move
            
            # POST-LOSS BEHAVIOR PREDICTION using learned Markov transitions (when AI won, user lost)
            elif last_result == "win":  # AI won, user lost
                # Use learned post-loss transition matrix
                loss_probs = us.post_loss_transitions[last_ai_move, :] / us.post_loss_transitions[last_ai_move, :].sum()
                
                # Get most likely user move after loss
                expected_user = int(np.argmax(loss_probs))
                confidence = float(loss_probs[expected_user])
                
                # If we have strong learned pattern
                if confidence >= 0.35:  # Lower threshold for faster learning
                    counter_move = (expected_user + 1) % 3
                    # Scale confidence - users are very predictable after losses
                    boosted_confidence = min(0.95, confidence * 2.5)  # Higher boost for post-loss
                    return boosted_confidence, counter_move
                    
                # Fallback to historical pattern analysis
                loss_follow_pattern = {"repeat": 0.0, "counter": 0.0, "beat": 0.0, "switch": 0.0}
                
                for i in range(1, len(us.history)):
                    if us.history[i-1]["result"] == "win":  # AI won previous round
                        prev_user = us.history[i-1]["u_move"]
                        prev_ai = us.history[i-1]["ai_move"] 
                        next_user = us.history[i]["u_move"]
                        
                        if next_user == prev_user:  # Repeated losing move
                            loss_follow_pattern["repeat"] += 1.5
                        elif next_user == (prev_ai + 1) % 3:  # Counter AI's winning move
                            loss_follow_pattern["counter"] += 2.5
                        elif next_user == (prev_ai + 2) % 3:  # Beat AI's winning move
                            loss_follow_pattern["beat"] += 2.0
                        else:
                            loss_follow_pattern["switch"] += 1.0
                
                # Predict with high confidence
                max_pattern = max(loss_follow_pattern.values())
                if max_pattern >= 1.2:  # Lower threshold
                    pattern_type = max(loss_follow_pattern.keys(), key=lambda k: loss_follow_pattern[k])
                    
                    if pattern_type == "repeat":
                        expected_user = last_user_move
                        confidence = 0.82
                    elif pattern_type == "counter":
                        expected_user = (last_ai_move + 1) % 3
                        confidence = 0.94  # Very high - users often counter after loss
                    elif pattern_type == "beat":
                        expected_user = (last_ai_move + 2) % 3
                        confidence = 0.89
                    else:
                        # They switch randomly - use most common non-losing move
                        non_losing = [(last_ai_move + 1) % 3, (last_ai_move + 2) % 3]
                        expected_user = non_losing[0]  # Default to counter
                        confidence = 0.75
                    
                    counter_move = (expected_user + 1) % 3
                    return confidence, counter_move

        # HYPER-INSTANT PATTERN DETECTION - Detect from just 2 moves!
        if len(us.history) >= 2:
            # Check last moves for various patterns
            recent_moves = [h["u_move"] for h in us.history[-6:]]
            recent_results = [h["result"] for h in us.history[-6:]]
            
            # PRIORITY 0: Check for 2-move alternating (Paper-Scissors-Paper-Scissors)
            if len(recent_moves) >= 2:
                # Check if we have A-B-A pattern (2-move alternating)
                alternating_strength = 0
                for i in range(2, len(recent_moves)):
                    if recent_moves[i] == recent_moves[i-2]:  # Same as 2 moves ago
                        if recent_results[i] in ["lose", "draw"]:  # And it was successful
                            alternating_strength += 4  # Strong evidence
                        else:
                            alternating_strength += 1  # Weak evidence
                
                # If we have ANY alternating evidence, predict immediately
                if alternating_strength >= 2:  # Even lower threshold
                    # Predict next move in alternating pattern
                    # In an A-B-A-B-... sequence, the next move equals the move from two steps ago.
                    # Using parity here can be wrong for odd-length prefixes (e.g., A-B-A -> expect B).
                    expected_user = recent_moves[-2]
                    confidence = min(0.92, 0.5 + 0.1 * alternating_strength)
                    counter_move = (expected_user + 1) % 3
                    return confidence, counter_move
            
            # PRIORITY 1: Check for 3-move cycles (Paper-Rock-Scissors-Paper-Rock-Scissors)
            if len(recent_moves) >= 3:
                # Check if we have a 3-move cycle pattern
                cycle_matches = 0
                for i in range(3, len(recent_moves)):
                    if recent_moves[i] == recent_moves[i-3]:  # Same as 3 moves ago
                        if recent_results[i] in ["lose", "draw"]:  # And it was successful
                            cycle_matches += 2  # Extra weight for successful cycles
                        else:
                            cycle_matches += 1
                
                # If we have 2+ cycle matches, predict next move in cycle
                if cycle_matches >= 2:
                    cycle_pos = len(recent_moves) % 3
                    if cycle_pos == 0:
                        expected_user = recent_moves[-3] if len(recent_moves) >= 3 else recent_moves[0]
                    elif cycle_pos == 1:
                        expected_user = recent_moves[-2] if len(recent_moves) >= 2 else recent_moves[1] 
                    else:
                        expected_user = recent_moves[-1]
                    
                    confidence = min(0.95, 0.6 + 0.1 * cycle_matches)
                    counter_move = (expected_user + 1) % 3
                    return confidence, counter_move
            
            # PRIORITY 2: Check for simple repeats  
            if len(recent_moves) >= 2:
                last_move = recent_moves[-1]
                repeat_count = 0
                for i in range(len(recent_moves) - 1, -1, -1):
                    if recent_moves[i] == last_move:
                        repeat_count += 1
                    else:
                        break
                
                # If user repeated same move 2+ times, MAXIMUM confidence counter
                if repeat_count >= 2:
                    # Check if they had success with this move
                    success_count = sum(1 for i in range(len(recent_results) - repeat_count, len(recent_results)) 
                                      if recent_results[i] in ["lose", "draw"])
                    
                    if success_count >= 1:  # Any success = assume they'll continue
                        confidence = min(0.99, 0.7 + 0.1 * repeat_count)  # Higher confidence with more repeats
                        counter_move = (last_move + 1) % 3  # Direct counter
                        return confidence, counter_move
        
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
                
            # Check for repeating moves with MAXIMUM emphasis
            if i > 0:
                prev_user = recent[i-1]["u_move"]
                if user_now == prev_user:
                    # ANY repeat gets weight, successful repeats get MASSIVE weight
                    if result in ["lose", "draw"]:
                        repeat_cnt += weight * 6.0  # 6x weight for successful repeats
                    else:
                        repeat_cnt += weight * 3.0  # 3x weight for any repeat
                    
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
        
        # MAXIMUM SENSITIVITY - activate on ANY pattern hint whatsoever
        min_threshold = 0.02 if len(us.history) <= 3 else 0.01  # Extremely low thresholds
        
        if max_prob < min_threshold:
            return None, None
            
        prev_ai = us.history[-1]["ai_move"] if us.history else None
        last_user = us.history[-1]["u_move"] if us.history else None
        
        if prev_ai is None:
            return None, None
        
        # MAXIMUM CONFIDENCE GUARANTEE - AI MUST WIN when pattern detected
        confidence_boost = 5.0  # 5x reported confidence for absolute dominance
        
        if alt_2_prob == max_prob and len(us.history) >= 2:
            # 2-move alternating pattern detected (Paper-Scissors-Paper-Scissors)
            user_moves = [h["u_move"] for h in us.history[-6:]]  # Look at more moves
            if len(user_moves) >= 2:
                # DIRECT ALTERNATING PREDICTION - If last move was Paper(1), next should be Scissors(2)
                # If last move was Scissors(2), next should be Paper(1)
                
                # Look at the established pattern
                if len(user_moves) >= 4:
                    # Check if it's truly alternating
                    pattern_a = user_moves[-2]  # 2 moves ago
                    pattern_b = user_moves[-1]  # 1 move ago
                    
                    # Predict they'll continue alternating
                    expected_user = pattern_a  # Will alternate back to first move
                else:
                    # For shorter history, predict based on position
                    if len(user_moves) % 2 == 0:  # Even position, expect first move of pattern
                        expected_user = user_moves[-2] if len(user_moves) >= 2 else user_moves[0]
                    else:  # Odd position, expect second move of pattern  
                        expected_user = user_moves[-1] if len(user_moves) >= 1 else 0
                
                anti_move = (expected_user + 1) % 3  # Counter their expected move
                # MAXIMUM confidence for alternating - it's very predictable
                ultra_confidence = min(alt_2_prob * confidence_boost * 3.0, 0.96)  # Extra 3x boost
                return ultra_confidence, anti_move
                
        elif cycle_3_prob == max_prob and len(us.history) >= 3:
            # 3-move cycle pattern detected (Paper-Rock-Scissors-Paper-Rock-Scissors)
            user_moves = [h["u_move"] for h in us.history[-9:]]  # Look at more moves
            if len(user_moves) >= 3:
                # DIRECT PATTERN PREDICTION - If last 3 moves were Paper(1)-Rock(0)-Scissors(2)
                # Then next should be Paper(1) again
                cycle_pos = len(user_moves) % 3
                
                # Get the move from 3 positions ago (start of current cycle)
                if len(user_moves) >= 3:
                    expected_user = user_moves[-3]  # Move from 3 positions ago
                else:
                    expected_user = user_moves[cycle_pos]  # Fallback
                
                anti_move = (expected_user + 1) % 3
                # MAXIMUM confidence for 3-move cycles - they're very predictable
                ultra_confidence = min(cycle_3_prob * confidence_boost * 2.0, 0.98)  # Extra 2x boost
                return ultra_confidence, anti_move
                
        elif repeat_prob == max_prob and last_user is not None:
            # They're repeating moves - GUARANTEED counter with maximum confidence
            return min(repeat_prob * confidence_boost, 0.99), (last_user + 1) % 3
            
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
        
    def _detect_user_meta_strategy(self, us: UserState) -> Dict[str, float]:
        """Detect user's meta-strategy type with confidence scores."""
        if len(us.history) < 5:
            return {"unknown": 1.0}
            
        strategies = {
            "counter_puncher": 0.0,    # Always counters AI moves
            "pattern_repeater": 0.0,   # Repeats same patterns  
            "anti_pattern": 0.0,       # Breaks patterns intentionally
            "random_switcher": 0.0,    # Random or chaos strategy
            "emotional_reactor": 0.0,  # Changes after wins/losses
            "meta_gamer": 0.0         # Adapts to AI behavior
        }
        
        recent = us.history[-10:] if len(us.history) >= 10 else us.history
        
        # Analyze patterns
        counter_moves = 0
        repeat_moves = 0
        emotional_changes = 0
        pattern_breaks = 0
        
        for i in range(1, len(recent)):
            prev_entry = recent[i-1]
            curr_entry = recent[i]
            
            prev_ai = prev_entry["ai_move"]
            prev_result = prev_entry["result"]
            prev_user = prev_entry["u_move"]
            curr_user = curr_entry["u_move"]
            
            if prev_ai is not None:
                # Counter-puncher detection
                if curr_user == (prev_ai + 1) % 3:
                    counter_moves += 1
                    
                # Pattern repeater detection
                if curr_user == prev_user:
                    repeat_moves += 1
                    
                # Emotional reactor detection
                if prev_result in ["win"] and curr_user != prev_user:  # AI won, user changed
                    emotional_changes += 1
                    
                # Anti-pattern detection - breaks established patterns
                if i >= 3:
                    # Check if user breaks their own 3-move pattern
                    pattern_move = recent[i-3]["u_move"]
                    if curr_user != pattern_move and curr_user != prev_user:
                        pattern_breaks += 1
        
        total_moves = len(recent) - 1
        if total_moves > 0:
            strategies["counter_puncher"] = counter_moves / total_moves
            strategies["pattern_repeater"] = repeat_moves / total_moves  
            strategies["emotional_reactor"] = emotional_changes / max(1, sum(1 for h in recent[:-1] if h["result"] == "win"))
            strategies["anti_pattern"] = pattern_breaks / max(1, total_moves - 2)
            
            # Random switcher - high entropy in moves
            user_moves = [h["u_move"] for h in recent]
            move_counts = [user_moves.count(i) for i in range(3)]
            entropy = -sum(p/len(user_moves) * np.log(p/len(user_moves) + 1e-8) for p in move_counts if p > 0)
            strategies["random_switcher"] = entropy / np.log(3)  # Normalized entropy
            
            # Meta gamer - adapts based on AI behavior
            strategies["meta_gamer"] = us.user_psychology_profile["meta_awareness"]
        
        return strategies
        
    def _calculate_recent_win_rate(self, us: UserState) -> float:
        """Calculate AI win rate in recent games."""
        if len(us.history) < 3:
            return 0.5  # Neutral if not enough data
            
        # Look at last 8 games or all if less
        recent = us.history[-8:] if len(us.history) >= 8 else us.history
        wins = sum(1 for h in recent if h["result"] == "win")
        
        return wins / len(recent)

    def _repeat_probability(self, us: UserState, window: int = 5) -> float:
        """Estimate probability that the user repeats their last move next round.
        Factors:
        - How often the last move appeared in a short recent window
        - Whether the last two moves are the same
        - Whether the last result was favorable to user (AI lose or draw)
        Returns value in [0,1].
        """
        if len(us.history) == 0:
            return 0.33
        last_user = us.history[-1]["u_move"]
        recent = us.history[-window:] if len(us.history) >= window else us.history
        freq = sum(1 for h in recent if h["u_move"] == last_user) / max(1, len(recent))
        streak = 1
        for i in range(len(us.history) - 2, -1, -1):
            if us.history[i]["u_move"] == last_user:
                streak += 1
            else:
                break
        last_res = us.history[-1]["result"]
        favorable = 1.0 if last_res in ["lose", "draw"] else 0.0  # user had success
        # Weighted combination
        p = 0.6 * freq + 0.25 * (streak / max(2, window)) + 0.15 * favorable
        return float(max(0.0, min(1.0, p)))

    def _copy_suspect_score(self, us: UserState, window: int = 5) -> float:
        """Measure tendency that user copies our last move.
        Recency-weighted count of events where user_move == previous ai_move.
        Heavier if that copy yielded user non-loss (AI lose/draw). Returns [0,1].
        """
        if len(us.history) < 2:
            return 0.0
        recent = us.history[-window:]
        score = 0.0
        total = 0.0
        n = len(recent)
        for i in range(1, n):
            prev_ai = recent[i-1]["ai_move"]
            u_now = recent[i]["u_move"]
            res_now = recent[i]["result"]
            if prev_ai is None:
                continue
            # recency weight (newer rounds count more)
            rec_w = 0.6 + 0.4 * (i / max(1, n-1))
            success_w = 1.8 if res_now in ["lose", "draw"] else 1.0
            w = rec_w * success_w
            total += w
            if u_now == prev_ai:
                score += w
        if total <= 0:
            return 0.0
        return float(max(0.0, min(1.0, score / total)))

    def _neural_prediction(self, us: UserState) -> Tuple[Optional[float], Optional[int]]:
        """Advanced neural prediction using sequence, psychology, and context."""
        if len(us.history) < 2:
            return None, None
            
        # SEQUENCE ENCODING - Last 8 moves as features (fix bounds checking)
        sequence_features = np.zeros(64)  # 8 moves × 8 features each
        recent_history = us.history[-8:] if len(us.history) >= 8 else us.history
        
        for i, entry in enumerate(recent_history):
            if i >= 8:  # Safety check - only process first 8 entries
                break
                
            base_idx = i * 8
            if base_idx + 7 >= 64:  # Safety check - ensure we don't exceed bounds
                break
                
            # Encode: [ai_move_onehot(3), user_move_onehot(3), result_onehot(2), timing(1)]
            ai_move = entry["ai_move"] if entry["ai_move"] is not None else 0
            user_move = entry["u_move"] 
            result = entry["result"]
            
            # One-hot encoding with bounds checking
            if base_idx + ai_move < 64:
                sequence_features[base_idx + ai_move] = 1.0  # AI move (positions 0-2)
            if base_idx + 3 + user_move < 64:
                sequence_features[base_idx + 3 + user_move] = 1.0  # User move (positions 3-5)
            
            # Result encoding - only use 2 bits instead of 3 to save space
            result_idx = {"win": 0, "draw": 1, "lose": 1}.get(result, 1)  # Map lose->draw to save space
            if base_idx + 6 + result_idx < 64:
                sequence_features[base_idx + 6 + result_idx] = 1.0  # Result (positions 6-7)
            
            # Timing factor
            if base_idx + 7 < 64:
                sequence_features[base_idx + 7] = min(1.0, len(us.history) / 20.0)  # Timing (position 7 of each block)
        
        # SEQUENCE PREDICTION
        sequence_logits = us.neural_weights["sequence_weights"].T @ sequence_features
        sequence_probs = self._softmax(sequence_logits)
        
        # PSYCHOLOGY PROFILE PREDICTION  
        psychology_vector = np.array([
            us.user_psychology_profile["counter_tendency"],
            us.user_psychology_profile["repeat_tendency"], 
            us.user_psychology_profile["beat_tendency"],
            us.user_psychology_profile["meta_awareness"],
            us.user_psychology_profile["emotional_reaction"],
            us.user_psychology_profile["pattern_complexity"]
        ])
        psychology_logits = us.neural_weights["psychology_weights"].T @ psychology_vector
        psychology_probs = self._softmax(psychology_logits)
        
        # CONTEXT PREDICTION - Current situation
        last_result = us.history[-1]["result"]
        last_ai = us.history[-1]["ai_move"] if us.history[-1]["ai_move"] is not None else 0
        last_user = us.history[-1]["u_move"]
        
        context_features = np.zeros(9)
        # Last result one-hot
        result_idx = {"win": 0, "draw": 1, "lose": 2}.get(last_result, 1)
        context_features[result_idx] = 1.0
        # Last moves one-hot  
        context_features[3 + last_ai] = 1.0
        context_features[6 + last_user] = 1.0
        
        context_logits = us.neural_weights["context_weights"].T @ context_features
        context_probs = self._softmax(context_logits)
        
        # ENSEMBLE PREDICTION with learned weights
        total_games = len(us.history)
        sequence_weight = min(0.6, total_games / 15.0)  # More weight with more data
        psychology_weight = 0.3
        context_weight = 0.4 - sequence_weight + 0.3  # Adjust context weight
        
        final_probs = (sequence_weight * sequence_probs + 
                      psychology_weight * psychology_probs + 
                      context_weight * context_probs)
        final_probs = final_probs / final_probs.sum()  # Normalize
        
        # Predict most likely user move
        predicted_user_move = int(np.argmax(final_probs))
        confidence = float(final_probs[predicted_user_move])
        
        # Only return if confidence is high enough
        if confidence >= 0.4:  # 40% threshold for neural prediction
            return confidence, predicted_user_move
            
        return None, None
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
        
    def _build_neural_features(self, us: UserState) -> Optional[np.ndarray]:
        """Build feature vector for neural prediction."""
        if len(us.history) < 1:
            return None
            
        features = np.zeros(79)  # 64 + 6 + 9
        
        # Sequence features (64 dims) - Fixed bounds checking
        recent_history = us.history[-8:] if len(us.history) >= 8 else us.history
        for i, entry in enumerate(recent_history):
            if i >= 8:  # Safety check
                break
                
            base_idx = i * 8
            if base_idx + 7 >= 64:  # Safety check
                break
                
            ai_move = entry["ai_move"] if entry["ai_move"] is not None else 0
            user_move = entry["u_move"]
            result = entry["result"]
            
            # Bounds-safe feature setting
            if base_idx + ai_move < 64:
                features[base_idx + ai_move] = 1.0
            if base_idx + 3 + user_move < 64:
                features[base_idx + 3 + user_move] = 1.0
            
            # Use 2-bit result encoding to save space
            result_idx = {"win": 0, "draw": 1, "lose": 1}.get(result, 1)
            if base_idx + 6 + result_idx < 64:
                features[base_idx + 6 + result_idx] = 1.0
            if base_idx + 7 < 64:
                features[base_idx + 7] = min(1.0, len(us.history) / 20.0)
        
        # Psychology features (6 dims)
        features[64:70] = np.array([
            us.user_psychology_profile["counter_tendency"],
            us.user_psychology_profile["repeat_tendency"],
            us.user_psychology_profile["beat_tendency"],
            us.user_psychology_profile["meta_awareness"],
            us.user_psychology_profile["emotional_reaction"],
            us.user_psychology_profile["pattern_complexity"]
        ])
        
        # Context features (9 dims)
        if us.history:
            last_result = us.history[-1]["result"]
            last_ai = us.history[-1]["ai_move"] if us.history[-1]["ai_move"] is not None else 0
            last_user = us.history[-1]["u_move"]
            
            result_idx = {"win": 0, "draw": 1, "lose": 2}.get(last_result, 1)
            features[70 + result_idx] = 1.0
            features[73 + last_ai] = 1.0
            features[76 + last_user] = 1.0
            
        return features

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
                
                # HYPER-SENSITIVE 2-move alternating (A-B-A-B like Paper-Scissors-Paper-Scissors)
                alt_2_matches = 0
                for i in range(2, len(user_moves)):
                    if user_moves[i] == user_moves[i-2]:  # Same as 2 moves ago
                        # Extra extra weight for successful alternating moves
                        if results[i] in ["lose", "draw"]:
                            alt_2_matches += 8.0  # Massive weight for successful patterns
                        else:
                            alt_2_matches += 3.0  # Still strong evidence
                
                # MUCH lower threshold for instant detection
                if alt_2_matches >= 1.0:  # Detect from just ONE match
                    patterns["alternating"] += alt_2_matches * 3.0  # Triple the impact
                
                # HYPER-SENSITIVE 3-move cycle (A-B-C-A-B-C) - Detect from 3 moves
                if len(user_moves) >= 3:
                    cycle_3_matches = 0
                    
                    # Check if last 3 moves form a specific pattern (e.g., Paper-Rock-Scissors)
                    if len(user_moves) >= 3:
                        # Check for consistency in pattern across all available moves
                        for i in range(3, len(user_moves)):
                            if user_moves[i] == user_moves[i-3]:  # Same as 3 moves ago
                                # Extra weight for success AND consistency
                                if results[i] in ["lose", "draw"]:
                                    cycle_3_matches += 6.0  # High weight for successful cycle moves
                                else:
                                    cycle_3_matches += 2.0  # Still evidence of pattern
                    
                    # Lower threshold for faster detection
                    if cycle_3_matches >= 1.0:  # Detect much earlier
                        patterns["cycle"] += cycle_3_matches * 2.0  # Double the impact
            
            # Trigger on ANY weighted pattern >= 0.2 (hyper-sensitive)
            if any(score >= 0.2 for score in patterns.values()):
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

    # ---- Dedicated simple detectors (reusable) ----
    def _detect_two_move_alternating(self, us: UserState, window: int = 6) -> Tuple[float, Optional[int]]:
        """Detect strong A-B-A-B alternating pattern in recent user moves.
        Returns (confidence[0..1], expected_user_next_move or None).
        """
        if len(us.history) < 3:
            return 0.0, None
        recent = us.history[-window:]
        user_moves = [h["u_move"] for h in recent]
        results = [h["result"] for h in recent]
        if len(user_moves) < 3:
            return 0.0, None
        strength = 0.0
        for i in range(2, len(user_moves)):
            if user_moves[i] == user_moves[i - 2] and user_moves[i] != user_moves[i - 1]:
                strength += 2.0 if results[i] in ["lose", "draw"] else 1.0
        if strength <= 0:
            return 0.0, None
        # Predict next user move equals move from two steps ago
        expected_user = user_moves[-2]
        conf = float(min(1.0, 0.15 * strength))
        return conf, int(expected_user)
