"""
Simplified GameBrain for serverless deployment with Supabase integration
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import math
import random
import os
import asyncio
import httpx

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def best_response_move(p_opp):
    """Given opponent probability distribution, return best response move."""
    return int(np.argmax([(p_opp[(i+2) % 3]) for i in range(3)]))

class SimpleBandit:
    def __init__(self):
        self.weights = np.zeros(3)
    
    def select_action(self, context):
        return random.randint(0, 2), {"bandit": "simple"}
    
    def update(self, action, reward, context):
        pass

class SimpleGameBrain:
    def __init__(self):
        self.rng = random.Random(42)
        self.users = {}
        self.first_move_counts = np.array([1.0, 1.0, 1.0])  # Rock, Paper, Scissors
        
        # Supabase config
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        self.use_supabase = bool(self.supabase_url and self.supabase_key)
    
    async def _load_user_from_db(self, uid: str) -> Optional[Dict]:
        """Load user state from Supabase"""
        if not self.use_supabase:
            return None
            
        try:
            import httpx
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.supabase_url}/rest/v1/user_states?user_id=eq.{uid}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        return data[0]["state_data"]
        except Exception as e:
            print(f"Error loading user from DB: {e}")
        
        return None
    
    async def _save_user_to_db(self, uid: str, state: Dict) -> bool:
        """Save user state to Supabase"""
        if not self.use_supabase:
            return False
            
        try:
            import httpx
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates"
            }
            
            payload = {
                "user_id": uid,
                "state_data": state
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.supabase_url}/rest/v1/user_states",
                    headers=headers,
                    json=payload
                )
                
                return response.status_code in [200, 201]
        except Exception as e:
            print(f"Error saving user to DB: {e}")
        
        return False
        
    async def predict(self, user_hint: Optional[str], ctx: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        uid = user_hint if user_hint else "anon"
        
        # Get or create user state
        if uid not in self.users:
            # Try to load from database first
            db_state = await self._load_user_from_db(uid)
            if db_state:
                self.users[uid] = {
                    "history": db_state.get("history", []),
                    "ngram_1": np.array(db_state.get("ngram_1", [1.0, 1.0, 1.0])),
                    "ngram_2": np.array(db_state.get("ngram_2", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])),
                    "last_move": db_state.get("last_move")
                }
            else:
                self.users[uid] = {
                    "history": [],
                    "ngram_1": np.ones(3),
                    "ngram_2": np.ones((3, 3)),
                    "last_move": None
                }
        
        us = self.users[uid]
        
        # Simple prediction based on history
        if len(us["history"]) == 0:
            # First move - avoid Paper, prefer Rock
            ai_move = 0  # Rock
            policy = "first-move"
        elif len(us["history"]) == 1:
            # Second move - counter their first move
            last_user = us["history"][-1]["u_move"]
            ai_move = (last_user + 1) % 3  # Counter their move
            policy = "counter-first"
        else:
            # Pattern detection
            recent_moves = [h["u_move"] for h in us["history"][-4:]]
            
            # Check for alternating pattern
            if len(recent_moves) >= 3 and recent_moves[-1] == recent_moves[-3]:
                # Alternating detected - predict next in pattern
                if len(recent_moves) % 2 == 0:
                    predicted_user = recent_moves[-2]
                else:
                    predicted_user = recent_moves[-1]
                ai_move = (predicted_user + 1) % 3
                policy = "alternating-counter"
            else:
                # Counter most common recent move
                if recent_moves:
                    from collections import Counter
                    most_common = Counter(recent_moves).most_common(1)[0][0]
                    ai_move = (most_common + 1) % 3
                    policy = "frequency-counter"
                else:
                    ai_move = self.rng.randint(0, 2)
                    policy = "random"
        
        # Create prediction distribution for display
        p_opp = np.ones(3) / 3
        if len(us["history"]) > 0:
            # Simple frequency-based prediction
            recent_moves = [h["u_move"] for h in us["history"][-5:]]
            for move in recent_moves:
                p_opp[move] += 0.2
            p_opp = p_opp / np.sum(p_opp)
        
        meta = {
            "uid": uid,
            "policy": policy,
            "p_opp": p_opp.tolist(),
            "br_move": best_response_move(p_opp),
            "history_length": len(us["history"])
        }
        
        return ai_move, meta
    
    async def feedback(self, user_hint: Optional[str], ai_move: int, user_move: int, dt_ms: int, result: str):
        uid = user_hint if user_hint else "anon"
        
        if uid not in self.users:
            self.users[uid] = {
                "history": [],
                "ngram_1": np.ones(3),
                "ngram_2": np.ones((3, 3)),
                "last_move": None
            }
        
        us = self.users[uid]
        
        # Add to history
        us["history"].append({
            "ai_move": ai_move,
            "u_move": user_move,
            "result": result,
            "dt_ms": dt_ms
        })
        
        # Keep only last 20 moves to prevent memory issues
        if len(us["history"]) > 20:
            us["history"] = us["history"][-20:]
        
        # Update n-grams
        us["ngram_1"][user_move] += 1.0
        if us["last_move"] is not None:
            us["ngram_2"][us["last_move"], user_move] += 1.0
        us["last_move"] = user_move
        
        # Save to database
        state_to_save = {
            "history": us["history"],
            "ngram_1": us["ngram_1"].tolist(),
            "ngram_2": us["ngram_2"].tolist(),
            "last_move": us["last_move"]
        }
        await self._save_user_to_db(uid, state_to_save)
    
    def save(self):
        # In serverless, we can't really save to disk
        # This is a no-op for now
        pass