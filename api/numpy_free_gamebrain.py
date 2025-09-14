"""
Numpy-free GameBrain for serverless deployment with Supabase integration
"""
from typing import Dict, List, Optional, Tuple, Any
import json
import math
import random
import os
import asyncio
import httpx

def softmax(x):
    """Pure Python softmax implementation"""
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]

def best_response_move(p_opp):
    """Given opponent probability distribution, return best response move."""
    # Find the move that beats the opponent's most likely move
    responses = [(p_opp[(i+2) % 3]) for i in range(3)]
    return responses.index(max(responses))

class CounterStrategy:
    def __init__(self):
        self.weights = [0.0, 0.0, 0.0]

class SimpleGameBrain:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY", "")
        
        # Initialize with pure Python lists instead of numpy arrays
        self.user_moves = []
        self.ai_moves = []
        self.counter_strategy = CounterStrategy()
        self.pattern_weights = {
            "frequency": 0.3,
            "ngram": 0.4,
            "markov": 0.3
        }
        
        # Use lists instead of numpy arrays
        self.first_move_counts = [1.0, 1.0, 1.0]  # Rock, Paper, Scissors
        
        self.pattern_memory = {
            "ngram_1": [1.0, 1.0, 1.0],
            "ngram_2": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        }

    async def _load_user_from_db(self):
        """Load user state from Supabase"""
        if not self.supabase_url or not self.supabase_key:
            return None
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.supabase_url}/rest/v1/user_states?user_id=eq.{self.user_id}&select=*",
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Content-Type": "application/json"
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return data[0] if data else None
        except Exception as e:
            print(f"Error loading user from DB: {e}")
            return None

    async def _save_user_to_db(self, user_state: dict):
        """Save user state to Supabase"""
        if not self.supabase_url or not self.supabase_key:
            return
            
        try:
            async with httpx.AsyncClient() as client:
                # Try to update first
                response = await client.patch(
                    f"{self.supabase_url}/rest/v1/user_states?user_id=eq.{self.user_id}",
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal"
                    },
                    json=user_state
                )
                
                # If no rows updated, insert new record
                if response.status_code == 200 and response.headers.get("content-range", "").startswith("0-0"):
                    user_state["user_id"] = self.user_id
                    await client.post(
                        f"{self.supabase_url}/rest/v1/user_states",
                        headers={
                            "apikey": self.supabase_key,
                            "Authorization": f"Bearer {self.supabase_key}",
                            "Content-Type": "application/json"
                        },
                        json=user_state
                    )
        except Exception as e:
            print(f"Error saving user to DB: {e}")

    async def load_user_state(self):
        """Load user state from database"""
        db_state = await self._load_user_from_db()
        
        if db_state:
            try:
                self.user_moves = db_state.get("user_moves", [])
                self.ai_moves = db_state.get("ai_moves", [])
                self.pattern_memory = {
                    "ngram_1": db_state.get("ngram_1", [1.0, 1.0, 1.0]),
                    "ngram_2": db_state.get("ngram_2", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                }
                self.first_move_counts = db_state.get("first_move_counts", [1.0, 1.0, 1.0])
            except Exception as e:
                print(f"Error loading user state: {e}")
                self.pattern_memory = {
                    "ngram_1": [1.0, 1.0, 1.0],
                    "ngram_2": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                }

    async def save_user_state(self):
        """Save current user state to database"""
        user_state = {
            "user_moves": self.user_moves,
            "ai_moves": self.ai_moves,
            "ngram_1": self.pattern_memory["ngram_1"],
            "ngram_2": self.pattern_memory["ngram_2"],
            "first_move_counts": self.first_move_counts,
            "total_games": len(self.user_moves)
        }
        await self._save_user_to_db(user_state)

    def _frequency_analysis(self):
        """Analyze frequency of opponent moves"""
        if not self.user_moves:
            return [1/3, 1/3, 1/3]
        
        counts = [0, 0, 0]
        for move in self.user_moves:
            counts[move] += 1
        
        total = sum(counts)
        return [c/total for c in counts]

    def _pattern_analysis(self):
        """Analyze patterns in opponent moves"""
        if len(self.user_moves) < 2:
            return [1/3, 1/3, 1/3]
        
        # Simple n-gram analysis
        predictions = [0.0, 0.0, 0.0]
        
        # 1-gram (frequency)
        freq_pred = self._frequency_analysis()
        for i in range(3):
            predictions[i] += 0.5 * freq_pred[i]
        
        # 2-gram (if enough history)
        if len(self.user_moves) >= 2:
            last_move = self.user_moves[-1]
            for move in range(3):
                self.pattern_memory["ngram_2"][last_move][move] += 0.1
        
        # Normalize predictions
        total = sum(predictions) if sum(predictions) > 0 else 1
        return [p/total for p in predictions]

    async def predict(self, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Predict the next move"""
        await self.load_user_state()
        
        # Analyze opponent patterns
        p_opp = [1/3, 1/3, 1/3]  # Default uniform distribution
        
        if len(self.user_moves) > 0:
            freq_pred = self._frequency_analysis()
            pattern_pred = self._pattern_analysis()
            
            # Combine predictions
            for i in range(3):
                p_opp[i] = (0.4 * freq_pred[i] + 0.6 * pattern_pred[i])
            
            # Normalize
            total = sum(p_opp)
            p_opp = [p/total for p in p_opp]
        
        # Choose best response
        ai_move = best_response_move(p_opp)
        
        return {
            "move": ai_move,
            "confidence": max(p_opp),
            "strategy": "pattern_analysis",
            "opponent_prediction": p_opp
        }

    async def process_feedback(self, user_move: int, ai_move: int) -> Dict[str, Any]:
        """Process game result and update patterns"""
        await self.load_user_state()
        
        # Add moves to history
        self.user_moves.append(user_move)
        self.ai_moves.append(ai_move)
        
        # Update pattern memory
        self.pattern_memory["ngram_1"][user_move] += 1
        
        # Determine winner
        if user_move == ai_move:
            result = "tie"
        elif (user_move + 1) % 3 == ai_move:
            result = "ai_wins"
        else:
            result = "user_wins"
        
        # Save updated state
        await self.save_user_state()
        
        return {
            "result": result,
            "total_games": len(self.user_moves),
            "learning_active": True
        }

    async def save_game_state(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save additional game state data"""
        await self.save_user_state()
        return {"status": "saved", "user_id": self.user_id}