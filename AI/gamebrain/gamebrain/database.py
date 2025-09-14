"""
Database adapters for cloud deployment with persistent user data storage.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import os
import sqlite3
from pathlib import Path


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters."""
    
    @abstractmethod
    async def get_user_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user state from database."""
        pass
    
    @abstractmethod
    async def save_user_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save user state to database."""
        pass
    
    @abstractmethod
    async def list_users(self) -> list[str]:
        """List all user IDs."""
        pass


class SQLiteAdapter(DatabaseAdapter):
    """SQLite adapter for local/development use."""
    
    def __init__(self, db_path: str = "data/gamebrain.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_states (
                    user_id TEXT PRIMARY KEY,
                    state_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_timestamp 
                AFTER UPDATE ON user_states
                BEGIN
                    UPDATE user_states SET updated_at = CURRENT_TIMESTAMP 
                    WHERE user_id = NEW.user_id;
                END
            """)
    
    async def get_user_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user state from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT state_data FROM user_states WHERE user_id = ?", 
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None
    
    async def save_user_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save user state to SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_states (user_id, state_data)
                    VALUES (?, ?)
                """, (user_id, json.dumps(state)))
            return True
        except Exception as e:
            print(f"Error saving user state: {e}")
            return False
    
    async def list_users(self) -> list[str]:
        """List all user IDs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT user_id FROM user_states")
            return [row[0] for row in cursor.fetchall()]


class SupabaseAdapter(DatabaseAdapter):
    """Supabase adapter for cloud deployment."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        # Note: Would need supabase-py package in production
        
    async def get_user_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user state from Supabase."""
        # Implementation would use supabase client
        # For now, return None to fall back to local storage
        return None
    
    async def save_user_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save user state to Supabase."""
        # Implementation would use supabase client
        return False
    
    async def list_users(self) -> list[str]:
        """List all user IDs."""
        return []


def get_database_adapter() -> DatabaseAdapter:
    """Factory function to get appropriate database adapter."""
    # Check for cloud database config
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if supabase_url and supabase_key:
        return SupabaseAdapter(supabase_url, supabase_key)
    
    # Default to SQLite
    db_path = os.getenv("DATABASE_PATH", "data/gamebrain.db")
    return SQLiteAdapter(db_path)