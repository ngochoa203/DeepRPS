from __future__ import annotations

import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any

from gamebrain import GameBrain


app = FastAPI(title="GameBrain RPS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use environment-specific state directory
state_dir = os.getenv("STATE_DIR", "./rps_state")
brain = GameBrain(state_dir=state_dir, remember_history=True)

# Mount static files if available (for production deployment)
static_path = os.getenv("STATIC_PATH", "./static")
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")


class PredictReq(BaseModel):
    user_hint: Optional[str] = None
    ctx: Optional[Dict[str, Any]] = None


class PredictRes(BaseModel):
    ai_move: int
    meta: Dict[str, Any]


class FeedbackReq(BaseModel):
    user_hint: Optional[str]
    ai_move: int
    user_move: int
    dt_ms: int
    result: str


@app.post("/predict", response_model=PredictRes)
def predict(req: PredictReq):
    ai_move, meta = brain.predict(req.user_hint, req.ctx)
    return PredictRes(ai_move=ai_move, meta=meta)


@app.post("/feedback")
def feedback(req: FeedbackReq):
    brain.feedback(req.user_hint, req.ai_move, req.user_move, req.dt_ms, req.result)
    return {"ok": True}


@app.post("/save")
def save():
    brain.save()
    return {"ok": True}


# Optional: simple 2-player relay via WebSocket (roomless broadcast example)
clients = set()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            # broadcast to others
            for c in list(clients):
                if c is not ws:
                    try:
                        await c.send_text(msg)
                    except Exception:
                        pass
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)
