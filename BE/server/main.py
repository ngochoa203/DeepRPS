from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from BE.gamebrain import GameBrain


app = FastAPI(title="GameBrain RPS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

brain = GameBrain(state_dir="./rps_state", remember_history=True)


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
