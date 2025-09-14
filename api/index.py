from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from .numpy_free_gamebrain import SimpleGameBrain

# Initialize FastAPI app
app = FastAPI(title="GameBrain RPS API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SimpleGameBrain for serverless
brain = SimpleGameBrain()

class PredictReq(BaseModel):
    user_hint: Optional[str] = None
    ctx: Optional[Dict[str, Any]] = None

class PredictRes(BaseModel):
    ai_move: int
    meta: Dict[str, Any]

class FeedbackReq(BaseModel):
    user_hint: Optional[str] = None
    ai_move: int
    user_move: int
    dt_ms: int
    result: str

@app.get("/")
async def root():
    return {"message": "GameBrain RPS API is running!"}

@app.post("/predict", response_model=PredictRes)
async def predict(req: PredictReq):
    result = await brain.predict(req.ctx)
    return PredictRes(ai_move=result["move"], meta=result)

@app.post("/feedback")
async def feedback(req: FeedbackReq):
    result = await brain.process_feedback(req.user_move, req.ai_move)
    return {"status": "ok", "result": result}

@app.post("/save")
async def save():
    result = await brain.save_game_state({})
    return {"status": "saved", "result": result}

# This is required for Vercel
handler = app