from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

class ChatMessage(BaseModel):
    message: str
    context: dict = None

@router.post("/message")
async def send_message(payload: ChatMessage):
    """Kirim pesan ke LLM consultant."""
    # Mock response
    return {"reply": f"Ini adalah balasan mock untuk pesan: {payload.message}"}

@router.get("/history")
async def get_history():
    """Riwayat chat session."""
    return {"history": []}
