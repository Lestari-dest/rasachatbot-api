from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
import os
from datetime import datetime
import logging

# Add this line for better .env loading
from dotenv import load_dotenv

from chatbot.rasa_chatbot import RasaChatbot

# Load environment variables
load_dotenv()


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RasaChatbot API",
    description="Enhanced Natural Adaptive Chatbot with Emotion Analysis",
    version="1.0.0"
)

# CORS middleware untuk Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk production, ganti dengan domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = None

# Pydantic models untuk request/response
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default"
    reset_session: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    sentiment: str
    confidence: float
    transition: Optional[str]
    empathy_level: int
    turn_count: int
    timestamp: str
    user_id: str
    # Data untuk penelitian
    research_data: Dict = {
        "style_analysis": {},
        "personality_profile": {},
        "short_term_memory": [],
        "special_case": False
    }

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    global chatbot
    try:
        model_name = os.getenv("MODEL_NAME", "tartarmee/indobert-sentiment-mental-health")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        chatbot = RasaChatbot(model_name, gemini_api_key)
        logger.info("✅ RasaChatbot initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="RasaChatbot API is running",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if chatbot else "unhealthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        if not chatbot:
            raise HTTPException(status_code=500, detail="Chatbot not initialized")
        
        if request.reset_session:
            chatbot.reset_chat_session()
        
        # Process message
        result = chatbot.chat(request.message)
        
        # Prepare research data
        research_data = {
            "style_analysis": result.get("style_analysis", {}),
            "personality_profile": chatbot.user_personality_profile.copy(),
            "short_term_memory": [
                {
                    "user": memory["user"],
                    "sentiment": memory["sentiment"],
                    "turn": memory["turn"]
                } for memory in chatbot.short_term_memory
            ],
            "special_case": result.get("special_case", False),
            "confidence_threshold": chatbot.confidence_threshold,
            "previous_sentiment": chatbot.previous_sentiment
        }
        
        return ChatResponse(
            response=result["response"],
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            transition=result.get("transition"),
            empathy_level=result.get("empathy_level", 1),
            turn_count=chatbot.turn_count,
            timestamp=datetime.now().isoformat(),
            user_id=request.user_id,
            research_data=research_data
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/reset")
async def reset_session(user_id: Optional[str] = "default"):
    """Reset chat session"""
    try:
        if not chatbot:
            raise HTTPException(status_code=500, detail="Chatbot not initialized")
        
        chatbot.reset_chat_session()
        return {"message": "Session reset successfully", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get chatbot statistics for research"""
    try:
        if not chatbot:
            raise HTTPException(status_code=500, detail="Chatbot not initialized")
        
        return {
            "turn_count": chatbot.turn_count,
            "personality_profile": chatbot.user_personality_profile,
            "memory_size": len(chatbot.short_term_memory),
            "previous_sentiment": chatbot.previous_sentiment,
            "confidence_threshold": chatbot.confidence_threshold,
            "available_sentiments": chatbot.labels
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
