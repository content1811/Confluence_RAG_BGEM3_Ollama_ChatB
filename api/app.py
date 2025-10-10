from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
from pathlib import Path
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from db.sqlite import SQLiteDB
from embeddings.encoder import EmbeddingEncoder
from embeddings.vector_store import VectorStore
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker
from llm.client import LocalLLMClient
from pipeline.query_pipeline import QueryPipeline

app = FastAPI(title="Confluence RAG API", version="1.0.0")

# Global state
config = None
pipeline = None
# Simple dict to store session histories
sessions = {}

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    
class QueryResponse(BaseModel):
    answer: str
    citations: list
    confidence: str
    chunks_used: Optional[int] = 0
    session_id: str

class SessionResponse(BaseModel):
    session_id: str

@app.on_event("startup")
async def startup_event():
    global config, pipeline
    
    config = load_config()
    
    db = SQLiteDB(config.paths.sqlite_path)
    db.connect()
    
    encoder = EmbeddingEncoder(
        model_name=config.embeddings.model,
        device=config.embeddings.device
    )
    
    vector_store = VectorStore(config.paths.chroma_dir)
    
    hybrid_search = HybridSearch(db, vector_store, encoder, config)
    
    reranker = None
    if config.rerank.enabled:
        reranker = Reranker(
            model_name=config.rerank.model,
            device="cpu"
        )
    
    llm_client = LocalLLMClient(
        base_url=config.llm.base_url,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
    )
    
    pipeline = QueryPipeline(db, hybrid_search, reranker, llm_client, config)
    
    print("✓ RAG pipeline initialized")

@app.post("/session", response_model=SessionResponse)
async def create_session():
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    return {"session_id": session_id}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the RAG system with conversation context"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Create session if not provided
    session_id = request.session_id
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = []
    
    # Get history for this session
    history = sessions[session_id]
    
    # Query with history
    result = pipeline.query(request.question, history=history)
    
    # Update history
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": result['answer']})
    
    # Keep only last 20 messages (10 turns)
    if len(history) > 20:
        sessions[session_id] = history[-20:]
    else:
        sessions[session_id] = history
    
    result['session_id'] = session_id
    return result

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    if session_id in sessions:
        sessions[session_id] = []
    return {"message": "Session cleared"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "pipeline": pipeline is not None,
        "active_sessions": len(sessions)
    }