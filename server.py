# server.py
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
import sys
import time
import json
import uuid
from datetime import datetime
import uvicorn
import os

from rag import MentalHealthRAG


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mental_health_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mental_health_server")

class ChatMessage(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session ID")
    is_mental_health: bool = Field(..., description="Whether query was mental health related")
    response_time: float = Field(..., description="Response time in seconds")
    timestamp: str = Field(..., description="Response timestamp")

class SessionInfo(BaseModel):
    session_id: str = Field(..., description="Session ID")
    created_at: str = Field(..., description="Session creation timestamp")
    message_count: int = Field(..., description="Number of messages in session")

class HealthCheck(BaseModel):
    status: str = Field(..., description="Server status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    groq_connected: bool = Field(..., description="Groq API connection status")
    chroma_connected: bool = Field(..., description="ChromaDB connection status")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")

class MentalHealthServer:
    def __init__(self):
        self.app = FastAPI(
            title="Mental Health Chatbot API",
            description="RAG-powered mental health assistant with Groq API and ChromaDB",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            logger.error("GROQ_API_KEY environment variable not set")
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.rag_system = None
        self.sessions = {}
        self.setup_middleware()
        self.setup_routes()
        
        logger.info("MentalHealthServer initialized")

    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            logger.info(f"Request {request_id} started: {request.method} {request.url}")
            
            response = await call_next(request)
            
            process_time = time.time() - start_time
            logger.info(f"Request {request_id} completed: {process_time:.3f}s - Status: {response.status_code}")
            
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            return response

    def setup_routes(self):
        
        @self.app.on_event("startup")
        async def startup_event():
            try:
                logger.info("Starting server initialization...")
                self.rag_system = MentalHealthRAG(groq_api_key=self.groq_api_key)
                
                sample_documents = [
                    {
                        "text": "Depression is treatable with therapy and medication. Symptoms include persistent sadness, loss of interest, and low energy.",
                        "source": "WHO",
                        "type": "condition",
                        "metadata": {"condition": "depression"}
                    },
                    {
                        "text": "Anxiety disorders involve excessive fear/worry. Treatment includes CBT, exposure therapy, and relaxation techniques.",
                        "source": "APA",
                        "type": "condition",
                        "metadata": {"condition": "anxiety"}
                    },
                    {
                        "text": "For mental health crisis, call/text 988 for free, confidential support 24/7.",
                        "source": "988 Lifeline",
                        "type": "resource",
                        "metadata": {"resource": "crisis"}
                    }
                ]
                
                self.rag_system.add_knowledge_documents(sample_documents)
                logger.info("Knowledge base initialized with sample data")
                
            except Exception as e:
                logger.error(f"Failed to initialize RAG system: {str(e)}")
                raise

        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            logger.info("Root endpoint accessed")
            return """
            <html>
                <head>
                    <title>Mental Health Chatbot API</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                        .header { background: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                        .endpoints { background: #f9f9f9; padding: 15px; border-radius: 5px; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>🤖 Mental Health Chatbot API</h1>
                        <p>RAG-powered mental health assistant with Groq API and ChromaDB</p>
                        <p>Developed by Group-33, B.Tech CSE Cloud Computing & Automation</p>
                    </div>
                    
                    <div class="endpoints">
                        <h2>Available Endpoints:</h2>
                        <ul>
                            <li><strong>GET /health</strong> - Health check</li>
                            <li><strong>POST /chat</strong> - Send a message to the chatbot</li>
                            <li><strong>GET /sessions</strong> - List active sessions</li>
                            <li><strong>DELETE /sessions/{session_id}</strong> - Delete a session</li>
                            <li><strong>GET /docs</strong> - API documentation</li>
                        </ul>
                    </div>
                    
                    <p>Visit <a href="/docs">/docs</a> for detailed API documentation.</p>
                </body>
            </html>
            """

        @self.app.get("/health", response_model=HealthCheck)
        async def health_check():
            try:
                health_status = HealthCheck(
                    status="healthy",
                    timestamp=datetime.now().isoformat(),
                    version="1.0.0",
                    groq_connected=self.groq_api_key is not None,
                    chroma_connected=self.rag_system is not None
                )
                logger.info("Health check passed")
                return health_status
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                raise HTTPException(status_code=500, detail="Health check failed")

        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(chat_message: ChatMessage):
            try:
                start_time = time.time()
                
                session_id = chat_message.session_id or str(uuid.uuid4())
                if session_id not in self.sessions:
                    self.sessions[session_id] = {
                        "created_at": datetime.now().isoformat(),
                        "messages": []
                    }
                    logger.info(f"Created new session: {session_id}")
                
                self.sessions[session_id]["messages"].append({
                    "role": "user",
                    "message": chat_message.message,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"Processing message in session {session_id}: {chat_message.message[:50]}...")

                response_data = self.rag_system.generate_response(chat_message.message)
                
                self.sessions[session_id]["messages"].append({
                    "role": "assistant",
                    "message": response_data["response"],
                    "timestamp": datetime.now().isoformat()
                })
                
                response_time = time.time() - start_time
                
                logger.info(f"Response generated in {response_time:.3f}s for session {session_id}")
                
                return ChatResponse(
                    response=response_data["response"],
                    session_id=session_id,
                    is_mental_health=response_data["is_mental_health"],
                    response_time=response_time,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Chat endpoint error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing message: {str(e)}"
                )

        @self.app.get("/sessions", response_model=List[SessionInfo])
        async def list_sessions():
            try:
                sessions_info = []
                for session_id, session_data in self.sessions.items():
                    sessions_info.append(SessionInfo(
                        session_id=session_id,
                        created_at=session_data["created_at"],
                        message_count=len(session_data["messages"])
                    ))
                
                logger.info(f"Listed {len(sessions_info)} active sessions")
                return sessions_info
                
            except Exception as e:
                logger.error(f"Error listing sessions: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error listing sessions: {str(e)}"
                )

        @self.app.get("/sessions/{session_id}", response_model=Dict)
        async def get_session(session_id: str):

            try:
                if session_id not in self.sessions:
                    logger.warning(f"Session not found: {session_id}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Session {session_id} not found"
                    )
                
                logger.info(f"Retrieved session: {session_id}")
                return {
                    "session_id": session_id,
                    **self.sessions[session_id]
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting session: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error getting session: {str(e)}"
                )

        @self.app.delete("/sessions/{session_id}")
        async def delete_session(session_id: str):

            try:
                if session_id not in self.sessions:
                    logger.warning(f"Session not found for deletion: {session_id}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Session {session_id} not found"
                    )
                
                del self.sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                
                return {"message": f"Session {session_id} deleted successfully"}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting session: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error deleting session: {str(e)}"
                )

        @self.app.get("/stats")
        async def get_stats():
            try:
                stats = {
                    "total_sessions": len(self.sessions),
                    "total_messages": sum(len(session["messages"]) for session in self.sessions.values()),
                    "active_since": datetime.now().isoformat(),
                    "rag_system_status": "connected" if self.rag_system else "disconnected"
                }
                
                if self.rag_system:
                    kb_stats = self.rag_system.get_collection_stats()
                    stats["knowledge_base"] = kb_stats
                
                logger.info("Retrieved server statistics")
                return stats
                
            except Exception as e:
                logger.error(f"Error getting stats: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error getting statistics: {str(e)}"
                )

        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            error_response = ErrorResponse(
                error=exc.detail,
                timestamp=datetime.now().isoformat()
            )
            logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response.dict()
            )

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            error_response = ErrorResponse(
                error="Internal server error",
                details=str(exc),
                timestamp=datetime.now().isoformat()
            )
            logger.error(f"Unhandled exception: {str(exc)}")
            return JSONResponse(
                status_code=500,
                content=error_response.dict()
            )

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

def main():
    try:
        if not os.getenv("GROQ_API_KEY"):
            print("ERROR: GROQ_API_KEY environment variable is required")
            print("Set it with: export GROQ_API_KEY=your_api_key")
            sys.exit(1)
        
        print("Starting Mental Health Chatbot Server...")
        print("Logs will be saved to mental_health_server.log")
        print("Server will be available at http://localhost:8000")
        print("API documentation at http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        
        server = MentalHealthServer()
        server.run()
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        logger.info("Server stopped by user")
    except Exception as e:
        print(f"Failed to start server: {e}")
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()