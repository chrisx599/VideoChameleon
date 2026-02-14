import asyncio
import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

def _init_env():
    base = Path(__file__).resolve().parents[1]
    env_file = base / ".env"
    if not env_file.exists():
        raise RuntimeError("Config missing: please copy univa/.env.example to univa/.env and fill your keys.")
    load_dotenv(dotenv_path=str(env_file), override=False)

_init_env()

from univa.univa_agent import ReActSystem


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['UniVA_HTTP_SERVER_MODE'] = 'true'

app = FastAPI(title="UniVA Chat API", version="0.1.0")

# CORS middleware setting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

global_react_system: Optional[ReActSystem] = None


async def initialize_global_agents() -> ReActSystem:
    global global_react_system
    
    if global_react_system:
        return global_react_system
    
    global_react_system = ReActSystem()
    await global_react_system.__aenter__()
    
    logger.info("Global ReActSystem initialized")
    
    return global_react_system



class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    project_id: Optional[str] = None
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    pad_sec: Optional[float] = None
    max_segments: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str


async def stream_chat_response(
    user_id: str,
    session_id: str,
    user_prompt: str,
    project_id: Optional[str] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    pad_sec: Optional[float] = None,
    max_segments: Optional[int] = None,
):
    """
    Stream chat response as SSE
    
    return SSE stream compatible with useCompletion
    """
    try:
        init_message = {
            'type': 'content',
            'content': 'Initializing system, please wait...'
        }
        init_json = json.dumps(init_message, ensure_ascii=False)
        logger.info("Sending system initialization message")
        sse_init = f"data: {init_json}\n\n"
        yield sse_init.encode('utf-8') if isinstance(sse_init, str) else sse_init
        await asyncio.sleep(0.01)
        
        system = await initialize_global_agents()
        
        logger.info(f"Streaming task execution for user {user_id}, session {session_id}")
        
        # calling agent's streaming execution method
        async for event in system.execute_task_stream(
            session_id,
            user_prompt,
            project_id=project_id,
            t_start=t_start,
            t_end=t_end,
            pad_sec=pad_sec if pad_sec is not None else 8.0,
            max_segments=max_segments if max_segments is not None else 12,
        ):
            if event.get('type') == 'finish':
                event['session_id'] = session_id
            
            json_str = json.dumps(event, ensure_ascii=False)
            logger.info(f"Sending SSE event: {event.get('type', 'unknown')}")
            logger.debug(f"Event details:\n{json_str}")
            
            sse_message = f"data: {json_str}\n\n"
            yield sse_message.encode('utf-8') if isinstance(sse_message, str) else sse_message
            
            await asyncio.sleep(0.01)
        
        logger.info("Stream completed successfully")
        
    except Exception as e:
        logger.error(f"Error in stream_chat_response: {e}")
        logger.error(traceback.format_exc())
        error_message = f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        yield error_message.encode('utf-8') if isinstance(error_message, str) else error_message

@app.post("/chat/stream")
async def chat(request: ChatRequest, req: Request):
    """
    Unified chat request handling endpoint - using ReActSystem (streaming)

    Returns a streaming response compatible with Vercel AI SDK
    """
    try:
        user_id = "local-user"
        
        # generate session_id if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"POST /chat/stream - user: {user_id}, session: {session_id}, prompt: {request.prompt[:50]}...")
        
        return StreamingResponse(
            stream_chat_response(
                user_id,
                session_id,
                request.prompt,
                project_id=request.project_id,
                t_start=request.t_start,
                t_end=request.t_end,
                pad_sec=request.pad_sec,
                max_segments=request.max_segments,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/stream")
async def chat_get(
    prompt: str,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    pad_sec: Optional[float] = None,
    max_segments: Optional[int] = None,
    request: Request = None
):
    """
    Get method chat endpoint for streaming responses.
    Returns a streaming response compatible with Vercel AI SDK.
    """
    try:
        user_id = "local-user"
        
        sid = session_id or str(uuid.uuid4())
        
        return StreamingResponse(
            stream_chat_response(
                user_id,
                sid,
                prompt,
                project_id=project_id,
                t_start=t_start,
                t_end=t_end,
                pad_sec=pad_sec,
                max_segments=max_segments,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )

@app.get("/")
async def root():
    return {"message": "UniVA Chat API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
