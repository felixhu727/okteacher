# api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from agent import initialize_rag_agent, process_message, stream_messages
import logging
import traceback
import sys
from typing import Optional
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger(__name__)

# 全局变量，用于存储初始化的代理
agent_executor = None

# 尝试初始化RAG代理，但捕获任何异常
try:
    logger.info("正在初始化RAG代理...")
    agent_executor = initialize_rag_agent()
    logger.info("RAG代理初始化成功")
except Exception as e:
    logger.error(f"RAG代理初始化失败: {str(e)}")
    logger.error(traceback.format_exc())
    # 不退出，允许API启动，但在请求时返回错误

app = FastAPI(
    title="RAG Agent API", 
    version="1.0.0",
    docs_url="/docs",  # 启用Swagger UI
    redoc_url="/redoc"  # 启用ReDoc
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="消息内容不能为空")
    thread_id: str = Field("default_thread", description="会话ID")

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

class ErrorResponse(BaseModel):
    detail: str
    status: str = "error"

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"全局异常: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误", "status": "error"}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"请求验证错误: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "status": "error"}
    )

@app.get("/")
async def root():
    return {"message": "RAG Agent API is running", "status": "success"}

@app.post("/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def chat_endpoint(request: ChatRequest):
    """处理聊天请求并返回完整响应"""
    start_time = time.time()
    
    # 检查代理是否初始化成功
    if agent_executor is None:
        logger.error("RAG代理未初始化，无法处理请求")
        raise HTTPException(
            status_code=503, 
            detail="服务暂时不可用，RAG代理初始化失败"
        )
    
    try:
        logger.info(f"处理消息: {request.message[:100]}... (thread_id: {request.thread_id})")
        
        response = process_message(
            agent_executor, 
            request.message, 
            request.thread_id
        )
        
        processing_time = time.time() - start_time
        logger.info(f"请求处理完成，耗时: {processing_time:.2f}秒")
        
        return ChatResponse(response=response)
        
    except Exception as e:
        logger.error(f"处理消息时出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"处理消息时出错: {str(e)}")

@app.post("/chat/stream", responses={500: {"model": ErrorResponse}})
async def chat_stream_endpoint(request: ChatRequest):
    """流式处理聊天请求"""
    start_time = time.time()
    
    # 检查代理是否初始化成功
    if agent_executor is None:
        logger.error("RAG代理未初始化，无法处理流式请求")
        raise HTTPException(
            status_code=503, 
            detail="服务暂时不可用，RAG代理初始化失败"
        )
    
    try:
        logger.info(f"处理流式消息: {request.message[:100]}... (thread_id: {request.thread_id})")
        
        def event_stream():
            try:
                for chunk in agent.stream_messages(
                    agent_executor, 
                    request.message, 
                    request.thread_id
                ):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
                
                processing_time = time.time() - start_time
                logger.info(f"流式请求处理完成，耗时: {processing_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"流式处理时出错: {str(e)}")
                logger.error(traceback.format_exc())
                yield f"data: ERROR: {str(e)}\n\n"
        
        return StreamingResponse(
            event_stream(), 
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"流式处理初始化时出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"流式处理初始化时出错: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    health_status = {
        "status": "healthy" if agent_executor is not None else "unhealthy",
        "agent_initialized": agent_executor is not None,
        "timestamp": time.time()
    }
    
    if agent_executor is None:
        logger.warning("健康检查: 服务不正常，代理未初始化")
    else:
        logger.info("健康检查: 服务正常")
    
    return health_status

@app.get("/info")
async def info():
    """获取API信息"""
    return {
        "name": "RAG Agent API",
        "version": "1.0.0",
        "agent_initialized": agent_executor is not None,
        "endpoints": [
            {"path": "/", "method": "GET", "description": "根端点"},
            {"path": "/chat", "method": "POST", "description": "处理聊天请求"},
            {"path": "/chat/stream", "method": "POST", "description": "流式处理聊天请求"},
            {"path": "/health", "method": "GET", "description": "健康检查"},
            {"path": "/info", "method": "GET", "description": "API信息"},
            {"path": "/docs", "method": "GET", "description": "Swagger文档"},
            {"path": "/redoc", "method": "GET", "description": "ReDoc文档"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config=None,  # 使用默认日志配置
        access_log=True  # 启用访问日志
    )