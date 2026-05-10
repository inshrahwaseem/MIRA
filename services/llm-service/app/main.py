"""
MIRA LLM Service — FastAPI Application Entry Point.
"""

from __future__ import annotations

import logging

import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.routers.llm import router as llm_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MIRA LLM Service",
    description="Therapeutic AI companion with RAG knowledge, long-term memory, and crisis response.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
        
    secret = request.headers.get("x-internal-secret")
    if secret != os.getenv("INTERNAL_API_SECRET", "dev_secret"):
        return JSONResponse(
            status_code=403, 
            content={"error": "Forbidden", "request_id": str(uuid.uuid4())}
        )
        
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, 
            content={"error": "Something went wrong", "request_id": str(uuid.uuid4())}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llm-service"}

app.include_router(llm_router, prefix="", tags=["llm"])
