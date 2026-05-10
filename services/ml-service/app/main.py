"""
MIRA ML Service — FastAPI Application Entry Point.
"""

from __future__ import annotations

import logging

import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.routers.analysis import router as analysis_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MIRA ML Service",
    description="Multimodal emotion analysis, clustering, drift detection, and crisis classification.",
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
    return {"status": "healthy", "service": "ml-service"}

app.include_router(analysis_router, prefix="", tags=["analysis"])
