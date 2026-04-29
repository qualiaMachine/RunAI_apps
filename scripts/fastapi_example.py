#!/usr/bin/env python3
"""Minimal FastAPI app for the RunAI deployment pattern in docs/06-fastapi-app.md.

A reference scaffold for exposing a custom HTTP API on the cluster — proxies a
chat completion to a vLLM Inference workload over internal cluster DNS and
returns the answer. Copy this file as a starting point for an internal wiki
backend, a RAG endpoint, a webhook receiver, etc.

Launch:
    uvicorn scripts.fastapi_example:app --host 0.0.0.0 --port 8000

Environment variables:
    VLLM_BASE_URL  - vLLM workload URL incl. /v1 suffix. When unset, /chat
                     returns 503 but /health and /info still work, so you can
                     stand up the FastAPI workload first and wire in the model
                     URL afterward.
                     Example: http://qwen-Qwen2.5--7B--Instruct.runai-myproject.svc.cluster.local/v1
    VLLM_MODEL     - Model ID passed to /v1/chat/completions.
                     Example: Qwen/Qwen2.5-7B-Instruct
    ALLOWED_ORIGIN - CORS allow-origin. Default "*" is fine for VPN-only
                     testing; tighten to your site's origin before exposing
                     externally.
    SERVICE_NAME   - Shown in /info. Default "fastapi-example".
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL")
VLLM_MODEL = os.environ.get("VLLM_MODEL")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
SERVICE_NAME = os.environ.get("SERVICE_NAME", "fastapi-example")

app = FastAPI(title=SERVICE_NAME, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_client: AsyncOpenAI | None = (
    AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-used") if VLLM_BASE_URL else None
)


class ChatRequest(BaseModel):
    question: str
    system: str = "You are a concise research assistant."
    max_tokens: int = 256


class ChatResponse(BaseModel):
    answer: str
    model: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
async def info() -> dict:
    return {
        "service": SERVICE_NAME,
        "vllm_base_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL,
        "vllm_configured": _client is not None and bool(VLLM_MODEL),
        "allowed_origin": ALLOWED_ORIGIN,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if _client is None or not VLLM_MODEL:
        raise HTTPException(
            status_code=503,
            detail="VLLM_BASE_URL and VLLM_MODEL must be set to use /chat",
        )
    resp = await _client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.question},
        ],
        max_tokens=req.max_tokens,
        temperature=0,
    )
    return ChatResponse(answer=resp.choices[0].message.content, model=VLLM_MODEL)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
