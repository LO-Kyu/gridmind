"""
GridMind-RL FastAPI Server Wrapper
Proxies requests to the Go environment server (or provides fallback endpoints)
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import httpx

app = FastAPI(title="GridMind-RL", version="1.0.0")

# Go server address (set via environment or default)
GO_SERVER_URL = os.getenv("GO_SERVER_URL", "http://localhost:8000")

# Timeout for Go server calls
TIMEOUT = 30


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(f"{GO_SERVER_URL}/health")
            return resp.json()
    except Exception:
        # Fallback response if Go server unreachable
        return {"status": "ok", "mode": "python"}


@app.get("/state")
async def get_state():
    """Get environment state."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(f"{GO_SERVER_URL}/state")
            return resp.json()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=503)


@app.post("/reset")
async def reset(request: dict):
    """Reset environment."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{GO_SERVER_URL}/reset", json=request)
            return resp.json()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=503)


@app.post("/step")
async def step(request: dict):
    """Step environment."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{GO_SERVER_URL}/step", json=request)
            return resp.json()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=503)


@app.get("/grade")
async def grade():
    """Grade environment."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(f"{GO_SERVER_URL}/grade")
            return resp.json()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=503)


def main():
    """Entry point for server."""
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
