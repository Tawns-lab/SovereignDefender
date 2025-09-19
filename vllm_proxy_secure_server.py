from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Iterable, List

import importlib.util

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# NVML detection must avoid try/except around imports per guidelines
_NVML_SPEC = importlib.util.find_spec("pynvml")
if _NVML_SPEC is not None:
    import pynvml  # type: ignore
else:
    pynvml = None  # type: ignore

NVML_AVAILABLE = False
if pynvml is not None:  # type: ignore
    try:
        pynvml.nvmlInit()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - depends on GPU runtime
        pynvml = None  # type: ignore
    else:
        NVML_AVAILABLE = True

# Prometheus metrics
REQ_COUNT = Counter(
    "vllm_proxy_requests_total",
    "Total number of proxied requests",
    ["method", "status", "path_template"],
)
REQ_LATENCY = Histogram(
    "vllm_proxy_request_latency_seconds",
    "Latency of proxied requests in seconds",
    ["method", "path_template"],
)
GPU_UTIL = Gauge("vllm_gpu_utilization_percent", "GPU utilization percent", ["gpu_index"])
GPU_MEM_USED = Gauge("vllm_gpu_memory_used_bytes", "GPU memory used bytes", ["gpu_index"])
GPU_MEM_TOTAL = Gauge("vllm_gpu_memory_total_bytes", "GPU memory total bytes", ["gpu_index"])
DEPLOYS = Counter("deploys_total", "Count of deploy events", ["service", "env"])

app = FastAPI(title="vLLM proxy + metrics")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def enforce_https(request: Request, call_next):
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    if proto != "https":
        raise HTTPException(status_code=403, detail="HTTPS required")
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault(
        "Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload"
    )
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Content-Security-Policy", "default-src 'self';")
    return response


@app.middleware("http")
async def max_size_middleware(request: Request, call_next):
    max_bytes = int(os.environ.get("MAX_REQUEST_BODY_BYTES", str(2 * 1024 * 1024)))
    cl = request.headers.get("content-length")
    if cl and int(cl) > max_bytes:
        raise HTTPException(status_code=413, detail="Payload too large")
    return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[".modal.run", "ai.example.com", "localhost", "127.0.0.1"],
)

VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"
VLLM_WS_PATH = os.environ.get("VLLM_WS_PATH", "/v1/realtime")
PROXY_API_KEY = os.environ.get("PROXY_API_KEY")
STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT", "60"))
ALLOW_OPEN_PROXY = os.environ.get("ALLOW_OPEN_PROXY", "false").lower() in {"1", "true", "yes"}

logger = logging.getLogger("vllm-proxy")
logger.setLevel(logging.INFO)


def log_info(request: Request, msg: str, **kwargs: Any) -> None:
    rid = getattr(request.state, "request_id", None)
    logger.info(json.dumps({"msg": msg, "request_id": rid, **kwargs}))


def normalize_path(path: str) -> str:
    path = re.sub(r"/\d+(/|$)", r"/:id\\1", path)
    path = re.sub(r"/[0-9a-fA-F]{8,}(/|$)", r"/:id\\1", path)
    return path


async def _check_vllm_once() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{VLLM_URL}/v1/models")
            return response.status_code == 200
    except Exception:
        return False


async def wait_for_vllm(timeout_s: int = STARTUP_TIMEOUT) -> bool:
    for _ in range(timeout_s):
        if await _check_vllm_once():
            return True
        await asyncio.sleep(1)
    return False


async def _read_nvidia_smi() -> List[Dict[str, Any]]:
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    lines = out.decode().strip().splitlines()
    results: List[Dict[str, Any]] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            idx, util, mem_used, mem_total = parts[:4]
            results.append(
                {
                    "index": idx,
                    "util": float(util),
                    "mem_used": int(mem_used) * 1024 * 1024,
                    "mem_total": int(mem_total) * 1024 * 1024,
                }
            )
    return results


async def _gpu_polling_task(interval: float = 5.0) -> None:
    if not NVML_AVAILABLE:
        while True:
            try:
                stats = await _read_nvidia_smi()
                for stat in stats:
                    GPU_UTIL.labels(gpu_index=stat["index"]).set(stat["util"])
                    GPU_MEM_USED.labels(gpu_index=stat["index"]).set(stat["mem_used"])
                    GPU_MEM_TOTAL.labels(gpu_index=stat["index"]).set(stat["mem_total"])
            except Exception:
                logger.exception("gpu polling (nvidia-smi) failed")
            await asyncio.sleep(interval)
    else:
        handle_count = pynvml.nvmlDeviceGetCount()  # type: ignore[union-attr]
        while True:
            try:
                for i in range(handle_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # type: ignore[union-attr]
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # type: ignore[union-attr]
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[union-attr]
                    GPU_UTIL.labels(gpu_index=str(i)).set(float(util))
                    GPU_MEM_USED.labels(gpu_index=str(i)).set(int(mem_info.used))
                    GPU_MEM_TOTAL.labels(gpu_index=str(i)).set(int(mem_info.total))
            except Exception:
                logger.exception("gpu polling (nvml) failed")
            await asyncio.sleep(interval)


@app.on_event("startup")
async def startup() -> None:
    app.state.gpu_task = asyncio.create_task(_gpu_polling_task())
    ready = await wait_for_vllm()
    if not ready:
        logger.warning("vLLM did not report ready within startup timeout")


@app.on_event("shutdown")
async def shutdown() -> None:
    task = getattr(app.state, "gpu_task", None)
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _auth_guard(req_headers: Dict[str, str]) -> None:
    if PROXY_API_KEY:
        token = req_headers.get("authorization") or req_headers.get("x-api-key")
        if not token:
            raise HTTPException(status_code=401, detail="missing auth")
        if token.lower().startswith("bearer "):
            token = token.split(None, 1)[1]
        if token != PROXY_API_KEY:
            raise HTTPException(status_code=403, detail="forbidden")
    elif not ALLOW_OPEN_PROXY:
        raise HTTPException(
            status_code=403,
            detail="proxy locked: set PROXY_API_KEY or ALLOW_OPEN_PROXY=true",
        )


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request) -> Dict[str, str]:
    return {"status": "ok", "message": "vLLM proxy alive"}


@app.get("/health")
@limiter.limit("20/minute")
async def health(request: Request) -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
@limiter.limit("20/minute")
async def ready(request: Request) -> JSONResponse:
    ok = await _check_vllm_once()
    return JSONResponse({"ready": ok}, status_code=(200 if ok else 503))


@app.get("/metrics")
async def metrics(request: Request) -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/internal/deploy")
async def internal_deploy(request: Request) -> Dict[str, Any]:
    token = request.headers.get("authorization") or request.headers.get("x-api-key")
    if not token:
        raise HTTPException(status_code=401, detail="missing auth")
    if token.lower().startswith("bearer "):
        token = token.split(None, 1)[1]
    if token != PROXY_API_KEY:
        raise HTTPException(status_code=403, detail="forbidden")

    body: Dict[str, Any]
    if request.content_type == "application/json":
        body = await request.json()
    else:
        body = {}
    service = body.get("service", "vllm-proxy")
    env = body.get("env", os.environ.get("ENV", "prod"))
    DEPLOYS.labels(service=service, env=env).inc()
    log_info(request, "Deploy event recorded", service=service, env=env)
    return {"ok": True, "service": service, "env": env}


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@limiter.limit("120/minute")
async def proxy_v1(path: str, request: Request):
    _auth_guard(request.headers)
    target_url = f"{VLLM_URL}/v1/{path}"
    start = time.time()
    method = request.method
    path_template = normalize_path(request.url.path)

    with REQ_LATENCY.labels(method, path_template).time():
        body = await request.body()
        params = dict(request.query_params)
        out_headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower()
            not in {
                "host",
                "connection",
                "content-length",
                "transfer-encoding",
                "upgrade",
            }
        }
        vllm_api_key = os.environ.get("VLLM_API_KEY")
        if vllm_api_key:
            out_headers["authorization"] = f"Bearer {vllm_api_key}"

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                resp = await client.request(
                    request.method,
                    target_url,
                    headers=out_headers,
                    content=body or None,
                    params=params,
                    stream=True,
                )
            except httpx.RequestError as exc:
                REQ_COUNT.labels(method, "502", path_template).inc()
                log_info(
                    request,
                    "Upstream request error",
                    error=str(exc),
                    method=method,
                    path=path,
                )
                raise HTTPException(status_code=502, detail=f"upstream error: {exc}")

            status = resp.status_code
            REQ_COUNT.labels(method, str(status), path_template).inc()
            resp_headers = {
                key: value
                for key, value in resp.headers.items()
                if key.lower()
                not in {
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "transfer-encoding",
                    "upgrade",
                }
            }
            log_info(
                request,
                "Proxied request",
                method=method,
                path=path,
                status=status,
                latency=time.time() - start,
            )
            return StreamingResponse(
                resp.aiter_bytes(),
                status_code=status,
                headers=resp_headers,
                media_type=resp.headers.get("content-type"),
            )


@app.websocket("/ws/{path:path}")
async def websocket_bridge(ws: WebSocket, path: str) -> None:
    await ws.accept()

    qs_token = ws.query_params.get("token")

    if PROXY_API_KEY:
        token = qs_token
        if not token:
            try:
                auth_msg = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
                try:
                    obj = json.loads(auth_msg)
                    token = obj.get("token") or obj.get("auth") or None
                except Exception:
                    token = auth_msg
            except asyncio.TimeoutError:
                await ws.close(code=1008)
                return

        if token is None:
            await ws.close(code=1008)
            return

        if token.lower().startswith("bearer "):
            token = token.split(None, 1)[1]

        if token != PROXY_API_KEY:
            await ws.close(code=1008)
            return
    elif not ALLOW_OPEN_PROXY:
        await ws.close(code=1008)
        return

    upstream_ws_url = f"ws://{VLLM_HOST}:{VLLM_PORT}{VLLM_WS_PATH.rstrip('/')}/{path}"
    upstream_headers = {}
    vllm_api_key = os.environ.get("VLLM_API_KEY")
    if vllm_api_key:
        upstream_headers["Authorization"] = f"Bearer {vllm_api_key}"

    import websockets

    try:
        async with websockets.connect(upstream_ws_url, extra_headers=upstream_headers) as upstream:
            async def client_to_upstream() -> None:
                try:
                    while True:
                        msg = await ws.receive_text()
                        await upstream.send(msg)
                except WebSocketDisconnect:
                    logger.info("WebSocket client disconnected")
                except Exception as exc:  # pragma: no cover - just logging
                    logger.error(f"Error in client_to_upstream: {exc}")

            async def upstream_to_client() -> None:
                try:
                    async for msg in upstream:
                        if isinstance(msg, bytes):
                            await ws.send_bytes(msg)
                        else:
                            await ws.send_text(msg)
                except Exception as exc:  # pragma: no cover - just logging
                    logger.error(f"Error in upstream_to_client: {exc}")

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(client_to_upstream()),
                    asyncio.create_task(upstream_to_client()),
                ],
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in pending:
                task.cancel()
    except Exception:
        logger.exception("WebSocket bridge failed")
        try:
            await ws.close(code=1011)
        except Exception:  # pragma: no cover - best effort
            pass


@app.post("/sse/v1/{path:path}")
@limiter.limit("60/minute")
async def sse_proxy(path: str, request: Request):
    _auth_guard(request.headers)
    target_url = f"{VLLM_URL}/v1/{path}"

    vllm_api_key = os.environ.get("VLLM_API_KEY")
    headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}

    body = await request.body()

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.stream("POST", target_url, headers=headers, content=body)

        async def event_generator() -> Iterable[Dict[str, str]]:
            async for chunk in resp.aiter_lines():
                if not chunk:
                    continue
                yield {"event": "delta", "data": chunk}

        from sse_starlette.sse import EventSourceResponse

        return EventSourceResponse(event_generator())
