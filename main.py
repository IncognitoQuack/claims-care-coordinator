"""
Claims & Care Coordinator — API Server

FastAPI application with:
- SSE (Server-Sent Events) for real-time agent activity streaming
- REST endpoints for claim processing
- Static file serving for the frontend
- Sample data loading
"""

import os
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from models.schemas import EHRData, ClaimRequest, AgentSource
from agents.coordinator import ClaimsCoordinator
from memory.ledger import MedicalNecessityLedger


# ── Config ──────────────────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DATA_DIR = Path(__file__).parent / "data"
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        print("\n⚠️  WARNING: ANTHROPIC_API_KEY not set!")
        print("   Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        print("   The app will still start but API calls will fail.\n")
    else:
        print("\n✅ ANTHROPIC_API_KEY detected. Agents are ready.\n")
    yield


app = FastAPI(
    title="Claims & Care Coordinator",
    description="Multi-agent prior authorization system with shared memory",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Sample Data ─────────────────────────────────────────────────────
def load_sample_ehr(name: str) -> dict:
    path = DATA_DIR / f"ehr_{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Sample EHR '{name}' not found")
    with open(path) as f:
        return json.load(f)


def load_policy() -> str:
    path = DATA_DIR / "sample_policy.txt"
    with open(path) as f:
        return f.read()


# ── API Routes ──────────────────────────────────────────────────────
@app.get("/api/samples")
async def list_samples():
    """List available sample EHR scenarios."""
    samples = []
    for p in sorted(DATA_DIR.glob("ehr_*.json")):
        with open(p) as f:
            data = json.load(f)
        samples.append({
            "id": p.stem.replace("ehr_", ""),
            "patient_name": data["patient_name"],
            "chief_complaint": data["chief_complaint"],
            "diagnosis_codes": data.get("diagnosis_codes", []),
        })
    return {"samples": samples, "policy_available": (DATA_DIR / "sample_policy.txt").exists()}


@app.get("/api/sample/{sample_id}")
async def get_sample(sample_id: str):
    """Get a specific sample EHR + full policy text."""
    ehr = load_sample_ehr(sample_id)
    policy = load_policy()
    return {"ehr": ehr, "policy_text": policy, "policy_length": len(policy)}


@app.get("/api/policy")
async def get_policy():
    """Get the full policy document text."""
    return {"text": load_policy()}


@app.post("/api/process")
async def process_claim_stream(request: ClaimRequest):
    """
    Process a claim and stream results via SSE.
    
    This is the main endpoint. It:
    1. Starts the Clinical Agent
    2. Streams all ledger events as SSE
    3. The Policy Agent reads the ledger and adapts
    4. Returns final determination
    """
    if not API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured")

    coordinator = ClaimsCoordinator(api_key=API_KEY)

    # Subscribe to ledger events BEFORE processing starts
    event_queue = coordinator.ledger.subscribe()

    async def event_stream():
        # Start processing in background
        result_holder = {"result": None, "error": None}

        async def run_processing():
            try:
                result = await coordinator.process_claim(
                    ehr=request.ehr_data,
                    policy_text=request.policy_document,
                    plan_name=request.plan_name,
                )
                result_holder["result"] = result
            except Exception as e:
                result_holder["error"] = str(e)
                # Write error to ledger so the stream gets it
                await coordinator.ledger.write(
                    source=AgentSource.SYSTEM,
                    event_type="ERROR",
                    message=f"Processing error: {str(e)}",
                    tags=["ERROR"],
                )

        task = asyncio.create_task(run_processing())

        # Stream events as they come in
        while True:
            try:
                entry = await asyncio.wait_for(event_queue.get(), timeout=120.0)
                event_data = {
                    "id": entry.id,
                    "timestamp": entry.timestamp.isoformat(),
                    "source": entry.source.value,
                    "event_type": entry.event_type,
                    "message": entry.message,
                    "data": entry.data,
                    "tags": entry.tags,
                    "severity": entry.severity.value,
                }
                yield f"data: {json.dumps(event_data)}\n\n"

                # Check if processing is complete
                if entry.event_type == "PROCESS_COMPLETE":
                    # Send the final result
                    await asyncio.sleep(0.1)
                    if result_holder["result"]:
                        # Serialize datetime objects
                        result = result_holder["result"]
                        if "ledger" in result:
                            for le in result["ledger"]:
                                if hasattr(le.get("timestamp"), "isoformat"):
                                    le["timestamp"] = le["timestamp"].isoformat()
                        yield f"event: result\ndata: {json.dumps(result, default=str)}\n\n"
                    break

                if entry.event_type == "ERROR":
                    yield f"event: error\ndata: {json.dumps({'error': entry.message})}\n\n"
                    break

            except asyncio.TimeoutError:
                yield f"event: keepalive\ndata: {json.dumps({'status': 'processing'})}\n\n"

            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                break

        coordinator.ledger.unsubscribe(event_queue)
        yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/process-sync")
async def process_claim_sync(request: ClaimRequest):
    """Synchronous version — processes and returns complete result."""
    if not API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured")

    coordinator = ClaimsCoordinator(api_key=API_KEY)
    try:
        result = await coordinator.process_claim(
            ehr=request.ehr_data,
            policy_text=request.policy_document,
            plan_name=request.plan_name,
        )
        return JSONResponse(content=json.loads(json.dumps(result, default=str)))
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")


# ── Frontend ────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend not found. Ensure static/index.html exists.</h1>")
    with open(index_path) as f:
        return HTMLResponse(f.read())


# Mount static files (CSS, JS, etc.)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("╔══════════════════════════════════════════════╗")
    print("║   Claims & Care Coordinator                 ║")
    print("║   Multi-Agent Prior Authorization System     ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
