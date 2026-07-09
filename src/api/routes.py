import time
import httpx
from fastapi import APIRouter, HTTPException

from src.api.schemas  import QueryRequest, QueryResponse, HealthResponse, SourceCitation
from src.api.metrics  import (
    REQUEST_COUNT, GUARDRAIL_BLOCKS,
    PII_REDACTIONS, HANDOFFS, PIPELINE_LATENCY,
)
from src.core.pipeline import run_pipeline
from src.core.session  import get_session_manager
from src.core.config   import settings
from src.core.logging_config import logger
from src.guardrails.human_handoff import get_satisfaction_tracker

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    REQUEST_COUNT.inc()
    start = time.time()

    session_mgr   = get_session_manager()
    state         = session_mgr.get_state(request.session_id)
    repeat_count  = session_mgr.count_repeats(request.session_id, request.query)

    try:
        result = run_pipeline(
            query                    = request.query,
            session_frustration_score = state["frustration_score"],
            repeated_question_count  = repeat_count,
        )
    except Exception as e:
        logger.exception(f"Pipeline error for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal pipeline error")

    elapsed = time.time() - start
    PIPELINE_LATENCY.labels(phase="total").observe(elapsed)

    # ── Record metrics ────────────────────────────────────────────────
    if result.status == "blocked_input":
        GUARDRAIL_BLOCKS.labels(
            phase="input", reason=result.blocked_reason
        ).inc()
    elif result.status == "blocked_output":
        GUARDRAIL_BLOCKS.labels(
            phase="output", reason=result.blocked_reason
        ).inc()
    if result.pii_redacted_input:
        PII_REDACTIONS.labels(phase="input").inc()
    if result.pii_redacted_output:
        PII_REDACTIONS.labels(phase="output").inc()
    if result.status == "handoff":
        HANDOFFS.labels(trigger=result.handoff_trigger).inc()

    # ── Update session for next turn ──────────────────────────────────
    tracker = get_satisfaction_tracker()
    delta   = tracker.score_turn(request.query)
    session_mgr.update_state(request.session_id, delta, request.query)

    return QueryResponse(
        answer              = result.final_answer,
        status              = result.status,
        blocked_reason      = result.blocked_reason,
        pii_redacted_input  = result.pii_redacted_input,
        pii_redacted_output = result.pii_redacted_output,
        grounding_score     = result.grounding_score,
        handoff_triggered   = result.status == "handoff",
        handoff_trigger     = result.handoff_trigger,
        sources             = [SourceCitation(**s) for s in result.sources],
    )


@router.get("/health", response_model=HealthResponse)
def health_check():
    faiss_ok = True
    try:
        from src.retrieval.retriever import get_retriever
        get_retriever()
    except Exception:
        faiss_ok = False

    llm_ok = True
    try:
        with httpx.Client(timeout=3.0) as client:
            r      = client.get(f"{settings.ollama_base_url}/api/tags")
            llm_ok = r.status_code == 200
    except Exception:
        llm_ok = False

    status = "healthy" if (faiss_ok and llm_ok) else "degraded"
    return HealthResponse(
        status       = status,
        faiss_loaded = faiss_ok,
        llm_reachable = llm_ok,
    )