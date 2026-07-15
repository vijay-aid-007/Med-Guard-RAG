from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.responses import PlainTextResponse
from prometheus_client import generate_latest

from src.api.routes      import router
from src.core.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MedGuard RAG — preloading models...")

    from src.retrieval.retriever        import get_retriever
    from src.retrieval.reranker         import get_reranker
    from src.guardrails.pii_scrubber    import get_pii_scrubber
    from src.guardrails.input_guard     import get_input_guardrail
    from src.guardrails.output_guard    import get_output_guardrail

    get_retriever()
    get_reranker()
    get_pii_scrubber()
    get_input_guardrail()
    get_output_guardrail()

    logger.info("All models preloaded — MedGuard RAG is ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "MedGuard RAG",
    description = "Medical QA with end-to-end guardrails",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.include_router(router, prefix="/api/v1")


@app.get("/metrics")
def metrics():
    return PlainTextResponse(
        generate_latest(), media_type="text/plain"
    )


@app.get("/")
def root():
    return {"service": "MedGuard RAG", "status": "running", "docs": "/docs"}