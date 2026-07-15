from dataclasses import dataclass, field
from src.guardrails.pii_scrubber    import get_pii_scrubber
from src.guardrails.input_guard     import get_input_guardrail
from src.guardrails.output_guard    import get_output_guardrail
from src.guardrails.pii_scan_output import scan_output_for_pii
from src.guardrails.human_handoff   import get_satisfaction_tracker
from src.retrieval.retriever        import get_retriever
from src.retrieval.reranker         import get_reranker
from src.generation.prompt_builder  import build_prompt
from src.generation.llm_client      import get_llm_client

CANNED_FALLBACK  = (
    "I can't help with that. Please rephrase or "
    "consult a healthcare professional."
)
GROUNDING_FAIL   = (
    "I'm not confident enough to answer that accurately. "
    "Please rephrase or consult a professional."
)


@dataclass
class PipelineResult:
    final_answer:        str
    status:              str        # answered | blocked_input | blocked_output | handoff
    blocked_reason:      str | None = None
    pii_redacted_input:  bool       = False
    pii_redacted_output: bool       = False
    grounding_score:     float | None = None
    handoff_trigger:     str | None = None
    sources:             list[dict]  = field(default_factory=list)


def run_pipeline(
    query: str,
    session_frustration_score: float = 0.0,
    repeated_question_count:   int   = 0,
) -> PipelineResult:

    # ── Phase 2a: PII scrub on input ─────────────────────────────────
    scrub       = get_pii_scrubber().scrub(query)
    clean_query = scrub["clean_text"]

    # ── Phase 2b: Input guardrail ─────────────────────────────────────
    guard = get_input_guardrail().check(clean_query)
    if not guard["passed"]:
        return PipelineResult(
            final_answer       = CANNED_FALLBACK,
            status             = "blocked_input",
            blocked_reason     = guard["reason"],
            pii_redacted_input = scrub["redacted"],
        )

    # ── Phase 3: Retrieve → rerank → generate ────────────────────────
    # candidates = get_retriever().retrieve(clean_query)
    
    candidates = get_retriever().retrieve_with_hyde(clean_query)
    top_chunks = get_reranker().rerank(clean_query, candidates)
    prompt     = build_prompt(clean_query, top_chunks)
    raw_answer = get_llm_client().generate(prompt)

    # ── Phase 4a: Output guardrail ────────────────────────────────────
    ctx_texts    = [c["text"] for c in top_chunks]
    out_check    = get_output_guardrail().check(raw_answer, ctx_texts)
    if not out_check["passed"]:
        return PipelineResult(
            final_answer       = GROUNDING_FAIL,
            status             = "blocked_output",
            blocked_reason     = out_check["reason"],
            pii_redacted_input = scrub["redacted"],
            grounding_score    = out_check.get("grounding_score"),
        )

    # ── Phase 4b: Output PII scan ─────────────────────────────────────
    pii_out    = scan_output_for_pii(raw_answer)
    safe_answer = pii_out["final_text"]

    # ── Phase 5: Satisfaction / handoff ──────────────────────────────
    tracker  = get_satisfaction_tracker()
    explicit = tracker.is_explicit_handoff_request(query)
    decision = tracker.should_handoff(
        session_frustration_score,
        repeated_question_count,
        explicit,
    )

    sources = [
        {"source": c["source"], "doc_id": c["doc_id"]}
        for c in top_chunks
    ]

    if decision["handoff"]:
        return PipelineResult(
            final_answer        = safe_answer,
            status              = "handoff",
            handoff_trigger     = decision["trigger"],
            pii_redacted_input  = scrub["redacted"],
            pii_redacted_output = pii_out["pii_leaked"],
            grounding_score     = out_check.get("grounding_score"),
            sources             = sources,
        )

    return PipelineResult(
        final_answer        = safe_answer,
        status              = "answered",
        pii_redacted_input  = scrub["redacted"],
        pii_redacted_output = pii_out["pii_leaked"],
        grounding_score     = out_check.get("grounding_score"),
        sources             = sources,
    )