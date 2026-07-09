from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query:      str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., description="Client session identifier")


class SourceCitation(BaseModel):
    source: str
    doc_id: str


class QueryResponse(BaseModel):
    answer:               str
    status:               str        # answered | blocked_input | blocked_output | handoff
    blocked_reason:       str | None = None
    pii_redacted_input:   bool       = False
    pii_redacted_output:  bool       = False
    grounding_score:      float | None = None
    handoff_triggered:    bool       = False
    handoff_trigger:      str | None = None
    sources:              list[SourceCitation] = []


class HealthResponse(BaseModel):
    status:        str
    faiss_loaded:  bool
    llm_reachable: bool