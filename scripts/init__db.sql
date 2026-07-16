-- ═══════════════════════════════════════════════════════════════════
-- MedGuard RAG — Database Schema
-- Run once: psql -U medguard -d medguard -f scripts/init_db.sql
-- ═══════════════════════════════════════════════════════════════════

-- ── Extensions ───────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- for text search on queries

-- ── Sessions ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id       VARCHAR(128) UNIQUE NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT now(),
    last_active_at   TIMESTAMPTZ DEFAULT now(),
    turn_count       INTEGER DEFAULT 0,
    frustration_score NUMERIC DEFAULT 0.0,
    is_active        BOOLEAN DEFAULT true
);

-- ── Audit Log — every query that enters the pipeline ─────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id                   BIGSERIAL PRIMARY KEY,
    session_id           VARCHAR(128) NOT NULL,
    query                TEXT,
    query_hash           VARCHAR(64),          -- for dedup/repeat detection
    pipeline_status      VARCHAR(32) NOT NULL, -- answered/blocked_input/blocked_output/handoff
    blocked_reason       VARCHAR(64),
    pii_redacted_input   BOOLEAN DEFAULT false,
    pii_redacted_output  BOOLEAN DEFAULT false,
    grounding_score      NUMERIC,
    latency_ms           INTEGER,              -- end-to-end response time
    llm_provider         VARCHAR(32),          -- groq/ollama
    retrieval_method     VARCHAR(32),          -- hyde/query_expansion/basic
    chunks_retrieved     INTEGER,
    chunks_reranked      INTEGER,
    created_at           TIMESTAMPTZ DEFAULT now()
);

-- ── Handoff Queue — cases routed to human agents ─────────────────────
CREATE TABLE IF NOT EXISTS handoff_queue (
    id                   BIGSERIAL PRIMARY KEY,
    session_id           VARCHAR(128) NOT NULL,
    query                TEXT NOT NULL,
    trigger_reason       VARCHAR(64) NOT NULL,  -- explicit_request/repetition/frustration
    frustration_score    NUMERIC,
    conversation_history JSONB,
    status               VARCHAR(32) DEFAULT 'pending',  -- pending/assigned/resolved/closed
    assigned_to          VARCHAR(128),          -- agent email/id
    assigned_at          TIMESTAMPTZ,
    resolved_at          TIMESTAMPTZ,
    resolution_notes     TEXT,
    created_at           TIMESTAMPTZ DEFAULT now()
);

-- ── Guardrail Events — blocked inputs/outputs ─────────────────────────
CREATE TABLE IF NOT EXISTS guardrail_events (
    id              BIGSERIAL PRIMARY KEY,
    session_id      VARCHAR(128) NOT NULL,
    guardrail_type  VARCHAR(32) NOT NULL,   -- input/output
    reason          VARCHAR(64) NOT NULL,   -- jailbreak/harmful/off_topic/toxic/low_grounding
    query           TEXT,
    matched_pattern TEXT,                   -- which regex fired
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ── Evaluation Runs — RAGAS/benchmark results over time ──────────────
CREATE TABLE IF NOT EXISTS eval_runs (
    id                  BIGSERIAL PRIMARY KEY,
    run_type            VARCHAR(32) NOT NULL,  -- benchmark/ragas
    faithfulness        NUMERIC,
    answer_relevancy    NUMERIC,
    context_precision   NUMERIC,
    context_recall      NUMERIC,
    benchmark_accuracy  NUMERIC,
    total_questions     INTEGER,
    correct_answers     INTEGER,
    embedding_model     VARCHAR(128),
    reranker_model      VARCHAR(128),
    llm_model           VARCHAR(128),
    corpus_size         INTEGER,
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ── PII Events — redaction audit trail ───────────────────────────────
CREATE TABLE IF NOT EXISTS pii_events (
    id              BIGSERIAL PRIMARY KEY,
    session_id      VARCHAR(128) NOT NULL,
    direction       VARCHAR(8) NOT NULL,     -- input/output
    entities_found  TEXT[],                  -- ['PERSON', 'EMAIL_ADDRESS']
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ── Indexes ───────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_audit_session      ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_status       ON audit_log(pipeline_status);
CREATE INDEX IF NOT EXISTS idx_audit_created      ON audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_handoff_status     ON handoff_queue(status);
CREATE INDEX IF NOT EXISTS idx_handoff_session    ON handoff_queue(session_id);
CREATE INDEX IF NOT EXISTS idx_guardrail_type     ON guardrail_events(guardrail_type);
CREATE INDEX IF NOT EXISTS idx_guardrail_reason   ON guardrail_events(reason);
CREATE INDEX IF NOT EXISTS idx_pii_session        ON pii_events(session_id);
CREATE INDEX IF NOT EXISTS idx_session_id         ON sessions(session_id);

-- ── Views — for Grafana dashboards ────────────────────────────────────
CREATE OR REPLACE VIEW v_pipeline_summary AS
SELECT
    DATE_TRUNC('hour', created_at)  AS hour,
    pipeline_status,
    COUNT(*)                         AS count,
    AVG(grounding_score)             AS avg_grounding,
    AVG(latency_ms)                  AS avg_latency_ms
FROM audit_log
GROUP BY 1, 2
ORDER BY 1 DESC;

CREATE OR REPLACE VIEW v_guardrail_summary AS
SELECT
    DATE_TRUNC('hour', created_at)  AS hour,
    guardrail_type,
    reason,
    COUNT(*)                         AS count
FROM guardrail_events
GROUP BY 1, 2, 3
ORDER BY 1 DESC;

CREATE OR REPLACE VIEW v_handoff_summary AS
SELECT
    DATE_TRUNC('day', created_at)   AS day,
    trigger_reason,
    status,
    COUNT(*)                         AS count,
    AVG(frustration_score)           AS avg_frustration
FROM handoff_queue
GROUP BY 1, 2, 3
ORDER BY 1 DESC;