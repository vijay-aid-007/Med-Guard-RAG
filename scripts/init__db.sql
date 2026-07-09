CREATE TABLE IF NOT EXISTS handoff_queue (
    id                   SERIAL PRIMARY KEY,
    session_id           VARCHAR(128) NOT NULL,
    query                TEXT NOT NULL,
    trigger_reason       VARCHAR(64) NOT NULL,
    frustration_score    NUMERIC,
    conversation_history JSONB,
    status               VARCHAR(32) DEFAULT 'pending',
    created_at           TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
    id                  SERIAL PRIMARY KEY,
    session_id          VARCHAR(128) NOT NULL,
    query               TEXT,
    pipeline_status     VARCHAR(32) NOT NULL,
    blocked_reason      VARCHAR(64),
    pii_redacted_input  BOOLEAN DEFAULT false,
    pii_redacted_output BOOLEAN DEFAULT false,
    grounding_score     NUMERIC,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_handoff_status ON handoff_queue(status);
CREATE INDEX IF NOT EXISTS idx_audit_session  ON audit_log(session_id);