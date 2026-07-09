from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "medguard_requests_total",
    "Total queries received",
)

GUARDRAIL_BLOCKS = Counter(
    "medguard_guardrail_blocks_total",
    "Queries blocked by guardrail",
    ["phase", "reason"],
)

PII_REDACTIONS = Counter(
    "medguard_pii_redactions_total",
    "PII redaction events",
    ["phase"],
)

HANDOFFS = Counter(
    "medguard_handoffs_total",
    "Human agent handoffs triggered",
    ["trigger"],
)

PIPELINE_LATENCY = Histogram(
    "medguard_pipeline_latency_seconds",
    "End-to-end pipeline latency",
    ["phase"],
)