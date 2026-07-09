import pytest


@pytest.mark.integration
def test_pipeline_answers_medical_query():
    from src.core.pipeline import run_pipeline
    r = run_pipeline("What are the symptoms of type 2 diabetes?", 0.0, 0)
    assert r.status in ("answered", "handoff", "blocked_output")
    assert len(r.final_answer) > 0


@pytest.mark.integration
def test_pipeline_blocks_off_topic():
    from src.core.pipeline import run_pipeline
    r = run_pipeline("What is the weather today?", 0.0, 0)
    assert r.status == "blocked_input"
    assert r.blocked_reason == "off_topic"
    assert r.sources == []


@pytest.mark.integration
def test_pipeline_redacts_pii():
    from src.core.pipeline import run_pipeline
    r = run_pipeline("My PAN is ABCDE1234F, what causes hypertension?", 0.0, 0)
    assert r.pii_redacted_input is True


@pytest.mark.integration
def test_pipeline_triggers_handoff_on_repetition():
    from src.core.pipeline import run_pipeline
    r = run_pipeline("What causes migraines?", 0.0, 2)
    assert r.status == "handoff"
    assert r.handoff_trigger == "repetition"