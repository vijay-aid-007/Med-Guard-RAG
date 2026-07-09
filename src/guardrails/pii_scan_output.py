"""

"""




from src.guardrails.pii_scrubber import get_pii_scrubber
from src.core.logging_config import logger


def scan_output_for_pii(answer: str) -> dict:
    result = get_pii_scrubber().scrub(answer)
    if result["redacted"]:
        logger.warning(
            f"Output PII scan caught: {result['entities_found']}"
        )
    return {
        "final_text":    result["clean_text"],
        "pii_leaked":    result["redacted"],
        "entities_found": result["entities_found"],
    }