"""
One sentence job: Detects and masks PII (including Indian PAN, Aadhaar, phone) in any text using Presidio — runs on both 
query input (Phase 2) and LLM output (Phase 4b).
Understand before writing:
Presidio combines two detection approaches: regex recognizers (fast, precise for structured formats like PAN's ABCDE1234F pattern) 
and spaCy NER (catches unstructured PII like person names that don't follow a fixed format). The built-in Presidio recognizers handle 
US/EU formats. You register three custom PatternRecognizer objects for Indian formats. The singleton pattern here is important — Presidio 
loads spaCy's model at construction, which takes ~2 seconds. One instance, reused for every request.

"""
"""
One sentence job: Detects and masks PII (including Indian PAN, Aadhaar, phone) in any text using Presidio — runs on both 
query input (Phase 2) and LLM output (Phase 4b).
"""

"""
Detects and masks PII (including Indian PAN, Aadhaar, phone) in any text
using Presidio — runs on both query input (Phase 2) and LLM output (Phase 4b).
"""

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from src.core.logging_config import logger


MEDICAL_TERM_WHITELIST = {
    "folate", "megaloblastic", "metformin", "hyperkalemia",
    "hypokalemia", "hypocalcemia", "hypernatremia", "hyponatremia",
    "streptococcus", "pneumoniae", "pneumococcus", "staphylococcus",
    "salmonella", "helicobacter", "pylori", "warfarin", "heparin",
    "insulin", "glucagon", "cortisol", "aldosterone", "epinephrine",
    "dopamine", "serotonin", "melatonin", "thyroxine", "calcitonin",
    "erythropoietin", "fibrinogen", "albumin", "creatinine",
    "hemoglobin", "hematocrit", "leukocyte", "erythrocyte",
    "thrombocyte", "macrophage", "lymphocyte", "neutrophil",
    "eosinophil", "basophil", "monocyte",
}


class PIIScrubber:

    def __init__(self) -> None:
        self.analyzer   = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self._register_indian_recognizers()

    def _register_indian_recognizers(self) -> None:
        pan = PatternRecognizer(
            supported_entity="IN_PAN",
            patterns=[Pattern("pan", r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b", 0.9)],
            context=["pan", "permanent account number"],
        )
        aadhaar = PatternRecognizer(
            supported_entity="IN_AADHAAR",
            patterns=[Pattern("aadhaar", r"\b\d{4}\s?\d{4}\s?\d{4}\b", 0.85)],
            context=["aadhaar", "uid"],
        )
        phone = PatternRecognizer(
            supported_entity="IN_PHONE",
            patterns=[Pattern("in_phone", r"\b(?:\+91[\-\s]?|0)?[6-9]\d{9}\b", 0.8)],
            context=["phone", "mobile", "contact"],
        )
        for r in (pan, aadhaar, phone):
            self.analyzer.registry.add_recognizer(r)
        logger.info("Registered Indian PII recognizers (PAN, Aadhaar, Phone)")

    def scrub(self, text: str) -> dict:
        entities = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
            "CREDIT_CARD", "LOCATION",
            "IN_PAN", "IN_AADHAAR", "IN_PHONE",
        ]
        results = self.analyzer.analyze(
            text=text, entities=entities, language="en"
        )

        if not results:
            return {
                "clean_text":     text,
                "redacted":       False,
                "entities_found": [],
            }

        # Filter false positives — medical terms misclassified as PERSON
        filtered_results = []
        for result in results:
            detected_text = text[result.start:result.end].lower().strip()
            if detected_text in MEDICAL_TERM_WHITELIST:
                logger.debug(
                    f"Whitelist override: '{detected_text}' kept as medical term"
                )
                continue
            filtered_results.append(result)

        if not filtered_results:
            return {
                "clean_text":     text,
                "redacted":       False,
                "entities_found": [],
            }

        operators = {
            "PERSON":        OperatorConfig("replace", {"new_value": "[NAME_REDACTED]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL_REDACTED]"}),
            "PHONE_NUMBER":  OperatorConfig("replace", {"new_value": "[PHONE_REDACTED]"}),
            "CREDIT_CARD":   OperatorConfig("replace", {"new_value": "[CARD_REDACTED]"}),
            "LOCATION":      OperatorConfig("replace", {"new_value": "[LOCATION_REDACTED]"}),
            "IN_PAN":        OperatorConfig("replace", {"new_value": "[PAN_REDACTED]"}),
            "IN_AADHAAR":    OperatorConfig("replace", {"new_value": "[AADHAAR_REDACTED]"}),
            "IN_PHONE":      OperatorConfig("replace", {"new_value": "[PHONE_REDACTED]"}),
        }

        anonymized     = self.anonymizer.anonymize(text, filtered_results, operators)
        entities_found = sorted({r.entity_type for r in filtered_results})

        if entities_found:
            logger.warning(f"PII redacted: {entities_found}")

        return {
            "clean_text":     anonymized.text,
            "redacted":       bool(entities_found),
            "entities_found": entities_found,
        }


_instance: PIIScrubber | None = None


def get_pii_scrubber() -> PIIScrubber:
    global _instance
    if _instance is None:
        _instance = PIIScrubber()
    return _instance