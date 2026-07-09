"""
One sentence job: Blocks jailbreaks, harmful requests, and off-topic queries before they ever reach retrieval or the LLM.
Understand before writing:
Three independent checks run in priority order — jailbreak first (cheapest, regex), harmful second (regex), domain relevance 
last (needs an embedding model call, so runs only if the first two pass). Domain relevance uses DOMAIN_ANCHORS — a list of example
 medical questions. Your query embedding is compared against all of them; if the max similarity is below the threshold, the query is off-topic. 
 This is a semantic check, not a keyword check — "what makes blood sugar rise" passes even without the word "diabetes."
"""



import re
from sentence_transformers import SentenceTransformer, util
from src.core.config import settings
from src.core.logging_config import logger

DOMAIN_ANCHORS = [
    # ── Symptoms & Presentation ──────────────────────────────────────────
    "What are the symptoms of a disease?",
    "What are the early warning signs of this condition?",
    "How is this condition diagnosed?",
    "What are the clinical features of this illness?",
    "What causes this medical condition?",
    "What are the presenting complaints of this disease?",
    "How does this disease present in elderly patients?",
    "What are the signs and symptoms of infection?",
    "What are the red flag symptoms I should not ignore?",
    "How do symptoms differ between acute and chronic disease?",

    # ── Pharmacology & Drugs ─────────────────────────────────────────────
    "Which drug is used to treat this condition?",
    "What is the mechanism of action of this medication?",
    "What are the side effects of this drug?",
    "Which electrolyte abnormality is caused by this drug?",
    "What is the first-line treatment for this disease?",
    "How does this drug work in hypertension?",
    "What is the pharmacological effect of beta-blockers?",
    "Which medication is contraindicated in this condition?",
    "What is the recommended dosage of this medication?",
    "What are the drug interactions I should be aware of?",
    "Which antibiotic is used for this bacterial infection?",
    "What is the difference between ACE inhibitors and ARBs?",
    "How do statins reduce cardiovascular risk?",
    "What is the role of insulin in diabetes management?",
    "Which drugs are nephrotoxic and should be avoided in kidney disease?",
    "What is the half-life of this medication?",
    "How does this drug affect blood pressure?",
    "What is the antidote for this drug overdose?",
    "Which medications require therapeutic drug monitoring?",
    "What are the contraindications of NSAIDs?",

    # ── Vitamins, Nutrition & Deficiencies ───────────────────────────────
    "Which vitamin deficiency causes this condition?",
    "What are the signs of nutritional deficiency?",
    "What is the treatment for vitamin B12 deficiency?",
    "What causes megaloblastic anemia?",
    "What is the role of vitamin D in bone health?",
    "Which mineral deficiency causes muscle cramps?",
    "What are the dietary sources of iron?",
    "What causes folate deficiency anemia?",
    "How is scurvy related to vitamin C deficiency?",
    "What are the symptoms of zinc deficiency?",

    # ── Pathophysiology & Mechanisms ─────────────────────────────────────
    "What is the pathophysiology of this disease?",
    "How does inflammation contribute to this condition?",
    "What is the underlying mechanism of insulin resistance?",
    "How does heart failure develop?",
    "What is the role of the renin-angiotensin system in hypertension?",
    "How does chronic kidney disease progress?",
    "What is the pathogenesis of atherosclerosis?",
    "How do autoimmune diseases damage tissues?",
    "What is the mechanism of septic shock?",
    "How does a blood clot form in deep vein thrombosis?",

    # ── Investigations & Lab Results ─────────────────────────────────────
    "What does this lab result indicate?",
    "What is the normal range for this blood test?",
    "Which electrolyte is affected by ACE inhibitors?",
    "What does an elevated creatinine level mean?",
    "How is HbA1c used to monitor diabetes?",
    "What does a high white blood cell count indicate?",
    "When should a chest X-ray be ordered?",
    "What is the significance of troponin levels in chest pain?",
    "How is thyroid function tested?",
    "What does a positive ANA test mean?",
    "What are the causes of elevated liver enzymes?",
    "How is anemia classified based on blood tests?",

    # ── Procedures & Interventions ───────────────────────────────────────
    "Explain this medical procedure.",
    "What are the risks of this surgical procedure?",
    "How is a lumbar puncture performed?",
    "What is the indication for a coronary angiogram?",
    "How is dialysis used in kidney failure?",
    "What is the difference between MRI and CT scan?",
    "When is a biopsy indicated?",
    "How is mechanical ventilation managed in ICU?",
    "What is the procedure for cardiopulmonary resuscitation?",
    "How is a central venous catheter inserted?",

    # ── Chronic Disease Management ───────────────────────────────────────
    "Is this drug safe to take with another drug?",
    "What are the complications of this condition?",
    "How is type 2 diabetes managed long-term?",
    "What lifestyle changes help control hypertension?",
    "How is chronic obstructive pulmonary disease managed?",
    "What is the management of heart failure with reduced ejection fraction?",
    "How should asthma be managed in adults?",
    "What is the target blood pressure in diabetic patients?",
    "How is rheumatoid arthritis treated?",
    "What is the role of physiotherapy in stroke rehabilitation?",

    # ── Emergency & Critical Care ────────────────────────────────────────
    "What is the emergency treatment for anaphylaxis?",
    "How is myocardial infarction managed acutely?",
    "What is the initial management of a stroke?",
    "How is diabetic ketoacidosis treated?",
    "What are the signs of hypovolemic shock?",
    "How is status epilepticus managed?",
    "What is the treatment for pulmonary embolism?",
    "How is acute kidney injury managed?",
    "What is the Glasgow Coma Scale used for?",
    "How is severe sepsis treated?",

    # ── Infectious Diseases ──────────────────────────────────────────────
    "What is the treatment for tuberculosis?",
    "How is HIV infection managed?",
    "What antibiotics are used for pneumonia?",
    "How is malaria diagnosed and treated?",
    "What is the difference between bacterial and viral meningitis?",
    "How is hepatitis B infection managed?",
    "What are the complications of untreated syphilis?",
    "How is COVID-19 managed in hospitalized patients?",
    "What is the mechanism of antibiotic resistance?",
    "Which vaccines are recommended for immunocompromised patients?",

    # ── Cardiology ───────────────────────────────────────────────────────
    "What are the types of heart failure?",
    "How is atrial fibrillation managed?",
    "What is the difference between STEMI and NSTEMI?",
    "How do calcium channel blockers work?",
    "What is the role of anticoagulation in cardiac disease?",
    "How is hypertension classified?",
    "What causes left ventricular hypertrophy?",
    "What is the treatment for stable angina?",

    # ── Endocrinology ────────────────────────────────────────────────────
    "What is the difference between type 1 and type 2 diabetes?",
    "How is hypothyroidism treated?",
    "What are the symptoms of Cushing syndrome?",
    "How is Addison disease managed?",
    "What causes polycystic ovary syndrome?",
    "How is hyperthyroidism treated?",
    "What is the role of cortisol in the body?",

    # ── Neurology ────────────────────────────────────────────────────────
    "What are the types of stroke?",
    "How is epilepsy managed?",
    "What is the difference between delirium and dementia?",
    "How is Parkinson disease treated?",
    "What are the signs of increased intracranial pressure?",
    "How is multiple sclerosis diagnosed?",
    "What causes peripheral neuropathy?",

    # ── Respiratory ──────────────────────────────────────────────────────
    "What is the difference between asthma and COPD?",
    "How is community-acquired pneumonia treated?",
    "What are the causes of pleural effusion?",
    "How is pulmonary fibrosis managed?",
    "What is the mechanism of bronchodilators?",
    "How is respiratory failure classified?",
]

JAILBREAK_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"you are now (in )?(dan|developer) mode",
    r"pretend (you are|to be) (an? )?(unfiltered|uncensored|jailbroken)",
    r"disregard (your|all) (safety|guidelines|rules)",
    r"act as (if you have )?no (restrictions|filters|rules)",
    r"reveal your (instructions|prompt|system message)",
]

HARMFUL_PATTERNS = [
    r"how (do|can) i (make|synthesize|create) (a )?(bomb|poison|drug)",
    r"how (much|many) .{0,20}(pills|tablets) (to|will) (kill|overdose)",
    r"lethal dose",
    r"how to (harm|hurt|kill) (myself|someone)",
]


class InputGuardrail:

    def __init__(self, similarity_threshold: float = 0.30):
        self.threshold  = similarity_threshold
        self.embed_model = SentenceTransformer(settings.embedding_model)
        self.anchor_vecs = self.embed_model.encode(
            DOMAIN_ANCHORS,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    def _jailbreak(self, text: str) -> str | None:
        low = text.lower()
        for p in JAILBREAK_PATTERNS:
            if re.search(p, low):
                return p
        return None

    def _harmful(self, text: str) -> str | None:
        low = text.lower()
        for p in HARMFUL_PATTERNS:
            if re.search(p, low):
                return p
        return None

    def _domain_score(self, text: str) -> float:
        q_vec  = self.embed_model.encode(
            text, convert_to_tensor=True, normalize_embeddings=True
        )
        scores = util.cos_sim(q_vec, self.anchor_vecs)[0]
        return float(scores.max())

    def check(self, query: str) -> dict:
        if match := self._jailbreak(query):
            logger.warning(f"Input blocked — jailbreak: {match}")
            return {"passed": False, "reason": "jailbreak_attempt"}

        if match := self._harmful(query):
            logger.warning(f"Input blocked — harmful: {match}")
            return {"passed": False, "reason": "harmful_request"}

        score = self._domain_score(query)
        if score < self.threshold:
            logger.info(f"Input blocked — off_topic (score={score:.3f})")
            return {"passed": False, "reason": "off_topic"}

        return {"passed": True, "reason": None}


_instance: InputGuardrail | None = None


def get_input_guardrail() -> InputGuardrail:
    global _instance
    if _instance is None:
        _instance = InputGuardrail()
    return _instance