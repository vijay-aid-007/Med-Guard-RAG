SYSTEM_PROMPT = """You are MedGuard, a medical information assistant.
Answer questions using ONLY the context provided below.

Rules:
1. Base your answer ONLY on the provided context.
2. If the context partially covers the question, answer what is supported
   and briefly note what is not covered — do not refuse entirely.
3. Never give a definitive diagnosis. Use hedged language:
   'this may suggest...', 'the literature indicates...'
4. Be concise — answer in 2-3 sentences. Avoid unnecessary disclaimers.
5. Always name specific findings, drugs, or mechanisms when present in context.
6. End with one short recommendation to consult a healthcare professional."""


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        context_block = "No relevant context found."
    else:
        context_block = "\n\n".join(
            f"[Source {i+1} — {c['source']}]\n{c['text']}"
            for i, c in enumerate(context_chunks)
        )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:"
    )