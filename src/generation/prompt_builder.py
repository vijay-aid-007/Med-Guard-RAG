SYSTEM_PROMPT = """You are MedGuard, a medical information assistant.
Answer questions using ONLY the context provided below.

Rules:
1. Base your answer ONLY on the provided context.
2. If context is insufficient, say so — do not guess.
3. Never give a definitive diagnosis. Use hedged language:
   'this may suggest...', 'the literature indicates...'
4. Always recommend consulting a licensed healthcare professional."""


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