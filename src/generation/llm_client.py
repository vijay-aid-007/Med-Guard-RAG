"""
LLM client - Groq as primary, Ollama as fallback.
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed
from src.core.config import settings
from src.core.logging_config import logger


class LLMClient:

    def __init__(self):
        self.provider = settings.llm_provider.lower()
        logger.info(f"LLM provider: {self.provider} | model: {settings.llm_model}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(4))
    def generate(self, prompt: str) -> str:
        if self.provider == "groq":
            return self._groq(prompt)
        return self._ollama(prompt)

    def _groq(self, prompt: str) -> str:
        try:
            from groq import Groq
            client = Groq(api_key=settings.groq_api_key)
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"Groq response: {answer[:80]}")
            return answer
        except Exception as e:
            logger.error(f"Groq call failed: {e} - falling back to Ollama")
            return self._ollama(prompt)

    def _ollama(self, prompt: str) -> str:
        try:
            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": "phi3:mini",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": settings.llm_temperature,
                            "num_predict": settings.llm_max_tokens,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json() or {}
                answer = data.get("response") or ""
                answer = answer.strip()
                logger.info(f"Ollama response: {answer[:80]}")
                return answer
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise


_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _instance
    if _instance is None:
        _instance = LLMClient()
    return _instance
