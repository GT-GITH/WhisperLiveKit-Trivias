"""Generieke on-prem LLM-backend.

Client voor een lokaal/on-prem OpenAI-compatibel chat-endpoint (Ollama,
vLLM, llama.cpp-server, of een endpoint dat een klant al zelf draait).
Vrijwel elke on-prem LLM-runtime biedt deze API aan, dus één generieke
client dekt zowel "wij leveren een gedownload model" als "klant heeft er al
een" -- geen per-tool-integratie nodig. Zelfde architectuurprincipe als de
bestaande pluggable ASR-backend-keuze (--backend), nu voor LLM's.

Nooit een cloud-default: base_url moet altijd expliciet naar een lokaal/
on-prem endpoint wijzen (dit project is on-prem-only, zie CLAUDE.md).
`build_llm_backend()` geeft None terug als er niets geconfigureerd is --
elke feature die dit gebruikt moet daarmee om kunnen gaan (fail-safe,
nooit een harde eis).
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("whisperlivekit.llm_backend")


class LLMBackend:
    """Dunne wrapper om één blocking chat-call. Geen state, geen retries,
    geen fail-safe-logica hier -- dat is de verantwoordelijkheid van de
    AANROEPER (bv. gehoorverslag.classify_segments()), zodat deze klasse
    voor elk toekomstig LLM-gebruik in dit project herbruikbaar blijft
    zonder impliciete aannames over hoe fouten afgehandeld moeten worden."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, timeout: float = 90.0):
        from openai import OpenAI  # lazy import, zelfde stijl als de bestaande openai-api ASR-backend

        self.base_url = base_url
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout)

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        """Eén chat-completion-call, retourneert de ruwe modeltekst."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


def build_llm_backend(args: Any) -> Optional[LLMBackend]:
    """Bouw de LLM-backend vanaf CLI-args, of None als er niets is
    geconfigureerd (--llm-backend-url niet gezet). Faalt niet hard bij een
    ontbrekende `openai`-dependency of onbereikbaar endpoint tijdens het
    bouwen zelf -- dat gebeurt pas bij een daadwerkelijke chat()-aanroep,
    en is dan de verantwoordelijkheid van de aanroeper om af te vangen."""
    base_url = getattr(args, "llm_backend_url", None)
    if not base_url:
        return None
    model = getattr(args, "llm_model", None)
    if not model:
        logger.warning(
            "[LLM] --llm-backend-url is gezet maar --llm-model niet -- "
            "LLM-backend wordt NIET gebouwd (fail-safe, geen gok op een modelnaam)."
        )
        return None
    api_key = getattr(args, "llm_api_key", None)
    logger.info(f"[LLM] backend geconfigureerd: base_url={base_url} model={model}")
    return LLMBackend(base_url=base_url, model=model, api_key=api_key)
