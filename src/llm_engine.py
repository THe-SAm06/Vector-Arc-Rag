"""
llm_engine.py
─────────────
Provider-agnostic LLM engine for RAG answer generation.

Supports any OpenAI-compatible API endpoint — switch providers by
changing three lines in your .env file, no code changes needed.

Recommended providers (all free, no credit card required):
──────────────────────────────────────────────────────────
  GROQ (best for this project):
    • Rate limit resets every MINUTE (not daily like Gemini)
    • Fastest inference engine in the world (LPU hardware)
    • Get key: https://console.groq.com
    • .env:
        LLM_PROVIDER=groq
        LLM_API_KEY=gsk_...
        LLM_MODEL=llama-3.3-70b-versatile

  SiliconFlow (massive free token allowance):
    • Get key: https://siliconflow.cn
    • .env:
        LLM_PROVIDER=siliconflow
        LLM_API_KEY=sk-...
        LLM_MODEL=Qwen/Qwen2.5-72B-Instruct

  OpenRouter (rotating free models):
    • Get key: https://openrouter.ai
    • .env:
        LLM_PROVIDER=openrouter
        LLM_API_KEY=sk-or-...
        LLM_MODEL=meta-llama/llama-3-8b-instruct:free

  Gemini (original, daily quota):
    • Get key: https://aistudio.google.com
    • .env:
        LLM_PROVIDER=gemini
        LLM_API_KEY=AIza...
        LLM_MODEL=gemini-2.0-flash

Provider base URLs (set automatically from LLM_PROVIDER):
─────────────────────────────────────────────────────────
  groq        → https://api.groq.com/openai/v1
  siliconflow → https://api.siliconflow.cn/v1
  openrouter  → https://openrouter.ai/api/v1
  gemini      → (uses google-generativeai SDK directly)
"""

import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Provider configuration ─────────────────────────────────────────────────────
_PROVIDER_URLS = {
    "groq":        "https://api.groq.com/openai/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "openrouter":  "https://openrouter.ai/api/v1",
    "github":      "https://models.inference.ai.azure.com",
}

# ── Default: Groq (per-minute resets, fastest inference) ─────────────────────
DEFAULT_PROVIDER = "groq"
DEFAULT_MODELS = {
    "groq":        "llama-3.3-70b-versatile",
    "siliconflow": "Qwen/Qwen2.5-72B-Instruct",
    "openrouter":  "meta-llama/llama-3-8b-instruct:free",
    "github":      "gpt-4o-mini",
    "gemini":      "gemini-2.0-flash",
}

_MAX_RETRIES = 3
_RETRY_DELAY_S = 65   # Wait > 60s to clear per-minute rate limits


class LLMEngine:
    """
    Provider-agnostic LLM engine.

    Reads LLM_PROVIDER, LLM_API_KEY, and LLM_MODEL from .env.
    Falls back to sensible defaults if env vars are missing.
    Automatically retries on 429 rate-limit errors.
    """

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
        self.api_key  = os.getenv("LLM_API_KEY", "")
        self.model    = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(self.provider, ""))

        if not self.api_key:
            logger.warning(
                "API key not configured.\n"
                "Please create a .env file and add your API key according to the README."
            )
            self._key_missing = True
        else:
            self._key_missing = False

        if self.provider == "gemini":
            self._init_gemini()
        else:
            self._init_openai_compatible()

        logger.info(
            f"LLMEngine ready  provider={self.provider}  model={self.model}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _init_openai_compatible(self):
        """Sets up an OpenAI-compatible client (Groq, SiliconFlow, OpenRouter…)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Please install the OpenAI SDK: pip install openai\n"
                "(Works for Groq, SiliconFlow, OpenRouter — all use OpenAI format)"
            )

        base_url = _PROVIDER_URLS.get(self.provider)
        if not base_url:
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                f"Choose from: {list(_PROVIDER_URLS.keys()) + ['gemini']}"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self._generate = self._generate_openai

    def _init_gemini(self):
        """Sets up the Google Gemini SDK."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._gemini_model = genai.GenerativeModel(self.model)
            self._generate = self._generate_gemini
        except ImportError:
            raise ImportError("pip install google-generativeai")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates a grounded answer from the retrieved context.
        Automatically retries on rate-limit errors.

        Args:
            query:   The user's natural-language question.
            context: Document text retrieved by Vector-ARC.

        Returns:
            A concise, context-grounded answer string.
        """
        if getattr(self, "_key_missing", False):
            return "API key not configured. Please create a .env file and add your API key according to the README."

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable research assistant. "
                    "Answer the user's query using the provided context. "
                    "Synthesize information from the context to give a helpful, "
                    "accurate answer. If the context is only partially relevant, "
                    "extract whatever useful information you can. "
                    "Only say 'I don't have enough information' if the context "
                    "is completely unrelated to the query."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuery: {query}",
            },
        ]

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._generate(prompt_messages)
            except Exception as e:
                err = str(e)
                if ("429" in err or "rate" in err.lower()) and attempt < _MAX_RETRIES:
                    logger.warning(
                        f"⏳ Rate limit on attempt {attempt}/{_MAX_RETRIES}. "
                        f"Waiting {_RETRY_DELAY_S}s…"
                    )
                    time.sleep(_RETRY_DELAY_S)
                    continue
                logger.error(f"LLM generation failed after {attempt} attempt(s): {e}")
                return f"System Error: LLM unavailable — {e}"

        return "System Error: Max retries exceeded."

    # ──────────────────────────────────────────────────────────────────────────
    # Provider-specific generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_openai(self, messages: list) -> str:
        """OpenAI-compatible completion (Groq / SiliconFlow / OpenRouter)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def _generate_gemini(self, messages: list) -> str:
        """Google Gemini native SDK completion."""
        # Flatten messages into a single prompt for Gemini
        prompt = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        response = self._gemini_model.generate_content(prompt)
        return response.text.strip()