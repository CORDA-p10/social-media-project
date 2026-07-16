"""Talks to any chat-completions-compatible endpoint over HTTP. The `model`
arg per call selects the backend the proxy routes to. Uses the structured-
outputs path so the model returns a Pydantic instance directly."""

from __future__ import annotations

from typing import Optional, Type, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    raw_response: str = ""


class RemoteLLM:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete_structured(
        self,
        system: str,
        user: str,
        response_model: Type[T],
        *,
        temperature: float = 1.0,
        model: str,
        seed: Optional[int] = None,
    ) -> tuple[T, LLMUsage]:
        # Prompt caching on Claude: the system prompt is ~85% of the input
        # and stable across an agent's ticks, so marking it cache_control
        # ephemeral cuts cached-input cost ~10× and trims TTFT. Litellm
        # passes the marker through to Anthropic; OpenAI/Gemini ignore the
        # unrecognized field, so guarding on model name is just hygiene.
        if "claude-" in model:
            system_message = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        else:
            system_message = {"role": "system", "content": system}

        # Forward `seed` on routes that accept it (Vertex, OpenAI incl. bare
        # gpt-* ids); Anthropic/AI-Studio routes 400 on it, so they get none.
        seed_kwargs = (
            {"seed": seed}
            if seed is not None and model.startswith(("vertex_ai/", "openai/", "gpt-"))
            else {}
        )
        resp = self._client.beta.chat.completions.parse(
            model=model,
            temperature=temperature,
            messages=[
                system_message,
                {"role": "user", "content": user},
            ],
            response_format=response_model,
            **seed_kwargs,
        )
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            raise RuntimeError(f"model {model!r} refused: {msg.refusal}")
        if msg.parsed is None:
            raise RuntimeError(f"model {model!r} returned unparseable content: {msg.content!r}")

        usage = resp.usage
        return msg.parsed, LLMUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw_response=msg.content or "",
        )
