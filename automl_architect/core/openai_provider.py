"""OpenAI Responses adapter. SDK retries are disabled; the harness owns limits."""

from __future__ import annotations

import time

from pydantic import ValidationError

from .errors import LLMOutputError, LLMRefusalError, LLMTransientError


class OpenAIProvider:
    def __init__(self, owner, client=None):
        self.owner = owner
        self._sdk = client

    @property
    def client(self):
        if self._sdk is None:
            from openai import OpenAI

            key = self.owner.settings.openai_api_key
            self._sdk = OpenAI(
                api_key=key.get_secret_value() if key else None,
                max_retries=0,
                timeout=self.owner.settings.llm_timeout_seconds,
            )
        return self._sdk

    def call(self, *, user, system, max_tokens, output_model=None, max_attempts=3):
        import openai

        from .llm import LLMResult, StructuredResult, Usage

        owner = self.owner
        model = owner.settings.model
        started = time.perf_counter()
        total = Usage()
        prompt = user
        for attempt in range(1, min(max_attempts, owner.settings.llm_max_retries + 1) + 1):
            owner._before_call(system, prompt, max_tokens, output_model)
            request = {
                "model": model,
                "input": [
                    {"role": "system", "content": "\n\n".join(b.text for b in system)},
                    {"role": "user", "content": prompt},
                ],
                "max_output_tokens": max_tokens,
                "store": False,
            }
            if output_model is not None:
                request["text_format"] = output_model
            try:
                client = self.client
                if owner._budget is not None and hasattr(client, "with_options"):
                    client = client.with_options(
                        timeout=min(
                            owner.settings.llm_timeout_seconds,
                            max(0.1, owner._budget.state.time_remaining),
                        )
                    )
                response = (
                    client.responses.parse(**request)
                    if output_model
                    else client.responses.create(**request)
                )
            except (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ) as exc:
                if attempt >= min(max_attempts, owner.settings.llm_max_retries + 1):
                    raise LLMTransientError(
                        f"OpenAI temporarily unavailable ({type(exc).__name__})."
                    ) from None
                owner._sleep(attempt)
                continue
            except ValidationError:
                # The SDK may fail after a billed response. Its reservation remains.
                if attempt >= max_attempts:
                    raise LLMOutputError(
                        "OpenAI output did not satisfy the application schema."
                    ) from None
                prompt = user + "\nReturn a complete object matching the required schema."
                continue
            except openai.APIStatusError as exc:
                from .errors import ConfigurationError

                raise ConfigurationError(
                    f"OpenAI rejected the request (HTTP {exc.status_code}). Check credentials and model access."
                ) from None
            details = getattr(response.usage, "input_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            usage = Usage(
                input_tokens=max(0, (getattr(response.usage, "input_tokens", 0) or 0) - cached),
                output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                cache_read_tokens=cached,
                calls=1,
            )
            owner._record(usage)
            total.merge(usage)
            for item in getattr(response, "output", []) or []:
                for block in getattr(item, "content", []) or []:
                    if getattr(block, "type", None) == "refusal":
                        raise LLMRefusalError(None, "The selected model declined the request.")
            if getattr(response, "status", "completed") != "completed":
                raise LLMOutputError(
                    "OpenAI returned an incomplete response; increase the output budget or simplify the request."
                )
            result = {
                "text": response.output_text,
                "usage": total,
                "model": response.model,
                "stop_reason": "end_turn",
                "seconds": time.perf_counter() - started,
                "attempts": attempt,
            }
            if output_model is None:
                return LLMResult(**result)
            try:
                value = output_model.model_validate(response.output_parsed)
            except ValidationError:
                if attempt >= max_attempts:
                    raise LLMOutputError("OpenAI output failed validation.") from None
                prompt = user + "\nCorrect the object to match the required schema."
                continue
            return StructuredResult(value=value, **result)
        raise LLMOutputError("OpenAI exhausted its bounded validation attempts.")
