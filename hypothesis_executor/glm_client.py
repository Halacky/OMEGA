"""GLM-5 API client with OpenAI-compatible interface, proxy and retry support."""

import logging
import time

import httpx
import openai

from .config import ExecutorConfig

logger = logging.getLogger("hypothesis_executor.glm_client")


class GLMClient:
    """Wrapper around OpenAI Python client configured for GLM-5 (BigModel API)."""

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.client = self._build_client()

    def _build_client(self) -> openai.OpenAI:
        kwargs: dict = {
            "api_key": self.config.glm_api_key,
            "base_url": self.config.glm_base_url,
        }
        if self.config.use_proxy:
            kwargs["http_client"] = httpx.Client(
                proxy=self.config.proxy_url,
                timeout=httpx.Timeout(300.0, connect=30.0),
            )
        else:
            kwargs["http_client"] = httpx.Client(
                timeout=httpx.Timeout(300.0, connect=30.0),
            )
        return openai.OpenAI(**kwargs)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call GLM-5 chat completions with optional deep-thinking mode.

        Returns the assistant message content.
        Retries with exponential backoff on transient errors.
        """
        for attempt in range(1, self.config.max_retries + 1):
            try:
                extra: dict = {}
                if self.config.glm_enable_thinking:
                    extra["extra_body"] = {"thinking": {"type": "enabled"}}

                logger.info(
                    "GLM-5 API call attempt %d/%d (model=%s, thinking=%s)",
                    attempt,
                    self.config.max_retries,
                    self.config.glm_model,
                    self.config.glm_enable_thinking,
                )

                response = self.client.chat.completions.create(
                    model=self.config.glm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.glm_temperature,
                    max_tokens=self.config.glm_max_tokens,
                    **extra,
                )

                content = response.choices[0].message.content
                logger.info(
                    "GLM-5 response received: %d chars",
                    len(content) if content else 0,
                )
                return content or ""

            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                delay = self.config.retry_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Transient API error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    self.config.max_retries,
                    e,
                    delay,
                )
                if attempt == self.config.max_retries:
                    raise
                time.sleep(delay)

            except openai.APIStatusError as e:
                if e.status_code in (429, 500, 502, 503):
                    delay = self.config.retry_delay_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Server error %d (attempt %d/%d): %s. Retrying in %.1fs",
                        e.status_code,
                        attempt,
                        self.config.max_retries,
                        e,
                        delay,
                    )
                    if attempt == self.config.max_retries:
                        raise
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError("Unreachable: all retries exhausted")
