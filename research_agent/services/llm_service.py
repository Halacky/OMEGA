"""LLM service supporting Anthropic (Claude) and DeepSeek via LangChain."""

import logging

from langchain_core.language_models import BaseChatModel

from research_agent.config import AgentConfig

logger = logging.getLogger("research_agent.llm")


def create_llm(config: AgentConfig) -> BaseChatModel:
    """Create an LLM instance based on the configured provider.

    Supports:
    - "anthropic": Claude models via langchain-anthropic
    - "deepseek": DeepSeek models via langchain-openai (OpenAI-compatible API)
    """
    provider = config.llm_provider.lower()

    if provider == "anthropic":
        return _create_anthropic_llm(config)
    elif provider == "deepseek":
        return _create_deepseek_llm(config)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported: 'anthropic', 'deepseek'"
        )


def _create_anthropic_llm(config: AgentConfig) -> BaseChatModel:
    """Create a Claude LLM instance via Anthropic API."""
    from langchain_anthropic import ChatAnthropic

    api_key = config.anthropic_api_key
    if not api_key:
        raise ValueError(
            "anthropic_api_key is required when llm_provider='anthropic'. "
            "Set OMEGA_AGENT_ANTHROPIC_API_KEY in .env"
        )

    logger.info(
        "Initializing Anthropic LLM: model=%s, temperature=%.2f, max_tokens=%d",
        config.anthropic_model,
        config.temperature,
        config.max_tokens,
    )
    return ChatAnthropic(
        model=config.anthropic_model,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def _create_deepseek_llm(config: AgentConfig) -> BaseChatModel:
    """Create a DeepSeek LLM instance via OpenAI-compatible API."""
    from langchain_openai import ChatOpenAI

    api_key = config.deepseek_api_key
    if not api_key:
        raise ValueError(
            "deepseek_api_key is required when llm_provider='deepseek'. "
            "Set OMEGA_AGENT_DEEPSEEK_API_KEY in .env"
        )

    logger.info(
        "Initializing DeepSeek LLM: model=%s, temperature=%.2f",
        config.deepseek_model,
        config.temperature,
    )
    return ChatOpenAI(
        model=config.deepseek_model,
        api_key=api_key,
        base_url=config.deepseek_base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
