from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return default


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(env(name))))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    base_url: str
    llm_key: str
    embed_key: str
    rerank_key: str
    llm_model: str
    emb_model: str
    reranker_model: str
    temperature: float
    max_tokens: int
    max_workers: int
    top_k: int
    reranker_top_n: int
    chunker: str
    reasoning_effort: str


def load_settings() -> Settings:
    llm_key = env("LLM_API_KEY", "OPENAI_API_KEY")
    return Settings(
        data_dir=Path(env("DATA_DIR", default="data")),
        base_url=env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/"),
        llm_key=llm_key,
        embed_key=env("EMBEDDER_API_KEY") or llm_key,
        rerank_key=env("RERANKER_API_KEY") or llm_key,
        llm_model=env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL", default="deepseek/deepseek-v4.1-flash"),
        emb_model=env("EMB_MODEL", default="baai/bge-m3"),
        reranker_model=env("RERANKER_MODEL", default="cohere/rerank-5-fast"),
        temperature=env_float("TEMPERATURE", 0.0, 0.0, 2.0),
        max_tokens=env_int("MAX_TOKENS", 1024, 128, 8192),
        max_workers=env_int("MAX_WORKERS", 4, 1, 8),
        top_k=env_int("TOP_K", 6, 1, 20),
        reranker_top_n=env_int("RERANKER_TOP_N", 4, 1, 10),
        chunker=env("CHUNKER", default="common"),
        reasoning_effort=env("LLM_REASONING_EFFORT"),
    )


SETTINGS = load_settings()
