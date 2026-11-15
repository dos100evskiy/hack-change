from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass

import httpx

from config import SETTINGS, Settings

RETRY_PAUSES = (1.0, 3.0, 7.0)
CHAT_TIMEOUT = 60.0
MIN_ATTEMPT = 2.0
JSON_MODE_OFF = threading.Event()
SLOTS = threading.BoundedSemaphore(max(SETTINGS.max_workers, 4))


class OutOfTime(ValueError):
    pass


CLIENT_ERRORS = (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError)


@dataclass
class Budget:
    full_until: float = math.inf
    stop_at: float = math.inf

    def full(self) -> bool:
        return time.monotonic() < self.full_until


BUDGET = Budget()
STATS: Counter = Counter()
STATS_LOCK = threading.Lock()


def count(name: str) -> None:
    with STATS_LOCK:
        STATS[name] += 1


def make_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
    )


def time_left(until: float | None) -> float:
    return min(BUDGET.stop_at, math.inf if until is None else until) - time.monotonic()


def send(client: httpx.Client, url: str, payload: dict, headers: dict, timeout: float, until: float | None) -> httpx.Response:
    if not SLOTS.acquire(timeout=max(0.0, min(time_left(until), threading.TIMEOUT_MAX))):
        raise OutOfTime("no free slot before deadline")
    try:
        left = time_left(until)
        if left < MIN_ATTEMPT:
            raise OutOfTime("deadline reached")
        return client.post(url, json=payload, headers=headers, timeout=httpx.Timeout(min(timeout, left), connect=min(10.0, left)))
    finally:
        SLOTS.release()


def post_json(
    client: httpx.Client,
    settings: Settings,
    path: str,
    key: str,
    payload: dict,
    retries: int = 3,
    timeout: float | None = None,
    until: float | None = None,
) -> dict:
    if not settings.base_url:
        raise ValueError("base url is not set")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    limit = 60.0 if timeout is None else timeout
    for attempt in range(retries + 1):
        try:
            response = send(client, settings.base_url + path, payload, headers, limit, until)
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == retries:
                if response.is_error:
                    count(f"{path} {response.status_code}")
                response.raise_for_status()
                return response.json()
        except httpx.TransportError as exc:
            count(f"{path} {type(exc).__name__}")
            if attempt == retries:
                raise
        except OutOfTime:
            count(f"{path} out of time")
            raise
        pause = RETRY_PAUSES[min(attempt, len(RETRY_PAUSES) - 1)]
        if time_left(until) < pause + MIN_ATTEMPT:
            count(f"{path} out of time")
            raise OutOfTime("no time for retry")
        count(f"{path} retry")
        time.sleep(pause)
    raise ValueError("no response")


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def chat(
    client: httpx.Client,
    settings: Settings,
    messages: list[dict],
    json_mode: bool = True,
    timeout: float | None = CHAT_TIMEOUT,
    until: float | None = None,
) -> str:
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    if settings.reasoning_effort:
        payload["reasoning_effort"] = settings.reasoning_effort
    use_json = json_mode and not JSON_MODE_OFF.is_set()
    for _ in range(3):
        body = {**payload, "response_format": {"type": "json_object"}} if use_json else payload
        try:
            result = post_json(client, settings, "/chat/completions", settings.llm_key, body, timeout=timeout, until=until)
        except httpx.HTTPStatusError as exc:
            if use_json and exc.response.status_code in (400, 422):
                use_json = False
                continue
            raise
        if json_mode and not use_json:
            JSON_MODE_OFF.set()
        content = strip_think(result["choices"][0]["message"].get("content") or "")
        if content:
            return content
    raise ValueError("empty completion")


def embed(
    client: httpx.Client,
    settings: Settings,
    texts: list[str],
    retries: int = 2,
    timeout: float | None = None,
    until: float | None = None,
) -> list[list[float]]:
    payload = {"model": settings.emb_model, "input": texts, "encoding_format": "float"}
    result = post_json(client, settings, "/embeddings", settings.embed_key, payload, retries, timeout, until)
    items = sorted(result["data"], key=lambda item: int(item.get("index", 0)))
    vectors = [item["embedding"] for item in items]
    if len(vectors) != len(texts) or any(not vector for vector in vectors):
        raise ValueError("unexpected embeddings response")
    return vectors


def rerank(
    client: httpx.Client,
    settings: Settings,
    query: str,
    documents: list[str],
    top_n: int,
    retries: int = 3,
    timeout: float | None = 30.0,
    until: float | None = None,
) -> list[tuple[int, float]]:
    payload = {
        "model": settings.reranker_model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }
    result = post_json(client, settings, "/rerank", settings.rerank_key, payload, retries, timeout, until)
    items = result.get("results") or result.get("data") or []
    ranked = []
    for item in items:
        index = int(item["index"])
        score = item.get("relevance_score", item.get("score", 0.0))
        if 0 <= index < len(documents) and index not in {i for i, _ in ranked}:
            ranked.append((index, float(score or 0.0)))
    if not ranked:
        raise ValueError("unexpected rerank response")
    return ranked[:top_n]
