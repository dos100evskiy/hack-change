from __future__ import annotations


def check_citations(answer: dict, context: str, catalog_ids: set[str]) -> list[str]:
    raise NotImplementedError


def verify(answer: dict, context: str, catalog_ids: set[str]) -> dict:
    raise NotImplementedError
