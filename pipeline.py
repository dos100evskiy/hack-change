from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import date

import httpx

import llm_client
import prompts
import retriever as search
from chunker import Chunk, chunk_documents
from config import SETTINGS, Settings
from kb_loader import MONTHS, Document, format_dmy, load_knowledge_base, number_pattern, overlaps_suspicious, visible_text

FALLBACK = "Не удалось сформировать подтверждённый документами ответ по базе знаний."
TYPES = ("answer", "partial", "false_premise", "no_info", "ambiguous", "illegal", "off_topic", "injection")
MAX_ANSWER = 10000
MAX_CITATIONS = 6
MAX_QUOTE = 300
MAX_LOCATOR = 60
ROLE_LABELS = {"user": "Пользователь", "assistant": "Ассистент"}
HINT_AMBIGUOUS = (
    "вопрос может относиться к нескольким документам базы; если предмет вопроса не указывает на один из них, "
    "дай ответ по каждому"
)
HINT_NOTHING = "поиск не нашёл близких по теме положений; проверь по каталогу, есть ли в базе документы по теме"
CLOSING_TAG = re.compile(r"</(?=\s*(?:doc|fragment|history|question)\b)", re.I)
HAS_REFERENCE = re.compile(r"источник|(?<!\w)(?:п|ст|ч)\.\s*\d|(?<!\w)(?:пункт|стать|част)\w*\s+\d", re.I)
ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', re.S)
MARKDOWN_LINE = re.compile(r"^\s*(?:#{1,6}\s+|[-*•]\s+|>\s+)", re.M)
MARKDOWN_INLINE = re.compile(r"\*\*|__|`+")
DOC_ID = re.compile(r"d\d+", re.I)
ID_LEAK = re.compile(r"\s*\((?:[а-яё]+\.?\s*)?(?:d\.?\s?\d{1,3}|д\.?\s?0\d)\)", re.I)
FOLD = str.maketrans({
    "ё": "е", "\xa0": " ", "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‑": "-", "–": "-", "—": "-", "−": "-",
})


@dataclass
class Engine:
    settings: Settings
    client: httpx.Client
    docs: list[Document]
    chunks: list[Chunk]
    retriever: search.Retriever
    catalog: str


_engine: Engine | None = None
_engine_lock = threading.Lock()


def split_dialog(raw_question: str) -> tuple[list[tuple[str, str]], str]:
    turns = search.split_dialog(raw_question)
    last = max((i for i, (role, _) in enumerate(turns) if role == "user"), default=-1)
    if last < 0:
        return [], raw_question.strip()
    return turns[:last] + turns[last + 1:], turns[last][1]


def today_text() -> str:
    now = date.today()
    month = next(name for name, number in MONTHS.items() if int(number) == now.month)
    return f"{now.day} {month} {now.year} г."


def escape(text: str) -> str:
    return CLOSING_TAG.sub("< /", text)


def attribute(text: str) -> str:
    return " ".join(text.replace('"', "'").replace("<", "‹").replace(">", "›").split())


def period(doc: Document) -> str:
    parts = []
    if doc.valid_from:
        parts.append("с " + format_dmy(doc.valid_from))
    if doc.valid_to:
        parts.append("по " + format_dmy(doc.valid_to))
    return "срок действия: " + " ".join(parts) if parts else ""


def catalog_line(doc: Document) -> str:
    fields = [doc.doc_id, doc.title, doc.doc_type]
    if doc.number and not doc.is_law:
        fields.append("№ " + doc.number)
    if doc.date:
        fields.append("от " + format_dmy(doc.date))
    fields.append(doc.status)
    if doc.parties:
        fields.append("стороны: " + doc.parties)
    if period(doc):
        fields.append(period(doc))
    if doc.exported:
        fields.append("редакция на " + doc.exported)
    return " | ".join(" ".join(str(value).split()) for value in fields)


def build_catalog(docs: list[Document]) -> str:
    return "\n".join(catalog_line(doc) for doc in docs)


def doc_open(doc: Document, part: str) -> str:
    return (
        f'<doc id="{doc.doc_id}" title="{attribute(doc.title)}" type="{attribute(doc.doc_type)}" '
        f'status="{attribute(doc.status)}" part="{part}">'
    )


def build_context(retrieval: search.Retrieval, by_id: dict[str, Document]) -> str:
    blocks = []
    for doc in retrieval.selected_docs:
        blocks.append(f"{doc_open(doc, 'полный текст')}\n{escape(visible_text(doc).strip())}\n</doc>")
    grouped: dict[str, list[Chunk]] = {}
    for chunk in retrieval.law_chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    for doc_id, chunks in grouped.items():
        doc = by_id[doc_id]
        parts = [
            f'<fragment locator="{attribute(chunk.locator)}">\n{escape(chunk.text.strip())}\n</fragment>'
            for chunk in sorted(chunks, key=lambda c: c.start)
        ]
        blocks.append(f"{doc_open(doc, 'отдельные фрагменты')}\n" + "\n".join(parts) + "\n</doc>")
    return "\n\n".join(blocks)


def build_history(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"{ROLE_LABELS[role]}: {escape(' '.join(text.split()))}" for role, text in history)


def date_text(value: str) -> str:
    parts = value.split("-")
    return ".".join(reversed(parts))


def build_hints(flags: dict[str, bool]) -> str:
    hints = []
    if flags.get("ambiguous"):
        hints.append(HINT_AMBIGUOUS)
    if flags.get("nothing_found"):
        hints.append(HINT_NOTHING)
    return "; ".join(hints)


def get_engine() -> Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            docs = load_knowledge_base(SETTINGS.data_dir)
            chunks = chunk_documents(docs)
            client = llm_client.make_client()
            _engine = Engine(
                settings=SETTINGS,
                client=client,
                docs=docs,
                chunks=chunks,
                retriever=search.Retriever(docs, chunks, SETTINGS, client),
                catalog=build_catalog(docs),
            )
        return _engine


def parse_json(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "answer" in data:
            return data
    return None


def salvage(text: str) -> str:
    match = ANSWER_FIELD.search(text)
    if not match:
        return re.sub(r"^```\w*|```$", "", text.strip()).strip()
    body = match.group(1)
    try:
        return json.loads('"' + body + '"')
    except json.JSONDecodeError:
        return body.replace('\\"', '"').replace("\\n", " ")


def generate(engine: Engine, messages: list[dict], extra_calls: bool = True) -> tuple[dict, dict]:
    info = {"calls": 1, "json_ok": True}
    text = llm_client.chat(engine.client, engine.settings, messages)
    data = parse_json(text)
    if data is None and not extra_calls:
        info["json_ok"] = False
        data = {"type": "answer", "answer": salvage(text), "citations": [], "missing": ""}
    if data is None:
        info["calls"] = 2
        retry = messages + [{"role": "assistant", "content": text}, {"role": "user", "content": prompts.FORMAT_REMINDER}]
        try:
            second = llm_client.chat(engine.client, engine.settings, retry)
        except llm_client.CLIENT_ERRORS:
            second = ""
        data = parse_json(second)
        if data is None:
            info["json_ok"] = False
            data = {"type": "answer", "answer": salvage(second) or salvage(text), "citations": [], "missing": ""}
    return data, info


def clean_answer(value: object) -> str:
    text = value if isinstance(value, str) else ""
    text = MARKDOWN_INLINE.sub("", MARKDOWN_LINE.sub("", text))
    text = ID_LEAK.sub("", text)
    return " ".join(text.split())


def fold(text: str) -> str:
    return text.lower().translate(FOLD)


def fold_map(text: str) -> tuple[str, list[int]]:
    chars = []
    positions = []
    space = True
    for index, char in enumerate(fold(text)):
        if char.isspace():
            if not space:
                chars.append(" ")
                positions.append(index)
            space = True
        else:
            chars.append(char)
            positions.append(index)
            space = False
    return "".join(chars), positions


def find_quote(doc: Document, quote: str) -> tuple[int, int]:
    quote = quote.strip().strip("…").strip(".").strip()
    if len(quote) < 8:
        return -1, -1
    start = doc.text.find(quote)
    if start >= 0:
        return start, start + len(quote)
    folded, positions = fold_map(doc.text)
    needle = " ".join(fold(quote).split())
    start = folded.find(needle)
    if start < 0:
        return -1, -1
    return positions[start], positions[start + len(needle) - 1] + 1


def locator_key(text: str) -> str:
    low = fold(text)
    low = re.sub(r"пункт\w*", "п", low)
    low = re.sub(r"стать\w*", "ст", low)
    low = re.sub(r"част\w*", "ч", low)
    return re.sub(r"[^0-9a-zа-я]", "", low)


def find_by_locator(doc: Document, locator: str, chunks: list[Chunk]) -> tuple[int, int]:
    key = locator_key(locator)
    if not key:
        return -1, -1
    own = [chunk for chunk in chunks if chunk.doc_id == doc.doc_id]
    for chunk in own:
        if locator_key(chunk.locator) == key:
            return chunk.start, chunk.end
    for chunk in own:
        if locator_key(chunk.locator).endswith(key) and not key.isdigit():
            return chunk.start, chunk.end
    return -1, -1


def resolve_doc(value: object, engine: Engine) -> Document | None:
    text = str(value or "").strip()
    by_id = engine.retriever.by_id
    match = DOC_ID.fullmatch(text) or DOC_ID.search(text)
    if match and match.group().lower() in by_id:
        return by_id[match.group().lower()]
    low = fold(text)
    for doc in engine.docs:
        pattern = None if doc.is_law else number_pattern(doc.number)
        if (pattern and pattern.search(low)) or (low and low == fold(doc.title)):
            return doc
    return None


def short_line(value: object, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def build_sources(citations: object, engine: Engine) -> list[dict]:
    sources = []
    seen = set()
    for item in citations if isinstance(citations, list) else []:
        if not isinstance(item, dict) or len(sources) >= MAX_CITATIONS:
            continue
        doc = resolve_doc(item.get("doc_id"), engine)
        if doc is None:
            continue
        locator = short_line(item.get("locator"), MAX_LOCATOR)
        quote = short_line(item.get("quote"), MAX_QUOTE)
        start, end = find_quote(doc, quote)
        if start < 0:
            start, end = find_by_locator(doc, locator, engine.chunks)
        if start >= 0 and overlaps_suspicious(start, end, doc.suspicious):
            continue
        key = (doc.doc_id, locator, start)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "doc_id": doc.doc_id, "title": doc.title, "locator": locator, "quote": quote, "start": start, "end": end,
        })
    return sources


def source_label(source: dict, doc: Document) -> str:
    locator = source["locator"]
    if not doc.is_law:
        pattern = number_pattern(doc.number)
        number = f"№ {doc.number}" if pattern and not pattern.search(doc.title) else ""
        return ", ".join(part for part in (doc.title, number, locator) if part)
    article = f"ст. {doc.article}" if doc.article else ""
    if locator_key(locator).startswith(locator_key(article)) and article:
        return ", ".join(part for part in (doc.law, locator) if part)
    return ", ".join(part for part in (doc.law, article, locator) if part)


def finalize(answer: str, sources: list[dict], engine: Engine) -> str:
    if answer and sources and not HAS_REFERENCE.search(answer):
        labels = dict.fromkeys(source_label(s, engine.retriever.by_id[s["doc_id"]]) for s in sources)
        tail = " Источник: " + "; ".join(labels) + "."
        answer = answer.rstrip() + ("" if answer.rstrip().endswith((".", "!", "?")) else ".") + tail
    return answer[:MAX_ANSWER].strip()


def answer_question(raw_question: str, use_reranker: bool = True, self_check: bool = True) -> dict:
    started = time.monotonic()
    debug: dict = {}
    try:
        engine = get_engine()
        history, question = split_dialog(str(raw_question or ""))
        earlier = [text for role, text in history if role == "user"]
        retrieval = engine.retriever.search(question, earlier, use_reranker)
        debug.update(flags=retrieval.flags, retrieval=retrieval.debug, entities=retrieval.entities)
        user = prompts.build_user_prompt(
            engine.catalog,
            build_context(retrieval, engine.retriever.by_id),
            build_history(history),
            escape(question),
            ", ".join(date_text(value) for value in retrieval.entities.get("dates", [])),
            build_hints(retrieval.flags),
        )
        messages = [
            {"role": "system", "content": prompts.build_system_prompt(today_text())},
            {"role": "user", "content": user},
        ]
        debug["prompt_chars"] = len(user)
        data, info = generate(engine, messages, self_check and llm_client.BUDGET.full())
        debug.update(info, self_check=self_check and llm_client.BUDGET.full())
        kind = str(data.get("type", "")).strip().lower()
        kind = kind if kind in TYPES else "answer"
        sources = build_sources(data.get("citations"), engine)
        answer = finalize(clean_answer(data.get("answer")), sources, engine)
        debug.update(type=kind, missing=short_line(data.get("missing"), 500), seconds=round(time.monotonic() - started, 2))
        if not answer:
            return {"answer": FALLBACK, "status": "error", "sources": [], "debug": {**debug, "error": "empty answer"}}
        return {"answer": answer, "status": kind, "sources": sources, "debug": debug}
    except Exception as exc:
        debug.update(error=type(exc).__name__, seconds=round(time.monotonic() - started, 2))
        return {"answer": FALLBACK, "status": "error", "sources": [], "debug": debug}
