from __future__ import annotations

import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

import llm_client
from chunker import Chunk
from config import SETTINGS, Settings
from kb_loader import DATE_NUM, DATE_WORDS, ORG, Document, number_pattern, title_tokens, to_iso, visible_text

CONTEXT_BUDGET = 24000
EMBED_BATCH = 32
EMBED_CHARS = 6000
RERANK_CHARS = 2000
RRF_K = 60
RRF_DEPTH = 60
BOOST = 1.0
MIN_KEEP = 6
FAILURE_LIMIT = 3
RERANK_OFF_STATUSES = (400, 404)
RERANK_LOST_LIMIT = 120.0
INDEX_SECONDS = 180.0
SCORE_EXACT = 100
SCORE_INFLECTED = 95
SCORE_FUZZY = 88
SCORE_PARTIAL = 70
LEX_FLOOR = 0.1
LEX_ONLY_FLOOR = 0.06
VEC_FLOOR = 0.45
SMALL_DOC = 7000
LEAD = 3
DOC_WEIGHT = 0.5
DOC_DEPTH = 5
EXPANSIONS = {
    "ндс": "налог на добавленную стоимость",
    "ндфл": "налог на доходы физических лиц",
    "мрот": "минимальный размер оплаты труда",
    "пдн": "персональные данные",
    "усн": "упрощенная система налогообложения",
}

TRANSLATION = str.maketrans({"‑": "-", "–": "-", "—": "-", "−": "-", "ё": "е", "\xa0": " "})
ENDING = set("аеиоуыэюяйьмхвг")
WORD = re.compile(r"[0-9a-zа-я]+")
TURN = re.compile(r"(?:^|(?<=\s))(Пользователь|Ассистент)\s*:", re.I)
ARTICLE = re.compile(r"\bст(?:\.|ать\w*)\s*(\d+(?:\.\d+)*)")
POINT = re.compile(r"\b(?:пункт\w*|п|част\w*|ч)\.?\s*(\d+(?:\.\d+)*)")
DOC_NUMBER = re.compile(r"(?<![\w\-/])[A-Za-zА-Яа-яЁё]{1,4}-\d+(?:[/\-]\d+)?(?![\w\-/])")
MONTH_KEYS = ("январ", "феврал", "март", "апрел", "ма", "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр")
MONTH_YEAR = re.compile(
    r"\b(январ|феврал|март|апрел|ма[йяе]|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-я]*\s+(\d{4})\b", re.I
)
YEAR = re.compile(r"(?<![\w./\-])(?:19|20)\d{2}(?![\w/\-]|\.\d)")
AMOUNT = re.compile(r"\d[\d  ]*(?:[.,]\d+)?\s*(?:млрд|млн|тыс|руб|₽|%|процент)", re.I)


@dataclass
class Retrieval:
    selected_docs: list[Document] = field(default_factory=list)
    law_chunks: list[Chunk] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)
    entities: dict[str, list[str]] = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


@dataclass
class Binding:
    bound: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    laws: list[str] = field(default_factory=list)
    points: list[str] = field(default_factory=list)
    excluded: set[str] = field(default_factory=set)
    orgs: list[str] = field(default_factory=list)
    ambiguous: bool = False


def lower(text: str) -> str:
    return text.lower().translate(TRANSLATION)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", lower(text)).strip()


def split_dialog(raw: str) -> list[tuple[str, str]]:
    marks = list(TURN.finditer(raw))
    if not marks:
        return [("user", raw.strip())]
    turns = []
    head = raw[: marks[0].start()].strip()
    if head:
        turns.append(("user", head))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(raw)
        text = raw[mark.end():end].strip()
        if text:
            turns.append(("user" if mark.group(1).lower().startswith("п") else "assistant", text))
    return turns or [("user", raw.strip())]


def user_turns(raw: str) -> tuple[str, list[str]]:
    users = [text for role, text in split_dialog(raw) if role == "user"]
    if not users:
        return raw.strip(), []
    return users[-1], users[:-1]


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def extract_entities(text: str) -> dict[str, list[str]]:
    numbers = unique([m.group() for m in DOC_NUMBER.finditer(text)])
    rest = DOC_NUMBER.sub(lambda m: " " * len(m.group()), text)
    dates = []
    for pattern in (DATE_NUM, DATE_WORDS):
        dates += [to_iso(m.group()) for m in pattern.finditer(rest)]
        rest = pattern.sub(lambda m: " " * len(m.group()), rest)
    for match in MONTH_YEAR.finditer(rest):
        month = next(i for i, key in enumerate(MONTH_KEYS, start=1) if match.group(1).lower().startswith(key))
        dates.append(f"{match.group(2)}-{month:02d}")
    rest = MONTH_YEAR.sub(lambda m: " " * len(m.group()), rest)
    dates += [m.group() for m in YEAR.finditer(rest)]
    amounts = [re.sub(r"\s+", " ", m.group()).strip() for m in AMOUNT.finditer(text)]
    return {"numbers": numbers, "dates": unique(dates), "amounts": unique(amounts)}


def expand(text: str) -> str:
    low = lower(text)
    extra = [value for key, value in EXPANSIONS.items() if re.search(rf"(?<!\w){key}(?!\w)", low) and value not in low]
    return "\n".join([text, *extra])


def same_word(left: str, right: str) -> bool:
    if left == right:
        return True
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    tails = left[prefix:] + right[prefix:]
    return prefix >= 3 and len(left) - prefix <= 3 and len(right) - prefix <= 3 and set(tails) <= ENDING


def name_score(window: list[str], key: tuple[str, ...]) -> int:
    if tuple(window) == key:
        return SCORE_EXACT
    if all(same_word(a, b) for a, b in zip(window, key)):
        return SCORE_INFLECTED
    target = " ".join(key)
    if len(target) < 6:
        return 0
    ratio = fuzz.ratio(" ".join(window), target)
    return min(SCORE_INFLECTED - 1, int(ratio)) if ratio >= SCORE_FUZZY else 0


def is_named(text: str, low: str, start: int) -> bool:
    if len(text) != len(low):
        return True
    return text[start].isupper() or text[max(0, start - 1):start] in ("«", '"', "“", "„")


class Retriever:
    def __init__(
        self,
        docs: list[Document],
        chunks: list[Chunk],
        settings: Settings = SETTINGS,
        client: httpx.Client | None = None,
        use_embeddings: bool = True,
    ) -> None:
        self.docs = docs
        self.chunks = chunks
        self.settings = settings
        self.client = client or llm_client.make_client()
        self.by_id = {doc.doc_id: doc for doc in docs}
        self.sizes = {doc.doc_id: len(visible_text(doc)) for doc in docs}
        self.chunk_docs = np.array([chunk.doc_id for chunk in chunks], dtype=object)
        self.doc_rows = {doc.doc_id: i for i, doc in enumerate(docs)}
        self.chunk_rows = np.array([self.doc_rows[chunk.doc_id] for chunk in chunks], dtype=np.int64)
        self.lock = threading.Lock()
        self.failures = {"embeddings": 0, "rerank": 0}
        self.rerank_lost = 0.0
        self.reranker_off = False
        self.embeddings_off = not use_embeddings
        self.vectorizer = None
        self.tfidf = None
        self.doc_tfidf = None
        self.doc_vectors = None
        if chunks:
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), preprocessor=norm, sublinear_tf=True, dtype=np.float32
            )
            self.tfidf = self.vectorizer.fit_transform([chunk.text for chunk in chunks])
            self.doc_tfidf = self.vectorizer.transform([doc.title + "\n" + visible_text(doc) for doc in docs])
        self.names, self.name_labels = self.build_names()
        self.phrases = self.build_phrases()
        self.vectors = None if self.embeddings_off or not chunks else self.embed_chunks()
        if self.vectors is not None:
            sums = np.zeros((len(docs), self.vectors.shape[1]), dtype=np.float32)
            np.add.at(sums, self.chunk_rows, self.vectors)
            self.doc_vectors = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-9)

    def build_names(self) -> tuple[dict[tuple[str, ...], list[str]], dict[tuple[str, ...], str]]:
        names: dict[tuple[str, ...], list[str]] = {}
        labels: dict[tuple[str, ...], str] = {}
        with_parties = 0
        for doc in self.docs:
            if doc.is_law:
                continue
            found = ORG.findall(doc.parties)
            with_parties += bool(found)
            for form, name in found:
                key = tuple(WORD.findall(lower(name)))
                if key and doc.doc_id not in names.setdefault(key, []):
                    names[key].append(doc.doc_id)
                    labels[key] = f"{form} «{name}»"
        own = {key for key, ids in names.items() if with_parties >= 3 and len(ids) * 2 > with_parties}
        return {k: v for k, v in names.items() if k not in own}, labels

    def build_phrases(self) -> dict[tuple[str, str], list[str]]:
        phrases: dict[tuple[str, str], list[str]] = {}
        for doc in self.docs:
            tokens = [] if doc.is_law else title_tokens(doc.title)
            if len(tokens) >= 2:
                phrases.setdefault((tokens[0], tokens[1]), []).append(doc.doc_id)
        return phrases

    def embed_chunks(self) -> np.ndarray | None:
        texts = [chunk.text[:EMBED_CHARS] for chunk in self.chunks]
        batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
        until = time.monotonic() + INDEX_SECONDS
        try:
            with ThreadPoolExecutor(max_workers=min(4, self.settings.max_workers)) as pool:
                parts = list(pool.map(lambda batch: llm_client.embed(self.client, self.settings, batch, until=until), batches))
            matrix = np.asarray([vector for part in parts for vector in part], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if matrix.ndim != 2 or matrix.shape[0] != len(texts) or not np.isfinite(matrix).all() or (norms == 0).any():
                raise ValueError("bad embeddings matrix")
            return matrix / norms
        except Exception as exc:
            print(f"Эмбеддинги отключены: {type(exc).__name__}; поиск только лексический", file=sys.stderr)
            self.embeddings_off = True
            return None

    def note_failure(self, kind: str, exc: Exception, seconds: float = 0.0) -> None:
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else 0
        unset = not self.settings.base_url
        with self.lock:
            self.failures[kind] += 1
            if kind == "rerank":
                self.rerank_lost += seconds
                if self.reranker_off:
                    return
                if status in RERANK_OFF_STATUSES or unset:
                    self.reranker_off = True
                    print(f"Реранкер отключён: статус {status}; используется порядок слияния", file=sys.stderr)
                elif self.rerank_lost >= RERANK_LOST_LIMIT:
                    self.reranker_off = True
                    print(f"Реранкер отключён: на сбои ушло {self.rerank_lost:.0f} с; используется порядок слияния", file=sys.stderr)
                return
            permanent = (400 <= status < 500 and status != 429) or unset
            if (permanent or self.failures[kind] >= FAILURE_LIMIT) and not self.embeddings_off:
                self.embeddings_off = True
                print(f"Эмбеддинги отключены: {type(exc).__name__}; поиск только лексический", file=sys.stderr)

    def note_success(self, kind: str) -> None:
        with self.lock:
            self.failures[kind] = 0

    def match_names(self, text: str) -> tuple[set[tuple[str, ...]], bool]:
        low = lower(text)
        parts = [(m.group(), m.start(), m.end()) for m in WORD.finditer(low)]
        candidates = []
        for key in self.names:
            size = len(key)
            for i in range(len(parts) - size + 1):
                score = name_score([p[0] for p in parts[i:i + size]], key)
                if score:
                    candidates.append((score, parts[i][1], parts[i + size - 1][2], key))
            if size > 1 and len(key[0]) >= 4:
                for word, start, end in parts:
                    if same_word(word, key[0]) and is_named(text, low, start):
                        candidates.append((SCORE_PARTIAL, start, end, key))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        accepted = []
        for cand in candidates:
            if not any(a[0] > cand[0] and a[1] < cand[2] and cand[1] < a[2] for a in accepted):
                accepted.append(cand)
        found = {c[3] for c in accepted if c[0] >= SCORE_FUZZY}
        groups: dict[tuple[int, int], set[tuple[str, ...]]] = {}
        for score, start, end, key in accepted:
            if score < SCORE_FUZZY:
                groups.setdefault((start, end), set()).add(key)
        ambiguous = False
        partial: set[tuple[str, ...]] = set()
        for keys in groups.values():
            if not keys & found:
                partial |= keys
                ambiguous = ambiguous or len(keys) > 1
        return found | partial, ambiguous

    def match_laws(self, low: str) -> list[str]:
        result = []
        for article in unique(ARTICLE.findall(low)):
            docs = [doc for doc in self.docs if doc.is_law and doc.article == article]
            named = [d for d in docs if d.law and re.search(r"(?<!\w)" + re.escape(lower(d.law).split()[0]) + r"(?!\w)", low)]
            result += [doc.doc_id for doc in (named or docs)]
        return unique(result)

    def mutually_related(self, ids: list[str]) -> bool:
        return all(a == b or a in self.by_id[b].relates_to for a in ids for b in ids)

    def bind(self, text: str) -> Binding:
        low = norm(text)
        binding = Binding()
        strong = []
        for doc in self.docs:
            pattern = None if doc.is_law else number_pattern(doc.number)
            if pattern and pattern.search(low):
                strong.append(doc.doc_id)
        keys, binding.ambiguous = self.match_names(text)
        binding.orgs = sorted(self.name_labels[key] for key in keys)
        for key in keys:
            strong += self.names[key]
        strong = unique(strong)
        weak = []
        stems = [word[:5] for word in re.findall(r"[а-яa-z]{4,}", low)]
        pairs = set(zip(stems, stems[1:]))
        for phrase, ids in self.phrases.items():
            if phrase in pairs and not set(ids) & set(strong):
                weak += ids
                binding.ambiguous = binding.ambiguous or not self.mutually_related(ids)
        binding.bound = sorted(unique(strong + weak), key=self.doc_rows.get)
        related = [r for d in binding.bound for r in self.by_id[d].relates_to if r in self.by_id]
        binding.related = [r for r in unique(related) if r not in binding.bound]
        binding.laws = self.match_laws(low)
        binding.points = unique(POINT.findall(low)) if binding.laws else []
        if binding.bound and not binding.ambiguous:
            keep = set(binding.bound) | set(binding.related)
            types = {self.by_id[d].doc_type for d in binding.bound} - {"документ"}
            binding.excluded = {
                d.doc_id for d in self.docs if not d.is_law and d.doc_type in types and d.doc_id not in keep
            }
        return binding

    def lexical_scores(self, text: str, by_doc: bool = False) -> np.ndarray:
        query = self.vectorizer.transform([text])
        return ((self.doc_tfidf if by_doc else self.tfidf) @ query.T).toarray().ravel()

    def query_vector(self, text: str) -> np.ndarray | None:
        if self.embeddings_off or self.vectors is None:
            return None
        try:
            raw = llm_client.embed(self.client, self.settings, [text[:EMBED_CHARS]], retries=1, timeout=30.0)[0]
            vector = np.asarray(raw, dtype=np.float32)
            length = float(np.linalg.norm(vector))
            if vector.shape[0] != self.vectors.shape[1] or not np.isfinite(length) or length == 0:
                raise ValueError("bad query embedding")
        except llm_client.CLIENT_ERRORS as exc:
            self.note_failure("embeddings", exc)
            return None
        self.note_success("embeddings")
        return vector / length

    def boosts(self, binding: Binding, boosted_ids: set[str]) -> np.ndarray:
        boost = np.where(np.isin(self.chunk_docs, list(boosted_ids)), BOOST, 0.0)
        if binding.points:
            numbers = "|".join(re.escape(number) for number in binding.points)
            pattern = re.compile(rf"(?:^|, )(?:п|ч)\. (?:{numbers})(?:,|$)")
            laws = set(binding.laws)
            for index, chunk in enumerate(self.chunks):
                if chunk.doc_id in laws and pattern.search(chunk.locator):
                    boost[index] += BOOST
        return boost

    def fuse(
        self,
        chunk_scores: list[np.ndarray],
        doc_scores: list[np.ndarray],
        allowed: np.ndarray,
        boost: np.ndarray,
    ) -> list[int]:
        fused = np.zeros(len(self.chunks), dtype=np.float64)
        for scores in chunk_scores:
            masked = np.where(allowed, scores, -np.inf)
            top = [int(i) for i in np.argsort(-masked, kind="stable")[:RRF_DEPTH] if masked[i] > 0]
            for rank, index in enumerate(top, start=1):
                fused[index] += 1.0 / (RRF_K + rank)
        found = fused > 0
        for scores in doc_scores:
            bonus = np.zeros(len(self.docs), dtype=np.float64)
            top = [int(i) for i in np.argsort(-scores, kind="stable")[:DOC_DEPTH] if scores[i] > 0]
            for rank, index in enumerate(top, start=1):
                bonus[index] = DOC_WEIGHT / (RRF_K + rank)
            fused[found] += bonus[self.chunk_rows[found]]
        fused[allowed] += boost[allowed]
        order = [int(i) for i in np.argsort(-fused, kind="stable") if fused[i] > 0]
        return order[: self.settings.top_k * 4]

    def rerank(self, text: str, order: list[int], enabled: bool) -> tuple[list[int], bool]:
        if not enabled or self.reranker_off or not order or not llm_client.BUDGET.full():
            return order, False
        documents = [self.chunks[i].text[:RERANK_CHARS] for i in order]
        started = time.monotonic()
        try:
            ranked = llm_client.rerank(
                self.client, self.settings, text, documents, len(order), until=llm_client.BUDGET.full_until
            )
        except llm_client.CLIENT_ERRORS as exc:
            self.note_failure("rerank", exc, time.monotonic() - started)
            return order, False
        self.note_success("rerank")
        seen = {i for i, _ in ranked}
        return [order[i] for i, _ in ranked] + [order[i] for i in range(len(order)) if i not in seen], True

    def whole_laws(self, order: list[int], keep: int) -> set[str]:
        counts: dict[str, int] = {}
        result = set()
        for position, index in enumerate(order[:keep]):
            doc_id = self.chunks[index].doc_id
            if not self.by_id[doc_id].is_law or self.sizes[doc_id] > SMALL_DOC:
                continue
            counts[doc_id] = counts.get(doc_id, 0) + 1
            if position < LEAD or counts[doc_id] >= 2:
                result.add(doc_id)
        return result

    def search(self, query: str, history_user_turns: list[str] | None = None, use_reranker: bool = True) -> Retrieval:
        text = "\n".join([query, *reversed(history_user_turns or [])]).strip()
        result = Retrieval(entities=extract_entities(text))
        binding = self.bind(text)
        result.entities["orgs"] = binding.orgs
        selected: list[str] = []
        used = 0

        def add(doc_id: str) -> bool:
            nonlocal used
            if doc_id in selected:
                return True
            if used + self.sizes[doc_id] > CONTEXT_BUDGET:
                return False
            selected.append(doc_id)
            used += self.sizes[doc_id]
            return True

        for doc_id in binding.bound + binding.related:
            add(doc_id)
        boosted_ids = set(binding.laws) | {d for d in binding.bound if d not in selected}
        order: list[int] = []
        ranked: list[int] = []
        reranked = False
        lexical_max = vector_max = 0.0
        vector = None
        keep = max(self.settings.reranker_top_n * 2, MIN_KEEP)
        if self.tfidf is not None and text:
            wide = expand(text)
            lexical = self.lexical_scores(wide)
            chunk_scores = [lexical]
            doc_scores = [self.lexical_scores(wide, by_doc=True)]
            vector = self.query_vector(wide)
            if vector is not None:
                chunk_scores.append(self.vectors @ vector)
                doc_scores.append(self.doc_vectors @ vector)
                vector_max = float(chunk_scores[1].max())
            lexical_max = float(lexical.max())
            allowed = ~np.isin(self.chunk_docs, list(set(selected) | binding.excluded))
            fused = self.fuse(chunk_scores, doc_scores, allowed, self.boosts(binding, boosted_ids))
            order, reranked = self.rerank(text, fused, use_reranker)

        whole = self.whole_laws(order, keep)
        hits: list[str] = []
        for index in order:
            if len(ranked) >= keep:
                break
            chunk = self.chunks[index]
            doc = self.by_id[chunk.doc_id]
            if doc.doc_id in selected:
                continue
            if (not doc.is_law or doc.doc_id in whole) and add(doc.doc_id):
                if not doc.is_law:
                    hits.append(doc.doc_id)
            elif used + len(chunk.text) <= CONTEXT_BUDGET:
                result.law_chunks.append(chunk)
                used += len(chunk.text)
            else:
                continue
            ranked.append(index)
        for doc_id in hits:
            for related in self.by_id[doc_id].relates_to:
                if related in self.by_id and related not in binding.excluded:
                    add(related)

        result.selected_docs = [self.by_id[doc_id] for doc_id in selected]
        weak = lexical_max < LEX_ONLY_FLOOR if vector is None else lexical_max < LEX_FLOOR and vector_max < VEC_FLOOR
        result.flags = {
            "ambiguous": binding.ambiguous,
            "nothing_found": not binding.bound and not binding.laws and (not ranked or weak),
            "embeddings_off": vector is None,
            "reranker_off": not reranked,
        }
        result.debug = {
            "bound": binding.bound,
            "related": binding.related,
            "laws": binding.laws,
            "points": binding.points,
            "excluded": sorted(binding.excluded),
            "ranked": [self.chunks[i].chunk_id for i in ranked],
            "lexical_max": round(lexical_max, 4),
            "vector_max": round(vector_max, 4),
            "context_chars": used,
        }
        return result
