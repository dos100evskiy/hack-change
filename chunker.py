from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from kb_loader import Document, Line, overlaps_suspicious, split_lines

MAX_CHUNK = 1200
CONTEXT_CAP = 400
NOTE_PREFIX = "Примечание о редакции: "

META_RE = re.compile(r"^(Источник|URL|Дата выгрузки)\s*:", re.I)
TITLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)*)\.\s*(.*)$")
POINT_RE = re.compile(r"^(\d+(?:\.\d+)*(?:-\d+)?)\.\s+\S")
SUB_RE = re.compile(r"^(\d+(?:\.\d+)*(?:-\d+)?)\)\s*\S")
NOTE_RE = re.compile(
    r"^(?:(?:Пункт|Пункты|Подпункт|Подпункты|Часть|Части|Статья|Абзац|Абзацы)\s+[\d.,\-\s и]*?"
    r"(?:изменен|дополнен|утратил|исключен|введен|признан|излож|приостановлен|применя|вступа)\w*"
    r"|Федеральн\w+\s+закон\w*\s+от\b|О проверке конституционности|См\.)"
)
SECTION_RE = re.compile(r"^(\d+)\.\s+\S")
REMARK_RE = re.compile(r"^Примечани[яе][.:]?$", re.I)
ITEM_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+\S")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    locator: str
    text: str
    notes: str
    start: int
    end: int


@dataclass
class Item:
    line: Line
    kind: str
    tag: str | None = None
    idx: int = 0

    @property
    def text(self) -> str:
        return self.line.text


@dataclass
class Raw:
    notes: list[tuple[int, Item]] = field(default_factory=list)
    body: list[Item] = field(default_factory=list)
    lead: Item | None = None


@dataclass
class Unit:
    locator: str
    items: list[Item] = field(default_factory=list)
    heading: str = ""
    paragraphs: bool = False


ABBREVIATIONS = {"ст", "п", "пп", "ч", "г", "гг", "тыс", "млн", "млрд", "руб", "т", "д", "др", "см", "им", "абз"}
SPLIT_POINT = re.compile(r"([.;])\s+(?=\S)")


def sentence_cuts(text: str) -> list[int]:
    cuts = []
    for match in SPLIT_POINT.finditer(text):
        end = match.end()
        if match.group(1) == ".":
            word = re.search(r"(\w+)$", text[: match.start()])
            if (word and word.group(1).lower() in ABBREVIATIONS) or not re.match(r"[А-ЯЁA-Z«\"(]", text[end:end + 1]):
                continue
        cuts.append(end)
    return cuts


def split_long(item: "Item") -> list["Item"]:
    text = item.text
    if len(text) <= MAX_CHUNK:
        return [item]
    pieces, begin, last = [], 0, 0
    for cut in sentence_cuts(text) + [len(text)]:
        if cut - begin > MAX_CHUNK and last > begin:
            pieces.append((begin, last))
            begin = last
        last = cut
    pieces.append((begin, len(text)))
    result = []
    for number, (a, b) in enumerate(pieces):
        part = text[a:b].strip()
        start = item.line.start + a + (len(text[a:b]) - len(text[a:b].lstrip()))
        line = Line(start, start + len(part), part)
        result.append(Item(line, item.kind if number == 0 else "text", item.tag, item.idx))
    return result


def body_length(items: list[Item]) -> int:
    return sum(len(item.text) for item in items)


def pack(items: list[Item], carry: list[Item]) -> tuple[list[Raw], list[Item]]:
    raws: list[Raw] = []
    notes: list[tuple[int, Item]] = []
    body: list[Item] = []
    lead: Item | None = None
    size = 0
    pending = list(carry)
    for item in items:
        if item.kind == "note":
            pending.append(item)
            continue
        length = len(item.text)
        if body and size + length > MAX_CHUNK:
            cut = max((i for i, b in enumerate(body) if i > 0 and b.kind == "sub"), default=0)
            if item.kind == "text" and cut and body_length(body[cut:]) + length <= MAX_CHUNK:
                raws.append(Raw([n for n in notes if n[0] < cut], body[:cut], lead))
                notes = [(i - cut, n) for i, n in notes if i >= cut]
                body = body[cut:]
                lead = None
            else:
                raws.append(Raw(notes, body, lead))
                lead = body[-1] if item.kind == "text" and len(body[-1].text) <= MAX_CHUNK // 2 else None
                notes, body = [], []
            size = body_length(body) + (len(lead.text) if lead else 0)
        for note in pending:
            notes.append((len(body), note))
        pending = []
        body.append(item)
        size += length
    if body:
        raws.append(Raw(notes, body, lead))
    return raws, pending


def shorten(text: str) -> str:
    return text if len(text) <= CONTEXT_CAP else text[:CONTEXT_CAP].rstrip() + "…"


def tag_key(tag: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", tag))


def tag_range(items: list[Item]) -> str:
    tags = []
    for item in items:
        if item.kind == "sub" and item.tag and item.tag not in tags:
            tags.append(item.tag)
    if not tags:
        return ""
    if len(tags) == 1:
        return tags[0]
    increasing = all(tag_key(a) < tag_key(b) for a, b in zip(tags, tags[1:]))
    return f"{tags[0]}–{tags[-1]}" if increasing else ", ".join(tags)


def paragraph_range(items: list[Item]) -> str:
    first, last = items[0].idx, items[-1].idx
    return f"абз. {first}" if first == last else f"абз. {first}–{last}"


def pick_context(unit: Unit, raw: Raw) -> Item | None:
    first = raw.body[0]
    items = [i for i in unit.items if i.kind != "note"]
    position = next(n for n, i in enumerate(items) if i is first)
    if position == 0 or (first.kind != "sub" and first.tag is None):
        return None
    candidates = [i for i in items[:position] if i.kind in ("point", "text") and i.tag is None]
    return candidates[-1] if candidates else None


def render(header: str, raw: Raw, context: Item | None) -> str:
    out = [header]
    for _, note in sorted(raw.notes, key=lambda pair: pair[0]):
        out.append(NOTE_PREFIX + note.text)
    if context is not None and context is not raw.lead:
        out.append(shorten(context.text))
    if raw.lead is not None:
        out.append(raw.lead.text)
    out.extend(item.text for item in raw.body)
    return "\n".join(out)


def finalize(doc: Document, unit: Unit, raws: list[Raw], header_prefix: str, counter: list[int]) -> list[Chunk]:
    chunks = []
    locators = []
    for raw in raws:
        if unit.paragraphs:
            locators.append("текст документа" if len(raws) == 1 else paragraph_range(raw.body))
        else:
            tags = tag_range(raw.body)
            locators.append(unit.locator + (f", подп. {tags}" if tags else ""))
    repeated = {loc for loc in locators if locators.count(loc) > 1}
    for raw, locator in zip(raws, locators):
        if locator in repeated:
            locator += ", " + paragraph_range(raw.body)
        header = f"{header_prefix}, {locator}" + (f" ({unit.heading})" if unit.heading else "")
        spans = [item.line for item in raw.body] + [n.line for _, n in raw.notes]
        notes = "\n".join(n.text for _, n in sorted(raw.notes, key=lambda pair: pair[0]))
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{counter[0]}",
                doc_id=doc.doc_id,
                locator=locator,
                text=render(header, raw, pick_context(unit, raw)),
                notes=notes,
                start=min(s.start for s in spans),
                end=max(s.end for s in spans),
            )
        )
        counter[0] += 1
    return chunks


def attach_tail_notes(chunks: list[Chunk], carry: list[Item]) -> None:
    if not carry or not chunks:
        return
    last = chunks[-1]
    lines = [NOTE_PREFIX + n.text for n in carry]
    head, _, rest = last.text.partition("\n")
    last.text = "\n".join([head, *lines, rest]) if rest else "\n".join([head, *lines])
    last.notes = "\n".join(filter(None, [last.notes, *[n.text for n in carry]]))
    last.start = min(last.start, *(n.line.start for n in carry))
    last.end = max(last.end, *(n.line.end for n in carry))


def visible_lines(doc: Document) -> list[Line]:
    return [ln for ln in split_lines(doc.text) if not overlaps_suspicious(ln.start, ln.end, doc.suspicious)]


def run_units(doc: Document, units: list[Unit], header_prefix: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    carry: list[Item] = []
    counter = [0]
    for unit in units:
        expanded = [piece for item in unit.items for piece in split_long(item)]
        number = 0
        for item in expanded:
            if item.kind != "note":
                number += 1
                item.idx = item.idx or number
        unit.items = expanded
        raws, carry = pack(unit.items, carry)
        chunks.extend(finalize(doc, unit, raws, header_prefix, counter))
    attach_tail_notes(chunks, carry)
    return chunks


def chunk_company_document(doc: Document) -> list[Chunk]:
    lines = visible_lines(doc)
    if sum(1 for ln in lines if ITEM_RE.match(ln.text)) < 2:
        unit = Unit("текст документа", paragraphs=True)
        unit.items = [Item(ln, "text", idx=i) for i, ln in enumerate(lines, start=1)]
        return run_units(doc, [unit] if unit.items else [], doc.title)

    units: list[Unit] = []
    current = Unit("преамбула")
    header_only = False
    section_title = ""
    for ln in lines:
        item_match = ITEM_RE.match(ln.text)
        section_match = SECTION_RE.match(ln.text)
        if item_match or section_match:
            if current.items and not header_only:
                units.append(current)
            if item_match:
                current = Unit(f"п. {item_match.group(1)}", [Item(ln, "text")], heading=section_title)
                header_only = False
            else:
                section_title = ln.text
                current = Unit(f"разд. {section_match.group(1)}", [Item(ln, "text")])
                header_only = True
        else:
            current.items.append(Item(ln, "text"))
            header_only = False
    if current.items:
        units.append(current)
    return run_units(doc, units, doc.title)


def classify_law_line(text: str) -> str:
    if NOTE_RE.match(text) and len(text) <= 700:
        return "note"
    if POINT_RE.match(text):
        return "point"
    if SUB_RE.match(text):
        return "sub"
    return "text"


def chunk_law(doc: Document) -> list[Chunk]:
    lines = [ln for ln in visible_lines(doc) if not META_RE.match(ln.text)]
    article = doc.article
    part_word = "ч." if sum(1 for ln in lines if ln.text.startswith("Часть ")) > sum(1 for ln in lines if ln.text.startswith("Пункт ")) else "п."
    base = f"ст. {article}" if article else "текст"
    seen_title = not any(TITLE_RE.match(ln.text) for ln in lines)

    units: list[Unit] = []
    current = Unit(base)
    sub_tag: str | None = None
    scope = base
    for ln in lines:
        text = ln.text
        if REMARK_RE.match(text) and seen_title:
            scope = f"{base}, примечание"
            current.items.append(Item(ln, "text", tag=sub_tag))
            continue
        if not seen_title:
            title = TITLE_RE.match(text)
            if title:
                seen_title = True
                name = title.group(2).strip()
                if name.endswith(".") or len(name) > 200:
                    current.items.append(Item(ln, "text"))
            else:
                current.items.append(Item(ln, "note"))
            continue
        kind = classify_law_line(text)
        if kind == "point":
            trailing = []
            while current.items and current.items[-1].kind == "note":
                trailing.insert(0, current.items.pop())
            units.append(current)
            number = POINT_RE.match(text).group(1)
            current = Unit(f"{scope}, {part_word} {number}", trailing)
            sub_tag = None
            current.items.append(Item(ln, "point"))
            continue
        if kind == "sub":
            sub_tag = SUB_RE.match(text).group(1)
            current.items.append(Item(ln, "sub", tag=sub_tag))
            continue
        if kind == "text" and text.endswith(":"):
            sub_tag = None
        current.items.append(Item(ln, kind, tag=sub_tag if kind == "text" else None))
    units.append(current)
    heading = doc.article_name if len(doc.article_name) <= 120 and not doc.article_name.endswith(".") else ""
    for u in units:
        u.heading = heading
    return run_units(doc, [u for u in units if u.items], doc.law)


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_law(doc) if doc.is_law else chunk_company_document(doc))
    return chunks


if __name__ == "__main__":
    from config import SETTINGS
    from kb_loader import load_knowledge_base

    knowledge = load_knowledge_base(SETTINGS.data_dir)
    all_chunks = chunk_documents(knowledge)
    by_doc = {d.doc_id: d for d in knowledge}
    print(f"Документов: {len(knowledge)}, фрагментов: {len(all_chunks)}")
    rng = random.Random()
    for label, is_law in (("ДОКУМЕНТЫ КОМПАНИИ", False), ("ЗАКОНЫ", True)):
        pool = [c for c in all_chunks if by_doc[c.doc_id].is_law == is_law]
        print(f"\n===== {label}: {len(pool)} фрагментов")
        for c in rng.sample(pool, min(3, len(pool))):
            print(f"--- {c.chunk_id} [{c.locator}] {c.start}:{c.end} notes={bool(c.notes)}")
            print(c.text)
    print("\n===== ПОДОЗРИТЕЛЬНЫЕ СТРОКИ")
    for d in knowledge:
        for s, e in d.suspicious:
            print(f"{d.doc_id} [{s}:{e}] {d.text[s:e]}")
