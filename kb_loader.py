from __future__ import annotations

import io
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path, PurePosixPath
from typing import NamedTuple
from xml.etree import ElementTree

ARCHIVE_NAMES = ("kb.zip", "knowledge-base.zip", "knowledge_base.zip")
READABLE_SUFFIXES = {".txt", ".md", ".docx"}
LAW_TYPE = "статья закона"
STATUS_ACTIVE = "действует"
STATUS_VOID = "утратил силу"

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
DATE_WORDS = re.compile(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})(?:\s*г\b)?", re.I)
DATE_NUM = re.compile(r"\b(\d{1,2})\.(\d{2})\.(\d{4})\b")
DATE_ANY = r"(?:\d{1,2}\.\d{2}\.\d{4}|\d{1,2}\s+(?:" + "|".join(MONTHS) + r")\s+\d{4})"
PLACE_DATE = re.compile(r"^г\.\s*[^—–\-]+[—–\-]\s*(" + DATE_ANY + ")", re.I)
VALID_FROM = re.compile(r"действует\s+с\s+(" + DATE_ANY + ")", re.I)
EDITION = re.compile(r"редакция\s+от\s+(" + DATE_ANY + ")", re.I)
VALID_TO = re.compile(r"действует\s+до\s+(" + DATE_ANY + ")", re.I)
VOID_BANNER = re.compile(r"утратил\w*\s+силу", re.I)
NUMBER = re.compile(r"№\s*([^\s,;()«»]+)")
ORG = re.compile(r"(ООО|АО|ПАО|ЗАО|ОАО|ИП)\s*«([^»]+)»")
LAW_ABBR = re.compile(r"\(([^()]*?\bРФ)\)")
LAW_FZ = re.compile(r"\bN\s*(\d+-ФЗ)")
ARTICLE_LINE = re.compile(r"^Статья\s+(\d+(?:\.\d+)*)\.\s*(.*)$")
ARTICLE_IN_SOURCE = re.compile(r"Статья\s+(\d+(?:\.\d+)*)\b")
EXPORTED = re.compile(r"Дата выгрузки:\s*(\d{1,2}\.\d{2}\.\d{4})")

TYPE_RULES = (
    ("дополнительное соглашение", "доп. соглашение"),
    ("доп соглашение", "доп. соглашение"),
    ("доп. соглашение", "доп. соглашение"),
    ("договор", "договор"),
    ("регламент", "регламент"),
    ("служебная записка", "служебная записка"),
    ("письмо", "письмо"),
    ("приказ", "приказ"),
    ("положение", "положение"),
    ("политика", "политика"),
    ("инструкция", "инструкция"),
    ("протокол", "протокол"),
    ("приложение", "приложение"),
    ("акт", "акт"),
)

SUSPICIOUS_CI = re.compile(
    "|".join(
        [
            r"\bассистент\w*",
            r"\bнейросет\w*",
            r"\bязыков\w+\s+модел\w*",
            r"\bчат-?бот\w*",
            r"\bигнорир\w+",
            r"\bне\s+упоминай\w*",
            r"\bотвечай\b",
            r"\bзабудь\b",
            r"\bпромпт\w*",
            r"\bignore\b",
            r"\bsystem\s+prompt\b",
            r"\bprompt\b",
            r"\bnew\s+instructions?\b",
            r"\bjailbreak\b",
            r"\bновые\s+инструкции\b",
        ]
    ),
    re.I,
)
SUSPICIOUS_CS = re.compile(r"\b(?:ИИ|AI|LLM)\b")


class Line(NamedTuple):
    start: int
    end: int
    text: str


@dataclass
class Document:
    doc_id: str
    file_name: str
    title: str
    doc_type: str
    number: str
    date: str
    parties: str
    status: str
    valid_from: str
    valid_to: str
    relates_to: list[str] = field(default_factory=list)
    text: str = ""
    suspicious: list[tuple[int, int]] = field(default_factory=list)
    law: str = ""
    article: str = ""
    article_name: str = ""
    exported: str = ""

    @property
    def is_law(self) -> bool:
        return self.doc_type == LAW_TYPE


def split_lines(text: str) -> list[Line]:
    result = []
    position = 0
    for raw in text.split("\n"):
        stripped = raw.strip()
        if stripped:
            start = position + len(raw) - len(raw.lstrip())
            result.append(Line(start, start + len(stripped), stripped))
        position += len(raw) + 1
    return result


def is_suspicious_line(line: str) -> bool:
    return bool(SUSPICIOUS_CI.search(line) or SUSPICIOUS_CS.search(line))


def mark_suspicious(text: str) -> list[tuple[int, int]]:
    return [(ln.start, ln.end) for ln in split_lines(text) if is_suspicious_line(ln.text)]


def overlaps_suspicious(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < s_end and s_start < end for s_start, s_end in spans)


def visible_text(doc: Document) -> str:
    if not doc.suspicious:
        return doc.text
    pieces = []
    position = 0
    for start, end in sorted(doc.suspicious):
        pieces.append(doc.text[position:start])
        position = end
    pieces.append(doc.text[position:])
    return "".join(pieces)


def find_archive(data_dir: Path) -> Path:
    directories = [data_dir, Path(__file__).resolve().parent, Path.cwd()]
    for directory in directories:
        if not directory.is_dir():
            continue
        by_name = {p.name.lower(): p for p in directory.iterdir() if p.is_file()}
        for name in ARCHIVE_NAMES:
            if name in by_name:
                return by_name[name]
    if data_dir.is_dir():
        archives = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".zip")
        if len(archives) == 1:
            return archives[0]
    raise FileNotFoundError("Архив базы знаний не найден")


def fix_zip_name(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1251", errors="replace")


def read_docx(raw: bytes) -> str:
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in root.iter(ns + "p"):
        parts = []
        for element in paragraph.iter():
            if element.tag == ns + "t":
                parts.append(element.text or "")
            elif element.tag in (ns + "tab", ns + "br"):
                parts.append(" ")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")


def read_documents(archive: Path) -> list[tuple[str, str]]:
    result = []
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(fix_zip_name(info).replace("\\", "/")).parts
            base = parts[-1] if parts else ""
            if "__MACOSX" in parts or base.startswith(".") or base.startswith("~$"):
                continue
            suffix = PurePosixPath(base).suffix.lower()
            if suffix not in READABLE_SUFFIXES:
                continue
            try:
                raw = bundle.read(info)
                text = read_docx(raw) if suffix == ".docx" else decode_bytes(raw)
            except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError, RuntimeError) as exc:
                print(f"Предупреждение: файл {base} не прочитан ({type(exc).__name__})", file=sys.stderr)
                continue
            text = normalize_text(text)
            if text.strip():
                result.append((base, text))
    return sorted(result, key=lambda item: item[0])


def to_iso(value: str) -> str:
    match = DATE_NUM.search(value)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
    else:
        match = DATE_WORDS.search(value)
        if not match:
            return ""
        day, month, year = int(match.group(1)), MONTHS[match.group(2).lower()], int(match.group(3))
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return ""


def format_dmy(iso: str) -> str:
    return ".".join(reversed(iso.split("-"))) if iso else ""


def day_before(iso: str) -> str:
    if not iso:
        return ""
    return (date.fromisoformat(iso) - timedelta(days=1)).isoformat()


def first_date(lines: list[str]) -> str:
    best = None
    for line in lines:
        for pattern in (DATE_NUM, DATE_WORDS):
            match = pattern.search(line)
            if match and (best is None or match.start() < best[0]):
                best = (match.start(), match.group(0))
        if best:
            return to_iso(best[1])
    return ""


def cap_name(name: str) -> str:
    if not name.isupper():
        return name
    return re.sub(r"\w+", lambda m: m.group(0).capitalize(), name)


def sentence_case(line: str) -> str:
    if not line.isupper():
        return line
    protected = re.compile(r"(№\s*\S+|(?:ООО|АО|ПАО|ЗАО|ОАО|ИП)\s*«[^»]*»)")
    out = []
    for index, part in enumerate(protected.split(line)):
        if index % 2:
            out.append(re.sub(r"«([^»]*)»", lambda m: "«" + cap_name(m.group(1)) + "»", part))
        else:
            out.append(part.lower())
    joined = "".join(out)
    return joined[:1].upper() + joined[1:]


def file_stem(file_name: str) -> str:
    stem = PurePosixPath(file_name).stem
    return stem.split("__", 1)[-1].strip()


def is_law_file(file_name: str, text: str) -> bool:
    if "__" in file_name and file_name.split("__", 1)[0].lower().startswith("закон"):
        return True
    head = "\n".join(text.split("\n")[:6])
    return "Дата выгрузки:" in head and "Источник:" in head


def detect_type(title: str, stem: str) -> str:
    low_title, low_stem = title.lower(), stem.lower()
    for key, value in TYPE_RULES:
        if low_title.startswith(key) or low_stem.startswith(key):
            return value
    for key, value in TYPE_RULES:
        if key in low_stem:
            return value
    return "документ"


def pick_title(head: list[str], stem: str) -> str:
    for index, line in enumerate(head):
        if VOID_BANNER.search(line) or not line.isupper() or re.match(r"(ООО|АО|ПАО|ЗАО|ОАО|ИП)\b", line):
            continue
        title = sentence_case(line)
        if index + 1 < len(head) and re.match(r"к\s", head[index + 1]):
            title += " " + head[index + 1]
        return title
    return stem


def find_parties(head: list[str]) -> str:
    seen = {}
    for line in head:
        for form, name in ORG.findall(line):
            seen.setdefault(name.lower(), f"{form} «{cap_name(name)}»")
    return "; ".join(seen.values())


def build_law_card(file_name: str, text: str) -> Document:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    source = next((ln for ln in lines[:6] if ln.startswith("Источник:")), "")
    article = ""
    article_name = ""
    for ln in lines:
        match = ARTICLE_LINE.match(ln)
        if match:
            article, article_name = match.group(1), match.group(2).strip()
            break
    if not article:
        match = ARTICLE_IN_SOURCE.search(source)
        article = match.group(1) if match else ""
    abbr = LAW_ABBR.search(source)
    fz = LAW_FZ.search(source)
    law = abbr.group(1).strip() if abbr else (fz.group(1) if fz else file_stem(file_name))
    shown = article_name if len(article_name) <= 120 else article_name[:120].rstrip() + "…"
    title = f"{law}, ст. {article}" + (f". {shown}" if shown else "") if article else f"{law}. {shown}".strip(". ")
    exported = EXPORTED.search(text)
    status = STATUS_VOID if VOID_BANNER.search(file_stem(file_name)) else STATUS_ACTIVE
    return Document(
        doc_id="",
        file_name=file_name,
        title=title,
        doc_type=LAW_TYPE,
        number=article,
        date="",
        parties="",
        status=status,
        valid_from="",
        valid_to="",
        law=law,
        article=article,
        article_name=article_name,
        exported=to_iso(exported.group(1)) if exported else "",
    )


def build_company_card(file_name: str, text: str) -> Document:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    head = lines[:8]
    stem = file_stem(file_name)
    banner = next((ln for ln in head[:2] if VOID_BANNER.search(ln)), "")
    if not banner and VOID_BANNER.search(stem):
        banner = stem
    body_head = [ln for ln in head if ln != banner]

    title = pick_title(body_head, stem)
    edition = next((EDITION.search(ln) for ln in body_head if EDITION.search(ln)), None)
    if edition and "ред." not in title:
        title += f" (ред. от {format_dmy(to_iso(edition.group(1)))})"

    number = ""
    for ln in body_head:
        if re.search(r"приказ", ln, re.I):
            continue
        match = NUMBER.search(ln)
        if match:
            number = match.group(1).rstrip(".")
            break
    if not number:
        match = re.search(r"([A-Za-zА-Яа-я]{1,4}-\d+)-(\d{4})", stem)
        if match:
            number = f"{match.group(1)}/{match.group(2)}"

    doc_date = ""
    for ln in body_head:
        match = PLACE_DATE.match(ln)
        if match:
            doc_date = to_iso(match.group(1))
            break
    if not doc_date:
        doc_date = first_date(body_head)

    valid_from = ""
    for ln in body_head:
        match = VALID_FROM.search(ln)
        if match:
            valid_from = to_iso(match.group(1))
            break
    if not valid_from and edition:
        valid_from = to_iso(edition.group(1))
    if not valid_from:
        valid_from = doc_date

    valid_to = ""
    status = STATUS_ACTIVE
    if banner:
        status = STATUS_VOID
        valid_to = day_before(first_date([banner]))
    else:
        match = VALID_TO.search(text)
        if match:
            valid_to = to_iso(match.group(1))

    return Document(
        doc_id="",
        file_name=file_name,
        title=title,
        doc_type=detect_type(title, stem),
        number=number,
        date=doc_date,
        parties=find_parties(head[:6]),
        status=status,
        valid_from=valid_from,
        valid_to=valid_to,
    )


def build_card(file_name: str, text: str) -> Document:
    card = build_law_card(file_name, text) if is_law_file(file_name, text) else build_company_card(file_name, text)
    card.text = text
    card.suspicious = mark_suspicious(text)
    return card


def number_pattern(number: str) -> re.Pattern | None:
    if not number or not (re.search(r"[^\W\d_]", number) or "/" in number):
        return None
    pieces = [re.escape(p) for p in re.split(r"[-/]", number)]
    return re.compile(r"(?<![\w\-/])" + r"[-/‑–]".join(pieces) + r"(?![\w\-/])", re.I)


def title_tokens(title: str) -> list[str]:
    cut = re.split(r"\bООО\b|\(", title, maxsplit=1)[0].lower()
    return [word[:5] for word in re.findall(r"[а-яёa-z]{4,}", cut)]


def mentions(source: Document, target: Document) -> bool:
    if target.number:
        pattern = number_pattern(target.number)
        return bool(pattern and pattern.search(source.text))
    tokens = title_tokens(target.title)
    if len(tokens) < 2:
        return False
    low = source.text.lower()
    return all(re.search(r"\b" + re.escape(token), low) for token in tokens)


def link_documents(docs: list[Document]) -> None:
    links = {doc.doc_id: set() for doc in docs}
    company = [doc for doc in docs if not doc.is_law]
    for source in company:
        for target in company:
            if source is not target and mentions(source, target):
                links[source.doc_id].add(target.doc_id)
                links[target.doc_id].add(source.doc_id)
    for doc in docs:
        doc.relates_to = sorted(links[doc.doc_id])


def load_knowledge_base(data_dir: Path) -> list[Document]:
    archive = find_archive(data_dir)
    docs = []
    for file_name, text in read_documents(archive):
        try:
            docs.append(build_card(file_name, text))
        except Exception as exc:
            print(f"Предупреждение: карточка {file_name} не построена ({type(exc).__name__})", file=sys.stderr)
    for index, doc in enumerate(docs, start=1):
        doc.doc_id = f"d{index:02d}"
    link_documents(docs)
    return docs


if __name__ == "__main__":
    from config import SETTINGS

    knowledge = load_knowledge_base(SETTINGS.data_dir)
    print(f"Документов: {len(knowledge)}")
    for item in knowledge:
        print(
            f"{item.doc_id} | {item.doc_type} | {item.title} | №{item.number or '-'} | {item.date or '-'} | "
            f"{item.status} | {item.valid_from or '-'}..{item.valid_to or '-'} | {item.parties or '-'} | "
            f"связи: {','.join(item.relates_to) or '-'}"
        )
    for item in knowledge:
        for start, end in item.suspicious:
            print(f"ПОДОЗРИТЕЛЬНО {item.doc_id} [{start}:{end}] {item.text[start:end]}")
