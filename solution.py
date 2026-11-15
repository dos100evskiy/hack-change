from __future__ import annotations

import time

STARTED = time.monotonic()

import csv
import os
import re
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import llm_client
from config import SETTINGS, env_float

FULL_SECONDS = env_float("TIME_FULL_SECONDS", 15 * 60, 0.0, 19 * 60)
STOP_SECONDS = env_float("TIME_STOP_SECONDS", 18 * 60, 1.0, 19 * 60)
llm_client.BUDGET.full_until = STARTED + FULL_SECONDS
llm_client.BUDGET.stop_at = STARTED + STOP_SECONDS

import pipeline

FALLBACK = pipeline.FALLBACK
OUTPUT = Path("submission.csv")
SAVE_EVERY = 10
WAIT_STEP = 5.0
SPACES = re.compile(r"\s+")


def log(message: str) -> None:
    print(f"[{time.monotonic() - STARTED:7.1f} с] {message}", flush=True)


def find_questions() -> Path:
    name = "questions.csv"
    roots = (SETTINGS.data_dir, Path(__file__).resolve().parent / SETTINGS.data_dir, Path.cwd())
    for root in roots:
        if (root / name).is_file():
            return root / name
    raise FileNotFoundError(name)


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {(name or "").strip().lower(): name for name in reader.fieldnames or []}
        if "id" not in fields or "question" not in fields:
            raise ValueError("questions.csv: нужны колонки id и question")
        rows = [(str(row[fields["id"]] or "").strip(), str(row[fields["question"]] or "")) for row in reader]
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for qid, question in rows:
        if qid in seen:
            log(f"Повтор id пропущен: {qid}")
            continue
        seen.add(qid)
        result.append((qid, question))
    return result


def clean(answer: object) -> str:
    text = SPACES.sub(" ", str(answer or "")).strip()[: pipeline.MAX_ANSWER].strip()
    return text or FALLBACK


def save(ids: list[str], answers: list[str]) -> None:
    temp = OUTPUT.with_name(OUTPUT.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        writer.writerows(zip(ids, (clean(answer) for answer in answers)))
    for attempt in range(5):
        try:
            os.replace(temp, OUTPUT)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5)


def work(question: str) -> dict:
    full = llm_client.BUDGET.full()
    try:
        result = pipeline.answer_question(question, use_reranker=full, self_check=full)
    except Exception as exc:
        result = {"answer": FALLBACK, "status": "error", "debug": {"error": type(exc).__name__}}
    result["full"] = full
    return result


def tally(result: dict, stats: Counter) -> None:
    debug = result.get("debug") or {}
    flags = debug.get("flags") or {}
    stats["status " + str(result.get("status"))] += 1
    stats["сокращённый режим"] += not result.get("full", True)
    stats["без реранкера"] += bool(flags.get("reranker_off"))
    stats["без эмбеддингов"] += bool(flags.get("embeddings_off"))
    stats["JSON не разобран"] += debug.get("json_ok") is False
    if debug.get("error"):
        stats["ошибка " + str(debug["error"])] += 1


def report(stats: Counter, answers: list[str]) -> None:
    stats["запасных ответов"] = sum(answer == FALLBACK for answer in answers)
    for name, value in sorted(stats.items()):
        if value:
            log(f"  {name}: {value}")
    for name, value in sorted(llm_client.STATS.items()):
        log(f"  запросы {name}: {value}")


def run(questions: list[tuple[str, str]], ids: list[str], answers: list[str]) -> bool:
    stats: Counter = Counter()
    pool = ThreadPoolExecutor(max_workers=SETTINGS.max_workers)
    futures: dict[Future, int] = {pool.submit(work, question): i for i, (_, question) in enumerate(questions)}
    pending = set(futures)
    done_count = 0
    while pending:
        left = llm_client.BUDGET.stop_at - time.monotonic()
        if left <= 0:
            break
        finished, pending = wait(pending, timeout=min(WAIT_STEP, left), return_when=FIRST_COMPLETED)
        for future in finished:
            result = future.result()
            answers[futures[future]] = clean(result.get("answer"))
            tally(result, stats)
            done_count += 1
            if done_count % SAVE_EVERY == 0:
                save(ids, answers)
                log(f"Готово {done_count}/{len(questions)}")
    pool.shutdown(wait=False, cancel_futures=True)
    if pending:
        stats["не дождались (лимит времени)"] = len(pending)
        log(f"Лимит времени: не дождались {len(pending)} ответов, у них запасной ответ")
    save(ids, answers)
    log(f"Готово {done_count}/{len(questions)}; деградации и счётчики:")
    report(stats, answers)
    return not pending


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    try:
        questions = load_questions(find_questions())
    except Exception as exc:
        log(f"Не удалось прочитать questions.csv: {type(exc).__name__}: {exc}")
        return 1
    ids = [qid for qid, _ in questions]
    answers = [FALLBACK] * len(questions)
    save(ids, answers)
    log(f"Вопросов: {len(questions)}; потоков: {SETTINGS.max_workers}; запасной submission.csv записан")
    try:
        engine = pipeline.get_engine()
        retriever = engine.retriever
        log(
            f"Документов: {len(engine.docs)}; фрагментов: {len(engine.chunks)}; "
            f"эмбеддинги: {'выкл' if retriever.embeddings_off else 'вкл'}"
        )
    except Exception as exc:
        log(f"База знаний не загружена: {type(exc).__name__}; все ответы запасные")
        log(f"Итого: {time.monotonic() - STARTED:.1f} с")
        return 0
    completed = run(questions, ids, answers)
    log(f"Итого: {time.monotonic() - STARTED:.1f} с; submission.csv: {len(ids)} строк")
    if not completed:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
