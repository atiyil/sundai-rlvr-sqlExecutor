import asyncio
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sqlite3
import verifiers as vf
from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetNotFoundError

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_SQL_BLOCK = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.DOTALL | re.IGNORECASE)

BUNDLED_DB_IDS: tuple[str, ...] = ("superhero", "student_club", "toxicology")

SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert Text-to-SQL assistant. Given a SQLite schema and a question, write a single correct SELECT query.\n\n"
    "Database schema:\n{schema}\n\n"
    "Question:\n{question}\n\n"
    "Respond using this exact structure:\n"
    "1. Put your reasoning inside \u003Cthink\u003E ... \u003C/think\u003E\n"
    "2. Put exactly one SQL statement inside <sql>...</sql>"
)


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _bundled_sqlite_path(db_id: str) -> Path:
    return _package_dir() / "databases" / db_id / f"{db_id}.sqlite"


def _bundled_databases_ready(db_ids: tuple[str, ...]) -> bool:
    return all(_bundled_sqlite_path(d).is_file() for d in db_ids)


def _ddl_from_sqlite(db_path: str) -> str:
    conn = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') "
            "AND sql IS NOT NULL ORDER BY name"
        )
        return "\n".join(row[0] for row in cur.fetchall() if row[0])
    finally:
        conn.close()


def _assistant_content(completion: vf.Messages) -> str:
    if not completion:
        return ""
    last = completion[-1]
    if isinstance(last, dict):
        c = last.get("content", "")
    else:
        c = getattr(last, "content", "") or ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        return "".join(parts)
    return str(c)


def _normalize_sql(sql: str | None) -> str | None:
    if not sql:
        return None
    s = sql.strip()
    if not s:
        return None
    if ";" in s:
        s = s.split(";")[0].strip()
    return s or None


def _parse_think_sql(text: str) -> tuple[str | None, str | None]:
    tm = _THINK_BLOCK.search(text)
    sm = _SQL_BLOCK.search(text)
    think = tm.group(1).strip() if tm else None
    sql = sm.group(1).strip() if sm else None
    sql = _normalize_sql(sql)
    if think == "":
        think = None
    return think, sql


def _resolve_bird_sqlite(db_id: str, bird_db_root: str | None) -> str | None:
    bundled = _bundled_sqlite_path(db_id)
    if bundled.is_file():
        return str(bundled.resolve())
    if not bird_db_root:
        return None
    root = Path(bird_db_root).expanduser()
    candidates = [
        root / "train" / "train_databases" / db_id / f"{db_id}.sqlite",
        root / "dev" / "dev_databases" / db_id / f"{db_id}.sqlite",
        root / "train_databases" / db_id / f"{db_id}.sqlite",
        root / "dev_databases" / db_id / f"{db_id}.sqlite",
    ]
    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    return None


def _sqlite_ro_uri(abs_path: str) -> str:
    return f"file:{Path(abs_path).resolve().as_posix()}?mode=ro"


def _multiset_from_query(db_path: str, sql: str) -> Counter:
    uri = _sqlite_ro_uri(db_path)
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    try:
        conn.execute("PRAGMA query_only = ON")
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return Counter(rows)
    finally:
        conn.close()


def _explain_ok(db_path: str, sql: str) -> bool:
    try:
        uri = _sqlite_ro_uri(db_path)
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            conn.execute("PRAGMA query_only = ON")
            conn.execute(f"EXPLAIN {sql}")
            return True
        finally:
            conn.close()
    except Exception:
        return False


def _compare_execution(db_path: str, pred_sql: str, gold_sql: str, timeout_s: float) -> float:
    def work() -> bool:
        c_pred = _multiset_from_query(db_path, pred_sql)
        c_gold = _multiset_from_query(db_path, gold_sql)
        return c_pred == c_gold

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(work)
            return 1.0 if fut.result(timeout=timeout_s) else 0.0
    except Exception:
        return 0.0


def _load_bird_table(
    hf_dataset: str,
    hf_config: str | None,
    split: str,
) -> Dataset:
    try:
        if hf_config is not None:
            return load_dataset(hf_dataset, hf_config, split=split)
        return load_dataset(hf_dataset, split=split)
    except DatasetNotFoundError:
        return load_dataset("xu3kev/BIRD-SQL-data-train", split=split)


def _train_dev_ranges(n: int, train_rows: int, dev_rows: int) -> tuple[range, range]:
    if n <= 0:
        raise ValueError("empty dataset after filter")
    need = train_rows + dev_rows
    if n >= need:
        return range(0, train_rows), range(train_rows, train_rows + dev_rows)
    n_train = min(train_rows, max(1, int(n * 0.85)))
    n_dev = min(dev_rows, n - n_train)
    if n_dev < 1:
        n_dev = 1
        n_train = n - n_dev
    return range(0, n_train), range(n_train, n_train + n_dev)


def _build_rows_xu(ds: Dataset) -> list[dict]:
    rows: list[dict] = []
    for item in ds:
        rows.append(
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_TEMPLATE.format(
                            schema=item["schema"],
                            question=item["question"],
                        ),
                    },
                ],
                "answer": item["SQL"],
                "info": json.dumps({"db_id": item["db_id"]}),
            }
        )
    return rows


def _build_rows_mini_dev(ds: Dataset, paths: dict[str, str]) -> list[dict]:
    schema_cache: dict[str, str] = {}
    rows: list[dict] = []
    for item in ds:
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = _ddl_from_sqlite(paths[db_id])
        rows.append(
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_TEMPLATE.format(
                            schema=schema_cache[db_id],
                            question=item["question"],
                        ),
                    },
                ],
                "answer": item["SQL"],
                "info": json.dumps({"db_id": db_id}),
            }
        )
    return rows


def load_environment(
    hf_dataset: str = "xlangai/BIRD",
    hf_config: str | None = None,
    train_rows: int = 400,
    dev_rows: int = 100,
    seed: int = 42,
    bird_db_root: str | None = None,
    query_timeout_s: float = 2.0,
    use_bundled_databases: bool = True,
    bundled_db_ids: tuple[str, ...] | None = None,
    **kwargs,
) -> vf.SingleTurnEnv:
    bird_db_root = bird_db_root or os.environ.get("BIRD_DB_ROOT")
    db_ids = bundled_db_ids if bundled_db_ids is not None else BUNDLED_DB_IDS

    if use_bundled_databases and _bundled_databases_ready(db_ids):
        raw = load_dataset("birdsql/bird_mini_dev", split="mini_dev_sqlite")
        allow = set(db_ids)
        raw = raw.filter(lambda x: x["db_id"] in allow)
        paths = {d: str(_bundled_sqlite_path(d).resolve()) for d in db_ids}
        shuffled = raw.shuffle(seed=seed)
        r_train, r_dev = _train_dev_ranges(len(shuffled), train_rows, dev_rows)
        train_ds = Dataset.from_list(
            _build_rows_mini_dev(shuffled.select(list(r_train)), paths)
        )
        eval_ds = Dataset.from_list(
            _build_rows_mini_dev(shuffled.select(list(r_dev)), paths)
        )
    else:
        raw_train = _load_bird_table(hf_dataset, hf_config, "train")
        raw_train = raw_train.shuffle(seed=seed)
        n_train = min(train_rows, len(raw_train))
        train_ds = Dataset.from_list(_build_rows_xu(raw_train.select(range(n_train))))

        start = n_train
        end = start + dev_rows
        if end <= len(raw_train):
            dev_slice = raw_train.select(range(start, end))
        else:
            extra = _load_bird_table(hf_dataset, hf_config, "train")
            extra = extra.shuffle(seed=seed + 1)
            dev_slice = extra.select(range(min(dev_rows, len(extra))))

        eval_ds = Dataset.from_list(_build_rows_xu(dev_slice))

    parser = vf.XMLParser(["sql"], answer_field="sql")

    async def format_reward(completion, state, **kwargs) -> float:
        text = _assistant_content(completion)
        think, sql = _parse_think_sql(text)
        parsed_xml = parser.parse(text)
        sql_xml = getattr(parsed_xml, "sql", None) if parsed_xml else None
        state["pred_sql"] = _normalize_sql(sql or (sql_xml.strip() if sql_xml else None))
        if think and state["pred_sql"]:
            return 0.5
        return 0.0

    async def syntax_reward(completion, state, bird_db_root, info, **kwargs) -> float:
        sql = state.get("pred_sql")
        if not sql:
            return 0.0
        info_d = info if isinstance(info, dict) else json.loads(info) if info else {}
        db_id = info_d.get("db_id", "")
        path = _resolve_bird_sqlite(str(db_id), bird_db_root)
        if not path:
            return 0.0
        ok = await asyncio.to_thread(_explain_ok, path, sql)
        return 0.25 if ok else 0.0

    async def execution_reward(
        completion, state, bird_db_root, answer, info, **kwargs
    ) -> float:
        sql = state.get("pred_sql")
        if not sql:
            return 0.0
        info_d = info if isinstance(info, dict) else json.loads(info) if info else {}
        db_id = info_d.get("db_id", "")
        path = _resolve_bird_sqlite(str(db_id), bird_db_root)
        if not path:
            return 0.0
        gold = _normalize_sql(str(answer) if answer is not None else None)
        if not gold:
            return 0.0
        return await asyncio.to_thread(
            _compare_execution, path, sql, gold, query_timeout_s
        )

    rubric = vf.Rubric(
        funcs=[format_reward, syntax_reward, execution_reward],
        weights=[1.0, 1.0, 1.0],
        parser=parser,
    )
    rubric.add_class_object("bird_db_root", bird_db_root)

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=rubric,
        parser=parser,
        **kwargs,
    )
