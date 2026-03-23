# sundai-rlvr-sqlExecutor

**Description:** Text-to-SQL RLVR with BIRD-style examples, SQLite execution rewards, and structured think/sql outputs.

**Prime Intellect Hub:** [`atiyil/sundai-rlvr-sqlExecutor`](https://app.primeintellect.ai/dashboard/environments/atiyil/sundai-rlvr-sqlExecutor)

**Tags:** single-turn, text-to-sql, sqlite, bird, train, eval

### Overview

| | |
|--|--|
| **Environment ID** | `sundai-rlvr-sqlExecutor` (Hub: `atiyil/sundai-rlvr-sqlExecutor`) |
| **Task** | Single-turn: system prompt with schema + question; model replies with reasoning and one SQL query |
| **Parser** | `verifiers.XMLParser` for `<sql>`; reasoning detected via paired `think` tags or paired `redacted_reasoning` tags (see system prompt in code) |

### Datasets

- **Bundled mode (default):** `birdsql/bird_mini_dev` split `mini_dev_sqlite`, filtered to bundled `db_id`s (`superhero`, `student_club`, `toxicology`). Schema text comes from each DB’s `sqlite_master`. Three small SQLite files ship under `databases/` (see `databases/README.md` for provenance).
- **Fallback:** `hf_dataset` / `hf_config` train split (defaults toward `xlangai/BIRD`, with fallback to `xu3kev/BIRD-SQL-data-train` if needed). Requires `BIRD_DB_ROOT` or compatible layout for execution paths.

Train/eval row counts follow `train_rows` and `dev_rows`; if the filtered mini-dev set is smaller than `train_rows + dev_rows`, splits are shrunk proportionally.

### Output format

Models should emit:

1. Non-empty reasoning inside either a `think` … `/think` block or a `redacted_reasoning` … `/redacted_reasoning` block (XML-style, as in `SYSTEM_PROMPT_TEMPLATE`).
2. Exactly one statement inside `<sql>...</sql>`.

Format reward (0.5) requires non-empty reasoning and a parsed SQL string; syntax (0.25) uses `EXPLAIN` on the target DB; execution (1.0) compares multiset query results to the gold SQL (timeout via `query_timeout_s`).

### Rubric (weights are equal; summed for total reward)

| Component | Weight in rubric | Notes |
|-----------|------------------|--------|
| `format_reward` | 1.0 | `think`/`redacted_reasoning` + `<sql>` |
| `syntax_reward` | 1.0 | SQLite `EXPLAIN` |
| `execution_reward` | 1.0 | Multiset equality vs gold |

Reported training metrics often include `metrics/format_reward`, `metrics/syntax_reward`, `metrics/execution_reward` (per-component means) and scalar `reward` (weighted sum of the three).

### Quickstart

```bash
prime env install atiyil/sundai-rlvr-sqlExecutor
prime eval run atiyil/sundai-rlvr-sqlExecutor -m gpt-4.1-mini -n 20 -r 3 -s
```

Pass environment kwargs as JSON:

```bash
prime eval run atiyil/sundai-rlvr-sqlExecutor -m gpt-4.1-mini -n 20 -r 3 -a '{"train_rows": 120, "dev_rows": 30}'
```

### `load_environment` arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `hf_dataset` | str | `xlangai/BIRD` | HF dataset id when not using bundled DBs |
| `hf_config` | str \| None | `None` | HF config name if required |
| `train_rows` | int | `400` | Target train examples |
| `dev_rows` | int | `100` | Target eval examples |
| `seed` | int | `42` | Shuffle seed |
| `bird_db_root` | str \| None | `None` | Root for BIRD-style folder layout; also read from `BIRD_DB_ROOT` |
| `query_timeout_s` | float | `2.0` | Execution comparison timeout |
| `use_bundled_databases` | bool | `True` | Use shipped SQLite + `bird_mini_dev` filter |
| `bundled_db_ids` | tuple \| None | `("superhero", "student_club", "toxicology")` | Subset of DBs to keep when bundled |

### Publishing

After local checks:

```bash
prime env push --path ./environments/sundai_rlvr_sqlExecutor
```
