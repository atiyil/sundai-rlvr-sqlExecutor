# sundai-rlvr-sqlExecutor

Prime Lab workspace for **text-to-SQL RLVR** using [Verifiers](https://github.com/primeintellect-ai/verifiers) and the Prime CLI. The trainable environment exposes `load_environment()` as a `SingleTurnEnv` with BIRD-style prompts, read-only SQLite execution, and a composite reward (format, syntax, execution).

**Hub environment:** [`atiyil/sundai-rlvr-sqlExecutor`](https://app.primeintellect.ai/dashboard/environments/atiyil/sundai-rlvr-sqlExecutor)

## Contents

| Path | Purpose |
|------|---------|
| `environments/sundai_rlvr_sqlExecutor/` | Environment package (dataset build, rubric, bundled SQLite DBs) |
| `configs/rl/sundai-sql.toml` | Hosted RL training config |
| `configs/rl/sundai-sql-smoke.toml` | Short RL smoke run (low step count) |
| `configs/endpoints.toml` | Endpoint aliases for evals |

## Quick commands

```bash
prime env install atiyil/sundai-rlvr-sqlExecutor
prime eval run atiyil/sundai-rlvr-sqlExecutor -m <model> -n 20 -r 3 -s
prime rl run configs/rl/sundai-sql-smoke.toml
```

See `environments/sundai_rlvr_sqlExecutor/README.md` for dataset details, rubric weights, and `load_environment` arguments.

## Requirements

Python 3.10+, dependencies managed with `uv` at the workspace and per-environment level (`pyproject.toml` files).
