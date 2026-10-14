# Contributing

Thanks for your interest in **BijliPredict AI**. This is a small, self-contained
forecasting project — contributions that keep it reproducible and dependency-light
are welcome.

## Setup

```bash
uv sync --locked --extra dev --extra serve --extra validate
source .venv/bin/activate        # optional; or prefix commands with `uv run`
```

Python **3.12+** (CI runs 3.12 and 3.14). PyTorch is pinned to the CPU wheel
index, so no GPU is required.

## Before you push

```bash
make lint     # ruff check .
make test     # pytest
```

Both must pass; CI runs the same on every PR (plus a Docker build & `/health`
smoke test).

## Conventions

- **Style:** `ruff` (line length 100, target py312; lint rules `E,F,W,I,UP,B`).
  Keep imports sorted and prefer modern typing (`X | None`, `list[...]`).
- **No comments** unless they explain *why* something non-obvious is done.
- **Determinism:** anything that trains or generates data must accept a `--seed`
  and stay reproducible; `make train` passes `--deterministic`.
- **Data contract:** a training CSV needs a date column and a consumption column
  (`date` / `consumption_kwh` by default). Multi-household data uses the long
  format `date,household,consumption_kwh`.
- **Tests:** add or extend a test under `tests/` for any behaviour change. Optional
  extras (FastAPI, Streamlit, statsmodels, pandera) are guarded with
  `pytest.importorskip`, so the suite passes with a minimal install.

## Changing the model or features

Because the pipeline is a set of composable scripts, a design change should be
justified with numbers, not intuition. Run the shared experiments and paste the
result into the PR:

```bash
python generate_data.py
python ablation.py --models gru,lstm --features none,calendar,weather --epochs 60
python tune.py --trials 16 --epochs 60 --features calendar
```

`ablation.py` reports each architecture × feature-set combination against a
seasonal-naive floor; `tune.py` keeps the configuration with the lowest
validation loss. Both reuse the exact training loop as `train.py`
(`train.fit_model`), so results are directly comparable.

## Commits & PRs

- Keep commits focused; a short imperative subject line (`fix: …`, `feat: …`).
- Update `README.md` (and `config.toml` defaults) when you change the CLI, the
  output files or the measured metrics.
- Don't commit secrets, trained artifacts (`artifacts/`), or run outputs
  (`outputs/`) — they are git-ignored.

## License

By contributing you agree that your work is released under the project's
[MIT License](LICENSE).
