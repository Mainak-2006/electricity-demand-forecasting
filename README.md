# BijliPredict AI

Electricity demand forecasting for Indian households — a local, ML-only MVP.

A GRU neural network (with an LSTM comparison model) is trained on historical
daily household electricity consumption and forecasts the **next 7 days** of
usage, along with an estimated energy bill in Indian Rupees (₹).

Everything runs locally, but the trained model is also exposed three ways:
a unified `bijli` CLI, a **FastAPI** JSON service (`uvicorn api:app`), and a
**Streamlit** dashboard (`streamlit run app.py`). A `Dockerfile` packages the
API into a slim CPU-only image.

---

## Pipeline

```text
Historical dataset (CSV)
        |
        v
Preprocessing (clean, gap-fill, chronological split, train-only scaling)
        |
        v
Sliding windows: 30-day input  ->  7-day target
        |
        v
GRU(64) -> Dropout(0.2) -> GRU(32) -> Dense(32, ReLU) -> Dense(7)
        |
        v
Evaluation (MAE / RMSE / WAPE vs seasonal-naive baseline)
        |
        v
outputs/predictions.csv + outputs/forecast.png + outputs/evaluation.json
        |
        v
Cost forecast + monthly / 3-month projection + outputs/bijli_report.xlsx (+ .html)
```

---

## Requirements

- Python 3.12+ (tested on 3.14)
- [uv](https://docs.astral.sh/uv/) (recommended) or plain `venv` + `pip`

Optional extras: `serve` (JSON API), `ui` (Streamlit dashboard), `weather`,
`stats` (SARIMAX/ETS baselines) and `validate` (pandera schema layer).

## Setup

```bash
# with uv (recommended) — creates .venv and installs locked deps + dev tools
uv sync --extra dev

# or plain venv + pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A `Makefile` wraps the common tasks (`make install | data | train | eval | predict | report | test | lint`).

Activate the environment for the commands below:

```bash
source .venv/bin/activate
```

## Quickstart (runs fully offline)

```bash
# 1. Create a reproducible synthetic Indian household dataset (3 years of daily kWh).
python generate_data.py

# 2. Train the GRU model (saves artifacts/best_model.pt + artifacts/scaler.joblib).
python train.py --model gru --epochs 100

# 3. Evaluate on the held-out chronological test period.
python evaluate.py

# 4. Forecast the next 7 days, estimate the bill, and build the Excel + HTML reports.
python predict.py --input data/sample_input.csv --rate 7
```

The LSTM comparison model uses the same interface:

```bash
python train.py --model lstm --epochs 100
python evaluate.py
```

---

## Unified CLI

The scripts are also wrapped behind one entry point, installed as `bijli`
(`[project.scripts] bijli = "cli:main"`):

```bash
bijli data          # generate the synthetic dataset
bijli train         # train the forecaster (deterministic by default)
bijli ablation      # compare architectures against feature sets on one split
bijli tune          # random hyperparameter search (best validation loss)
bijli evaluate      # metrics on the held-out test split
bijli predict       # 7-day forecast + bill + 90-day projection + reports
bijli report        # rebuild the Excel + HTML reports from outputs/
bijli serve         # run the FastAPI forecast service (uvicorn)
```

Optional extras: `serve` (FastAPI + uvicorn, for the API), `ui` (Streamlit
dashboard), `weather` (HTTPS fetch via httpx), `stats` (SARIMAX / ETS baselines),
`validate` (pandera input schema).

```bash
uv sync --extra dev --extra serve   # dev tools + API test deps
```

Defaults (epochs, batch size, splits, output paths, …) live in
[`config.toml`](config.toml); any CLI flag overrides them:

```bash
bijli train --epochs 20            # override config.toml
bijli train --log-level DEBUG      # verbose structured logging
bijli predict --no-validate        # skip input-data validation
bijli --config other.toml train    # use a different config file
```

The original scripts keep working directly (`python train.py …`).

---

## Weather & features

The forecaster can consume **exogenous daily features** in addition to the
consumption history. They are appended as extra input channels to every window
(channel 0 stays consumption), so the same GRU/LSTM trains on them with only its
`input_size` adjusted. Select a set with `--features`:

| `--features` | Adds | Columns |
| --- | --- | --- |
| `none` | nothing | – |
| `calendar` (default) | calendar + Indian public holidays | `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `is_weekend`, `is_holiday` |
| `weather` | calendar + holidays + Open-Meteo temperature | `temp_mean`, `temp_lag1..3`, `temp_roll7`, `cdd` |

```bash
bijli train --features calendar          # offline (calendar + holidays)
bijli train --features weather --city mumbai
bijli train --features weather --weather-file data/raw/weather/delhi.csv  # offline
```

* **Weather** is fetched from the [Open-Meteo](https://open-meteo.com/) archive
  API, cached per city at `data/raw/weather/<city>.csv`, and used at forecast
  time via the forecast API with a day-of-year **climatology fallback** for dates
  beyond the 16-day forecast window. `--weather-file` bypasses the network
  entirely. Known cities: `delhi`, `mumbai`, `kolkata`, `chennai`, `bengaluru`,
  `hyderabad`, `pune`, `ahmedabad`, `jaipur`, `lucknow` (+ aliases like `bombay`).
* **No leakage**: calendar/holiday flags are deterministic from the date; the
  consumption scaler and the separate **feature scaler**
  (`artifacts/feature_scaler.joblib`) are both fit on the train split only. The
  feature config is persisted in the checkpoint, and `evaluate.py` / `predict.py`
  rebuild byte-identical inputs — a parity test asserts this
  (`tests/test_features.py::test_train_eval_predict_feature_parity`).
* The 7-day horizon uses real forecast temperatures; the 90-day rollout falls
  back to climatology as it extends past the forecast window.

**Measured effect** on the synthetic dataset (60 epochs, patience 10, seed 42;
test-split metrics, `bijli ablation --models gru,lstm --features none,calendar,weather`):

| Model | Features | MAE | RMSE | WAPE | vs. seasonal-naive |
| --- | --- | --- | --- | --- | --- |
| GRU | `none` | 0.749 | 1.021 | 16.95% | –18.5% |
| GRU | `calendar` | 0.667 | 0.911 | **15.09%** | –27.5% |
| GRU | `weather` | 0.684 | 0.907 | 15.47% | –25.6% |
| LSTM | `none` | 0.900 | 1.154 | 20.38% | –2.1% |
| LSTM | `calendar` | 0.732 | 0.940 | 16.56% | –20.4% |
| LSTM | `weather` | 0.715 | 0.922 | 16.19% | –22.2% |

The generator now couples load to a synthetic temperature series
(`generate_data.py`), so the six temperature channels **help** relative to
`none`, though the deterministic calendar/holiday block remains the single
strongest addition. The GRU beats the LSTM on every feature set. Let the
ablation, not intuition, pick the features: run `bijli ablation` after any
change to the data generator.

---

## Ablation & tuning

Two experiment commands share the exact training loop, seeding and checkpoint
selection as `bijli train` (via `train.fit_model`), so results are comparable:

* **`bijli ablation`** trains one model per (architecture × feature-set)
  combination on the *same* split and scores each on the held-out test windows,
  next to a seasonal-naive floor. It answers "which design choices actually
  pull their weight?".
* **`bijli tune`** runs a random search over the learning rate (log-uniform),
  batch size and the GRU/LSTM hidden sizes/widths, keeping the configuration
  with the lowest validation loss, then optionally retrains and saves it as a
  standard checkpoint.

```bash
bijli ablation --models gru,lstm --features none,calendar,weather --epochs 60
bijli tune --trials 16 --epochs 60 --features calendar --save-artifacts artifacts
```

Both write a JSON result plus a flat CSV and (for ablation) a PNG chart to
`outputs/ablation/` and `outputs/tuning/`, read defaults from the `[ablation]`
and `[tune]` sections of `config.toml`, and record a run manifest.

Full ablation on the coupled synthetic dataset (60 epochs, patience 10, seed 42):

| Model | Features | Params | MAE | RMSE | WAPE | vs. naive |
| --- | --- | --- | --- | --- | --- | --- |
| GRU | `calendar` | 24,711 | 0.667 | 0.911 | **15.09%** | –27.5% |
| GRU | `weather` | 25,863 | 0.684 | 0.907 | 15.47% | –25.6% |
| LSTM | `weather` | 34,055 | 0.715 | 0.922 | 16.19% | –22.2% |
| LSTM | `calendar` | 32,519 | 0.732 | 0.940 | 16.56% | –20.4% |
| GRU | `none` | 23,559 | 0.749 | 1.021 | 16.95% | –18.5% |
| LSTM | `none` | 30,983 | 0.900 | 1.154 | 20.38% | –2.1% |

The random search (16 trials, GRU, `calendar`) selected
`lr ≈ 2.14e-3, batch_size = 64, hidden1 = 64, hidden2 = 48, dropout = 0.1,
dense = 64`. The retrained model scores **15.01% WAPE** on the test split
(`bijli evaluate`) — ahead of ETS (15.05%), SARIMAX (17.88%), seasonal-naive
(20.81%) and persistence (28.04%), and marginally better than the untuned
default.

---

## Prediction intervals

Forecasts ship with **split-conformal** prediction intervals so a single kWh
number becomes a range you can plan against. `evaluate.py` fits per-horizon
residual quantiles on the **validation** fold and writes
`artifacts/calibration.json` (plain JSON, no pickled sklearn objects), then
reports empirical coverage:

```
$ uv run bijli evaluate --model-path artifacts/best_model.pt
...
Conformal intervals (power, gamma=0.5; 169 val windows):
   nominal   empirical   mean width
       80%       80.3%       2.137
       90%       89.6%       2.788
```

The residual *score* is normalized by the forecast level,
`s = |y - ŷ| / σ(ŷ)` with `σ(ŷ) = ŷ^γ` (`--calibration-scale power`, default
`γ=0.5`). Household load error grows **sub-linearly** with the level, so this
variance-stabilizing scale keeps interval coverage stable across the low- and
high-consumption seasons — a plain absolute width under-covers the summer peak
on the test split. Use `--calibration-scale absolute` for the classic constant
width.

`predict.py` reads the calibration file (override with `--calibration-path`, or
pass an empty string to skip intervals) and adds:

- `lower_80 / upper_80 / lower_90 / upper_90` columns to
  `outputs/predictions.csv` and `outputs/projection_90d.csv`,
- an `interval` block in `outputs/cost_summary.json` (total kWh and bill range
  per level — bills stay monotonic in kWh through the slab tariff),
- shaded 80% bands on `cost_forecast.png` and `cost_projection.png`,
- `7-day bill 80% / 90% band` rows plus fitted-vs-nominal coverage rows in the
  Excel `Summary` sheet.

```bash
uv run bijli evaluate --model-path artifacts/best_model.pt   # writes calibration.json
uv run bijli predict  --model-path artifacts/best_model.pt   # uses it, if present
```

Design notes: intervals are finite-sample ordered statistics of the normalized
validation scores; beyond the calibrated horizon the per-step quantile
extrapolates as `q_H·√(h/H)`, and lower bounds are clipped at 0 kWh. Acceptance
target is empirical 80%/90% coverage on the test split within ±3% of nominal —
the current GRU lands at **80.3% / 89.6%**.

---

## Rolling-origin evaluation

A single chronological test split is a noisy estimate of generalisation. Pass
`--cv rolling` to additionally score the frozen model over **expanding-window
rolling-origin** folds: the splitter (`cv.py`) sweeps a fixed-size test window
forward through the series, each fold paired with the training prefix that
precedes it. Folds are anchored at the train/val boundary so every fold is fully
out-of-sample, and the loaded checkpoint is reused (no per-fold retraining).
Results are written to `outputs/evaluation_cv.json` with per-fold metrics plus
the **mean ± std** across folds:

```
$ uv run bijli evaluate --cv rolling --cv-folds 5
...
Rolling-origin CV (5 folds, 1163d min train):
 fold  windows      MAE     RMSE     WAPE
    0       35    0.631    0.931    10.77
    1       35    0.630    0.772     9.17
    2       35    0.880    1.241    12.24
    3       35    0.580    0.718     8.89
    4       35    0.720    1.005    12.87
WAPE across folds: 10.79% +/- 1.59%
Saved: outputs/evaluation_cv.json
```

Per-fold WAPE spans ~8.9–12.9%, so the fold spread (±1.6%) is a more honest
uncertainty band than the single-split point estimate — which sits right in the
middle of it.

---

## Serving & UI

The inference core lives in a FastAPI-free module (`service.py`) so the CLI, the
HTTP API (`api.py`) and the Streamlit app (`app.py`) share one code path.

### JSON API

```bash
uv sync --extra serve          # fastapi, uvicorn, pydantic, jinja2
uvicorn api:app --port 8000    # or: bijli serve --port 8000
```

| Method & path | Purpose |
| --- | --- |
| `GET /` | HTML index listing the endpoints |
| `GET /health` | model/feature/calibration status (loads on startup) |
| `POST /forecast` | 7-day (or `horizon`-day) forecast + conformal intervals |
| `POST /bill` | forecast → bill under flat / slab / preset tariff |
| `POST /projection` | autoregressive 90-day rollout + monthly / 3-month bill |

```bash
curl -s localhost:8000/health

curl -s localhost:8000/forecast -H 'content-type: application/json' \
  -d '{"history_kwh": [3.1, 2.9, 3.4, /* … 30 values … */]}'
```

`POST /forecast` takes `history_kwh` (≥ the model's input length, e.g. 30),
optional `horizon`, `last_date`, and per-request `rate / slab / preset / fixed /
tax_pct`. The model, scaler and calibration paths come from
`BIJLI_MODEL_PATH` / `BIJLI_SCALER_PATH` / `BIJLI_CALIBRATION_PATH` (defaults:
`artifacts/…`) and `BIJLI_DEVICE`. Invalid input (too-short history, NaN,
unknown preset, bad slab grammar) returns a structured **422**. The response
mirrors `outputs/` — same numbers as `predict.py`.

### Streamlit dashboard

```bash
uv sync --extra ui             # streamlit
streamlit run app.py           # then open the printed localhost URL
```

Upload a CSV (or use the bundled `data/sample_input.csv`), pick a tariff, and
read the forecast, bill and projection in tabs — with CSV / Excel / HTML
downloads built from the in-memory results.

### Docker

```bash
bijli train                    # produce artifacts/ (best_model.pt, scaler, calibration)
docker build -t bijli-predict-ai .
docker run -p 8000:8000 bijli-predict-ai     # or bind-mount fresh artifacts:
#   docker run -v "$PWD/artifacts:/app/artifacts:ro" -p 8000:8000 bijli-predict-ai
```

The image is CPU-only (torch from the PyTorch CPU wheel index), installs just
the `serve` extra from the lockfile, and ships a `HEALTHCHECK` against
`/health`. `artifacts/` is git-ignored, so a fresh clone must train first (or
mount artifacts) before the container will report healthy.

---

## Using real data

The pipeline is dataset-agnostic: any CSV with a date column and a consumption
column works. Recognised names include `date/datetime/timestamp` and
`consumption_kwh/consumption/kwh/energy_kwh`.

**Option A — your own CSV.** Drop it in `data/raw/` and train directly on it:

```bash
python train.py --data data/raw/my_household.csv
```

**Option B — the real iAWE dataset** (Indian Dataset for Ambient Water and
Energy, one New Delhi home, ~73 days). Download `electricity.tar.gz` from
<https://iawe.github.io/>, extract it to `data/raw/iawe/`, then:

```bash
python iawe_ingest.py --dir data/raw/iawe
python train.py --data data/processed/daily_consumption.csv
```

`iawe_ingest.py` integrates the instantaneous active power (`W`) of the main
meter(s) over time into daily kWh.

> **Data limitation:** iAWE covers only ~73 days from a single household — too
> short for a reliable, generalisable model. It is useful as a smoke test, which
> is why `generate_data.py` provides a longer, reproducible series by default.

---

## Data validation & run manifests

Every entry point validates its input CSV before touching the pipeline. The
dependency-free checks in `validation.py` enforce that the date and consumption
columns exist, that every date parses and every value is numeric, and that there
are at least a few rows. Duplicate timestamps, missing values and negative
readings are reported as **warnings** because `clean_series` already averages
duplicates, interpolates gaps and drops negatives. A bad file fails immediately
with a row-annotated message:

```
ValueError: Invalid consumption data:
  - 1 row(s) have an unparseable date in column 'date' (e.g. row 1: 'not-a-date').
Fix the file, or pass validate=False to skip validation.
```

Installing the optional `validate` extra (`uv sync --extra validate`) adds a
declarative **pandera** schema on top of the same contract; without it the
built-in checks still run, so validation is always available. Skip validation
per run with `--no-validate` (all commands) or `validate=False` (the Python API).

Each run also writes a JSON **manifest** to `outputs/manifests/` recording the
command and arguments, the git revision, input file hashes, output paths and
durations — enough to audit or reproduce a result later. Use `--no-manifest` to
disable, or `--manifest-dir PATH` to relocate. `--log-level DEBUG|INFO|WARNING|ERROR`
controls structured progress logging (stderr) independently of stdout reports.

---

## Outputs

| Path | Description |
| --- | --- |
| `artifacts/best_model.pt` | Best model weights + config + training history |
| `artifacts/scaler.joblib` | Train-only fitted scaler (reused by evaluate/predict) |
| `artifacts/calibration.json` | Conformal interval widths (written by `evaluate.py`) |
| `outputs/predictions.csv` | 7-day forecast (`date, predicted_kwh, lower_80, upper_80, lower_90, upper_90, estimated_cost_inr`) |
| `outputs/forecast.png` | Actual-vs-predicted consumption plots |
| `outputs/cost_forecast.png` | 7-day daily + cumulative cost chart (80% band shaded) |
| `outputs/cost_projection.png` | 90-day cumulative cost projection chart (80% band shaded) |
| `outputs/projection_90d.csv` | Per-day projected kWh / interval bounds / daily / cumulative cost |
| `outputs/cost_summary.json` | Bill breakdown + monthly & 3-month projection + interval ranges |
| `outputs/bijli_report.xlsx` | Multi-sheet Excel report (auto-built by `predict.py`) |
| `outputs/bijli_report.html` | Self-contained HTML report (auto-built by `predict.py` / `report.py`) |
| `outputs/evaluation.json` | MAE / RMSE / WAPE, baseline comparison, coverage |
| `outputs/evaluation_cv.json` | Per-fold + mean/std metrics (`evaluate.py --cv rolling`) |
| `outputs/ablation/` | Ablation JSON + CSV + WAPE chart (`ablation.py`) |
| `outputs/tuning/` | Tuning JSON + CSV, best config (`tune.py`) |
| `outputs/manifests/*.json` | Per-run provenance (command, args, git sha, input hashes, durations) |

## Cost prediction

`predict.py` forecasts **both consumption and cost** for each of the next 7 days.

### Tariff options

- **Flat rate** (default): `--rate` in ₹/kWh (default 7).
- **Slab tariff** (typical Indian discom): `--slab "LOW-HIGH:RATE,..."`. Units are
  charged progressively, e.g. the first 100 kWh at the first rate, the next 100
  at the second, etc.
- **Named discom preset**: `--preset bses`. Bundled presets live in
  [`tariff_presets.toml`](tariff_presets.toml) (`bses`, `bses_yamuna`,
  `tata_power`, `adani_mumbai`, `msedcl`, `bescom`) and are *illustrative*, not
  billing-grade.
- **Fixed charge** for the period: `--fixed` (₹).
- **Tax / surcharge** on the energy charge: `--tax-pct` (%); `--fixed`/`--tax-pct`
  override the preset's values when given.

```bash
# Flat
python predict.py --rate 7

# Named discom preset (equivalent through the CLI: `bijli predict --preset bses`)
bijli predict --preset bses

# Slabs + ₹50 fixed + 5% tax
python predict.py --slab "0-100:3.5,100-200:4.5,200-400:6.5,400-100000:7" \
    --fixed 50 --tax-pct 5
```

Bill = energy charge + tax + fixed charge. For slab tariffs the slabs are applied
to the **period total** and each day is priced at the margin, so the per-day
costs always sum to the period energy charge.

### Monthly & 3-month projections

Beyond the 7-day forecast, `predict.py` rolls the model forward
**autoregressively** (`--project-days`, default 90) to estimate a **monthly
(30-day)** and **3-month (90-day)** bill. Each predicted window is fed back as
the next input.

```bash
python predict.py --rate 7 --project-days 90
```

```text
Long-term cost projection (rolling forecast)
  Monthly   (30 days):    107.0 kWh   Estimated Bill: ₹748.70
  3-Month   (90 days):    327.5 kWh   Estimated Bill: ₹2292.24
  NOTE: long-horizon rollout accumulates error; treat as an estimate.
```

> Long-horizon projections compound model error, so they are directional
> estimates, not guarantees.

### Excel & HTML reports

`predict.py` also assembles a single multi-sheet workbook at
`outputs/bijli_report.xlsx` (sheets: `Summary`, `Forecast`, `Projection`,
`Evaluation`, `Tariff`) **and** a self-contained `outputs/bijli_report.html`
with the same tables (no external assets). Rebuild either from existing outputs
at any time:

```bash
python report.py --outputs-dir outputs \
    --out outputs/bijli_report.xlsx --html-out outputs/bijli_report.html
```

The notebook (`notebooks/exploration.ipynb`) builds the same workbook and
offers it as a download link.

### Example prediction output

```text
BIJLIPREDICT AI
-----------------------------

Forecast Period: 7 Days
Tariff: slab [0-100:₹3.5, 100-200:₹4.5, 200-400:₹6.5, 400-100000:₹7]

Day 1: 3.0 kWh   ₹ 10.45   (2025-12-31)
Day 2: 3.2 kWh   ₹ 11.05   (2026-01-01)
...
Day 7: 3.2 kWh   ₹ 11.07   (2026-01-06)

-----------------------------
Total Forecast: 24.5 kWh

Energy Charge      : ₹85.76
Tax (5%)           : ₹4.29
Fixed Charge       : ₹50.00
Estimated Bill     : ₹140.05

Prediction saved:
outputs/predictions.csv
Cost chart saved:
outputs/cost_forecast.png
Bill summary saved:
outputs/cost_summary.json
```

The estimate is indicative only — it does not model subsidies, time-of-day
rates, or per-appliance breakdowns.

---

## Metrics

Reported on the unseen chronological test period and compared against two naive
baselines:

- **MAE** — mean absolute error (kWh/day)
- **RMSE** — root mean squared error (kWh/day)
- **WAPE** — weighted absolute percentage error, `sum|y-ŷ| / sum|y|`
- **MAPE** — mean absolute percentage error

Baselines: **seasonal-naive (period 7)** — repeat the last observed week — and
**persistence** — repeat the last observed day. With the optional `stats` extra
(`pip install .[stats]`), two classical models are added as rolling 7-day
forecasts from each test origin, scored on the same windows as the neural model:
**SARIMAX** `(1,1,1)(1,0,0,7)` (state advanced with each observation via
`append`, no refit) and **ETS** (Holt-Winters, additive weekly seasonality,
refit per origin).

On the default (temperature-coupled) synthetic dataset the tuned GRU leads every
baseline on the single test split: **WAPE ≈ 15.0%** vs seasonal-naive ≈ 20.8%,
persistence ≈ 28.0%, SARIMAX ≈ 17.9% and ETS ≈ 15.0% (the two are within 0.04
percentage points). That a classical smoother is essentially level with a small
GRU is exactly why the rolling-origin numbers in `evaluation_cv.json` matter
more than any single-split ranking.

```bash
pip install '.[stats]'          # or: uv pip install statsmodels
uv run bijli evaluate --cv rolling
```

---

## Project layout

```text
bijli-predict-ai/
├── .github/workflows/ci.yml     # ruff + pytest on Python 3.12 / 3.14
├── data/
│   ├── raw/                     # source CSVs (synthetic or iAWE) + weather/ cache
│   ├── processed/               # daily_consumption.csv
│   └── sample_input.csv         # last 30 days, used by predict.py
├── notebooks/exploration.ipynb  # lightweight EDA
├── models/
│   ├── gru_model.py             # primary model
│   └── lstm_model.py            # comparison model
├── tests/                       # pytest suite (cli, tariffs, preprocessing,
│                                #   inference, weather, features, calibration,
│                                #   cv, baselines, api, ablation, tune)
├── artifacts/                   # best_model.pt, scaler.joblib,
│                                #   feature_scaler.joblib, calibration.json
├── outputs/                     # predictions.csv, forecast.png, evaluation.json,
│                                #   evaluation_cv.json, cost_forecast.png,
│                                #   cost_projection.png, projection_90d.csv,
│                                #   cost_summary.json, bijli_report.xlsx/.html,
│                                #   ablation/ (json+csv+png), tuning/ (json+csv)
├── cli.py                       # unified `bijli` entry point
├── config.toml                  # hyperparameters / defaults (tomllib)
├── tariff_presets.toml          # named discom tariffs
├── preprocessing.py             # load / clean / split / window / scale
├── features.py                  # calendar / holiday / temperature features
├── weather.py                   # Open-Meteo archive + forecast + climatology
├── calibration.py               # split-conformal prediction intervals
├── cv.py                        # rolling-origin CV splitter
├── baselines.py                 # SARIMAX / ETS statistical baselines
├── generate_data.py             # synthetic Indian household generator
├── iawe_ingest.py               # real iAWE -> daily kWh
├── tariffs.py                   # flat / slab / preset / fixed / tax bill calculation
├── inference.py                 # shared device selection + model loading
├── service.py                   # FastAPI-free inference/tariff core
├── api.py                       # FastAPI JSON service (uvicorn api:app)
├── app.py                       # Streamlit dashboard
├── report.py                    # Excel + HTML report builder
├── train.py                     # training entrypoint (fit_model / save_checkpoint)
├── ablation.py                  # architecture x feature-set ablation study
├── tune.py                      # random hyperparameter search
├── evaluate.py                  # metrics + baselines + plot
├── predict.py                   # 7-day consumption + cost forecast
├── Dockerfile                   # slim CPU-only API image
├── Makefile                     # task runner
├── pyproject.toml               # deps, scripts, ruff + pytest config
├── requirements.txt
└── uv.lock                      # locked dependency resolution
```

---

## Testing & quality

```bash
uv sync --extra dev --extra serve --extra validate   # or: pip install pytest ruff fastapi httpx pandera
make test             # run the unit tests (pytest)
make lint             # lint (ruff check .)
```

`make train` runs with `--deterministic` (seeded RNGs + deterministic kernels),
so repeated runs produce identical artifacts.

---

## Notes on correctness

- The train/val/test split is **chronological**; the scaler is fitted on the
  training segment only, so no test information leaks into training.
- Gap filling is applied **per split segment**, so validation/test values can
  never be interpolated into the training segment.
- Model artifacts are loaded with `torch.load(..., weights_only=True)`, so a
  tampered checkpoint cannot execute arbitrary code.
- Predictions are clipped to be **non-negative** (consumption cannot be
  negative).
- All random seeds are fixed (`--seed`, default 42) for reproducibility.
- Input CSVs are validated up front (see *Data validation & run manifests*), and
  every run writes a provenance manifest with input hashes and the git revision.
- This is a reproducible forecasting experiment with a thin local serving layer —
  not a hardened, authenticated production service.
