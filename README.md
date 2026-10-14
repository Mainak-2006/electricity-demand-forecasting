# BijliPredict AI

Electricity demand forecasting for Indian households — a local, ML-only MVP.

A GRU neural network (with an LSTM comparison model) is trained on historical
daily household electricity consumption and forecasts the **next 7 days** of
usage, along with an estimated energy bill in Indian Rupees (₹).

No frontend, backend, database, API or deployment. Everything runs locally in
Python.

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
Cost forecast + monthly / 3-month projection + outputs/bijli_report.xlsx
```

---

## Requirements

- Python 3.12+ (tested on 3.14)
- [uv](https://docs.astral.sh/uv/) (recommended) or plain `venv` + `pip`

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

# 4. Forecast the next 7 days, estimate the bill, and build the Excel report.
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
bijli evaluate      # metrics on the held-out test split
bijli predict       # 7-day forecast + bill + 90-day projection
bijli report        # rebuild the Excel report from outputs/
```

Defaults (epochs, batch size, splits, output paths, …) live in
[`config.toml`](config.toml); any CLI flag overrides them:

```bash
bijli train --epochs 20            # override config.toml
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

**Measured effect** on the synthetic dataset (GRU, window 30 / horizon 7,
`--deterministic`, seed 42; test-split WAPE):

| Features | MAE | WAPE | vs. seasonal-naive |
| --- | --- | --- | --- |
| `none` | 0.703 | 10.76% | –23.0% |
| `calendar` | 0.699 | 10.70% | –23.4% |
| `weather` | 0.709 | 10.85% | –22.3% |

Under the deterministic default the calendar/holiday features give only a
**marginal** gain (10.70% vs 10.76% WAPE); temperature does **not** help here —
the synthetic generator encodes a pure annual cycle with no weather coupling, so
the six temperature channels add noise. A deliberate reminder to let the data,
not intuition, pick features. (An earlier, non-deterministic comparison showed a
larger calendar gain; that was run-to-run variance, not signal.)

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
| `outputs/evaluation.json` | MAE / RMSE / WAPE, baseline comparison, coverage |

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

### Excel report

`predict.py` also assembles a single multi-sheet workbook at
`outputs/bijli_report.xlsx` (sheets: `Summary`, `Forecast`, `Projection`,
`Evaluation`, `Tariff`). Rebuild it from existing outputs at any time:

```bash
python report.py --outputs-dir outputs --out outputs/bijli_report.xlsx
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
**persistence** — repeat the last observed day.

On the default synthetic dataset the GRU beats both baselines
(e.g. WAPE ≈ 10.7% vs ≈ 14.0% for seasonal-naive and ≈ 17.0% for persistence).

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
│                                #   inference, weather, features, calibration)
├── artifacts/                   # best_model.pt, scaler.joblib,
│                                #   feature_scaler.joblib, calibration.json
├── outputs/                     # predictions.csv, forecast.png, evaluation.json,
│                                #   cost_forecast.png, cost_projection.png,
│                                #   projection_90d.csv, cost_summary.json,
│                                #   bijli_report.xlsx
├── cli.py                       # unified `bijli` entry point
├── config.toml                  # hyperparameters / defaults (tomllib)
├── tariff_presets.toml          # named discom tariffs
├── preprocessing.py             # load / clean / split / window / scale
├── features.py                  # calendar / holiday / temperature features
├── weather.py                   # Open-Meteo archive + forecast + climatology
├── calibration.py               # split-conformal prediction intervals
├── generate_data.py             # synthetic Indian household generator
├── iawe_ingest.py               # real iAWE -> daily kWh
├── tariffs.py                   # flat / slab / preset / fixed / tax bill calculation
├── inference.py                 # shared device selection + model loading
├── report.py                    # multi-sheet Excel report builder
├── train.py                     # training entrypoint
├── evaluate.py                  # metrics + baselines + plot
├── predict.py                   # 7-day consumption + cost forecast
├── Makefile                     # task runner
├── pyproject.toml               # deps, scripts, ruff + pytest config
├── requirements.txt
└── uv.lock                      # locked dependency resolution
```

---

## Testing & quality

```bash
uv sync --extra dev   # or: pip install pytest ruff
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
- This is a reproducible forecasting experiment, not a production prediction
  service.
