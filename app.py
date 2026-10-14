"""BijliPredict AI - Streamlit UI for forecasts, bills and cost projections.

Run it:
    streamlit run app.py

It reuses the same inference core as the CLI and API (`service.py`) and renders
the 7-day forecast with conformal intervals, the bill breakdown, a rolling cost
projection, and downloadable CSV / Excel / HTML reports.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import service
from report import render_html

st.set_page_config(page_title="BijliPredict AI", page_icon="⚡", layout="wide")

DEFAULT_SAMPLE = "data/sample_input.csv"
presets = {}
try:
    from tariffs import load_presets

    presets = load_presets()
except FileNotFoundError:
    presets = {}


@st.cache_resource(show_spinner="Loading model…")
def get_service(
    model_path: str, scaler_path: str | None, calibration_path: str | None
) -> service.ServiceState:
    return service.load_service(
        model_path=model_path,
        scaler_path=scaler_path or None,
        calibration_path=calibration_path or None,
        device="cpu",
    )


@st.cache_data(show_spinner=False)
def load_history_csv(raw: bytes) -> pd.Series:
    df = pd.read_csv(io.BytesIO(raw))
    lower = {c.lower().strip(): c for c in df.columns}
    value_col = next(
        (lower[k] for k in ("consumption_kwh", "kwh", "consumption", "value") if k in lower),
        df.columns[-1],
    )
    return pd.to_numeric(df[value_col], errors="coerce").dropna().reset_index(drop=True)


def default_history() -> pd.Series:
    path = Path(DEFAULT_SAMPLE)
    if path.exists():
        df = pd.read_csv(path)
        col = "consumption_kwh" if "consumption_kwh" in df.columns else df.columns[-1]
        return pd.to_numeric(df[col], errors="coerce").dropna().reset_index(drop=True)
    return pd.Series([5.0 + 0.4 * (i % 7) for i in range(60)])


def excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


# --- sidebar -----------------------------------------------------------------

st.sidebar.header("Model")
model_path = st.sidebar.text_input("Model checkpoint", service.DEFAULT_MODEL_PATH)
calib_path = st.sidebar.text_input("Calibration (optional)", service.DEFAULT_CALIBRATION_PATH)

st.sidebar.header("History")
uploaded = st.sidebar.file_uploader("Upload daily kWh CSV", type=["csv"])
if uploaded is not None:
    history_series = load_history_csv(uploaded.getvalue())
else:
    history_series = default_history()
st.sidebar.caption(
    f"{len(history_series)} days loaded ({DEFAULT_SAMPLE if uploaded is None else 'upload'})"
)

st.sidebar.header("Tariff")
mode = st.sidebar.radio("Mode", ["Flat", "Slab", "Preset"], horizontal=True)
rate = slab = preset = None
fixed = st.sidebar.number_input("Fixed charge (INR)", min_value=0.0, value=0.0, step=10.0)
tax_pct = st.sidebar.number_input("Tax / surcharge (%)", min_value=0.0, value=0.0, step=0.5)
if mode == "Flat":
    rate = st.sidebar.number_input("Rate (INR/kWh)", min_value=0.0, value=7.0, step=0.5)
elif mode == "Slab":
    slab = st.sidebar.text_input("Slabs", "0-100:3.5,100-200:4.5,200-100000:7")
else:
    preset = st.sidebar.selectbox("Preset", sorted(presets) or ["(none)"])

st.sidebar.header("Horizon")
project_days = st.sidebar.slider("Projection days", min_value=7, max_value=180, value=90, step=1)
last_date = st.sidebar.date_input("Last history date", value=pd.Timestamp.today().date())


# --- main --------------------------------------------------------------------

st.title("⚡ BijliPredict AI")
st.caption("Household electricity forecasting, billing and cost projection.")

try:
    svc = get_service(model_path, calib_path, calib_path)
except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
    st.error(f"Could not load the model: {exc}")
    st.stop()

if len(history_series) < svc.input_len:
    st.error(f"Need at least {svc.input_len} days of history, got {len(history_series)}.")
    st.stop()

history = history_series.tolist()
try:
    tariff = service.resolve_tariff(
        rate=rate, slab=slab, preset=preset, fixed=fixed, tax_pct=tax_pct
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Invalid tariff: {exc}")
    st.stop()

forecast, dates = service.run_forecast(svc, history, svc.horizon, last_date)
projection, proj_dates = service.run_forecast(svc, history, project_days, last_date)
iv = service.intervals(forecast, svc.calib)
proj_iv = service.intervals(projection, svc.calib)

bill = service.compute_bill(forecast, tariff)
proj_bill = service.compute_bill(projection, tariff)
monthly = service.compute_bill(projection[: service.MONTH_DAYS], tariff)
quarterly = service.compute_bill(projection[: service.QUARTER_DAYS], tariff)

tab_forecast, tab_bill, tab_projection = st.tabs(["Forecast", "Bill", "Projection"])

with tab_forecast:
    st.subheader(f"{svc.horizon}-day forecast")
    forecast_df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "predicted_kwh": [round(float(v), 3) for v in forecast],
        }
    )
    for pct in sorted(iv):
        forecast_df[f"lower_{pct}"] = [round(float(v), 3) for v in iv[pct][0]]
        forecast_df[f"upper_{pct}"] = [round(float(v), 3) for v in iv[pct][1]]
    st.line_chart(forecast_df, x="date", y=["predicted_kwh", "lower_80", "upper_80"])
    st.dataframe(forecast_df, use_container_width=True)

with tab_bill:
    st.subheader("Estimated bill")
    cols = st.columns(4)
    cols[0].metric("Total", f"{bill['total_kwh']:.1f} kWh")
    cols[1].metric("Energy", f"₹{bill['energy_charge']:.0f}")
    cols[2].metric("Tax", f"₹{bill['tax']:.0f}")
    cols[3].metric("Bill", f"₹{bill['total']:.0f}")
    st.caption(f"Tariff: {tariff.describe()}")
    if iv:
        for pct in sorted(iv):
            lo = service.compute_bill(iv[pct][0], tariff)["total"]
            hi = service.compute_bill(iv[pct][1], tariff)["total"]
            st.write(f"**{pct}% bill interval**: ₹{lo:.0f} – ₹{hi:.0f}")
    bill_df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "kwh": [round(float(v), 3) for v in forecast],
            "cost_inr": [round(float(c), 2) for c in bill["daily_costs"]],
        }
    )
    st.dataframe(bill_df, use_container_width=True)

with tab_projection:
    st.subheader(f"{project_days}-day cost projection")
    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly (30d)", f"₹{monthly['total']:.0f}")
    c2.metric("3-Month (90d)", f"₹{quarterly['total']:.0f}")
    c3.metric("Full horizon", f"₹{proj_bill['total']:.0f}")
    cumulative = pd.Series(proj_bill["daily_costs"]).cumsum().tolist()
    proj_df = pd.DataFrame(
        {
            "date": proj_dates.strftime("%Y-%m-%d"),
            "projected_kwh": [round(float(v), 3) for v in projection],
            "daily_cost_inr": [round(float(c), 2) for c in proj_bill["daily_costs"]],
            "cumulative_cost_inr": [round(float(c), 2) for c in cumulative],
        }
    )
    st.line_chart(proj_df, x="date", y="cumulative_cost_inr")
    st.dataframe(proj_df, use_container_width=True)
    st.caption("Long-horizon rollout accumulates error; treat as an estimate.")

# --- downloads ---------------------------------------------------------------

summary_rows = [
    ("Model", svc.model_name),
    ("Tariff", tariff.describe()),
    ("Forecast days", svc.horizon),
    ("Total kWh", round(bill["total_kwh"], 3)),
    ("Estimated bill (INR)", round(bill["total"], 2)),
    ("Monthly 30d (INR)", round(monthly["total"], 2)),
    ("3-Month 90d (INR)", round(quarterly["total"], 2)),
]
summary_df = pd.DataFrame(summary_rows, columns=["Item", "Value"])
sheets = {"Summary": summary_df, "Forecast": forecast_df, "Projection": proj_df}

st.divider()
d1, d2, d3 = st.columns(3)
d1.download_button("⬇️ Forecast CSV", forecast_df.to_csv(index=False).encode(), "forecast.csv")
d2.download_button("⬇️ Excel report", excel_bytes(sheets), "bijli_report.xlsx")
d3.download_button("⬇️ HTML report", render_html(sheets).encode(), "bijli_report.html")

with st.expander("Raw JSON"):
    st.code(
        json.dumps(
            {
                "forecast": forecast_df.to_dict(orient="records"),
                "summary": dict(summary_rows),
            },
            indent=2,
            default=str,
        ),
        language="json",
    )
