"""BijliPredict AI - assemble Excel and HTML reports from pipeline outputs.

Builds a single multi-sheet workbook (and a standalone HTML page) from the files
produced by `predict.py` and `evaluate.py`:

    Summary             bill + projection headline numbers
    Forecast            the 7-day forecast (kWh + cost)
    Projection          the 90-day rolling projection
    Evaluation          MAE / RMSE / WAPE vs baselines
    Tariff              the tariff used, per-day costs

Usage:
    python report.py                          # from default outputs/
    python report.py --outputs-dir outputs --out outputs/bijli_report.xlsx
    python report.py --html-out outputs/bijli_report.html
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

DEFAULT_OUTPUTS = "outputs"
DEFAULT_XLSX = "outputs/bijli_report.xlsx"
DEFAULT_HTML = "outputs/bijli_report.html"

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")

_HTML_STYLE = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0 auto; max-width: 960px; padding: 2rem 1.25rem; line-height: 1.45; }
h1 { border-bottom: 3px solid #1f4e78; padding-bottom: .4rem; }
h2 { margin-top: 2rem; color: #1f4e78; }
.meta { color: #666; font-size: .9rem; }
table.table { border-collapse: collapse; width: 100%; margin-top: .5rem; font-size: .92rem; }
table.table th { background: #1f4e78; color: #fff; text-align: left; padding: .5rem .6rem; }
table.table td { border-bottom: 1px solid #ddd; padding: .45rem .6rem; }
table.table tr:nth-child(even) td { background: rgba(31, 78, 120, 0.05); }
section { margin-bottom: 1.5rem; }
"""


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _summary_frame(cost: dict, evaluation: dict) -> pd.DataFrame:
    proj = cost.get("projection", {})
    monthly = proj.get("monthly_30d", {})
    quarterly = proj.get("three_month_90d", {})
    model_m = evaluation.get("model_metrics", {})
    base_m = evaluation.get("baseline_metrics", {}).get("seasonal_naive_p7", {})
    interval = cost.get("interval", {}).get("levels", {})

    rows = [
        ("Generated", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Model", evaluation.get("model", "gru").upper()),
        ("Tariff", cost.get("tariff", "")),
        ("", ""),
        ("7-day forecast (kWh)", round(cost.get("total_kwh", 0.0), 2)),
        ("7-day estimated bill (INR)", cost.get("estimated_bill_inr", 0.0)),
    ]
    for pct in sorted(interval):
        band = interval[pct]
        rows.append(
            (
                f"7-day bill {pct}% band (INR)",
                f"{band.get('estimated_bill_lower_inr')} – {band.get('estimated_bill_upper_inr')}",
            )
        )
    rows += [
        ("Monthly 30d (kWh)", round(monthly.get("total_kwh", 0.0), 2)),
        ("Monthly 30d bill (INR)", monthly.get("estimated_bill_inr", 0.0)),
        ("3-Month 90d (kWh)", round(quarterly.get("total_kwh", 0.0), 2)),
        ("3-Month 90d bill (INR)", quarterly.get("estimated_bill_inr", 0.0)),
        ("", ""),
        ("Test MAE (model)", round(model_m.get("mae", 0.0), 3)),
        ("Test RMSE (model)", round(model_m.get("rmse", 0.0), 3)),
        ("Test WAPE % (model)", round(model_m.get("wape", 0.0), 2)),
        ("Test WAPE % (seasonal-naive)", round(base_m.get("wape", 0.0), 2)),
    ]
    coverage = evaluation.get("calibration", {}).get("coverage", {})
    for pct in sorted(coverage):
        c = coverage[pct]
        rows.append(
            (
                f"Coverage {float(pct) * 100:.0f}% (nominal {c.get('target', 0) * 100:.0f}%)",
                f"{c.get('empirical', 0) * 100:.1f}%",
            )
        )
    return pd.DataFrame(rows, columns=["Item", "Value"])


def _evaluation_frame(evaluation: dict) -> pd.DataFrame:
    model_m = evaluation.get("model_metrics", {})
    baselines = evaluation.get("baseline_metrics", {})
    metrics = ["mae", "rmse", "wape", "mape"]
    data = {
        "metric": [m.upper() for m in metrics],
        evaluation.get("model", "model").upper(): [model_m.get(m) for m in metrics],
        "seasonal_naive_p7": [baselines.get("seasonal_naive_p7", {}).get(m) for m in metrics],
        "persistence": [baselines.get("persistence", {}).get(m) for m in metrics],
    }
    return pd.DataFrame(data)


def _tariff_frame(cost: dict) -> pd.DataFrame:
    rows = [("Tariff", cost.get("tariff", ""))]
    daily = cost.get("daily", [])
    for d in daily:
        rows.append((d.get("date"), d.get("cost_inr")))
    return pd.DataFrame(rows, columns=["Date / Info", "Daily cost (INR)"])


def _autofit(writer: pd.ExcelWriter) -> None:
    for ws in writer.book.worksheets:
        for col_cells in ws.columns:
            length = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(
                40, max(12, length + 2)
            )
        for cell in ws[1]:
            if cell.value is not None:
                cell.fill = HEADER_FILL
                cell.font = HEADER_FONT
                cell.alignment = Alignment(horizontal="center")
        ws.freeze_panes = "A2"


def _load_sheets(outputs_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Assemble the report sheets shared by the Excel and HTML builders."""
    outputs_dir = Path(outputs_dir)
    cost = _read_json(outputs_dir / "cost_summary.json")
    evaluation = _read_json(outputs_dir / "evaluation.json")
    forecast = _read_csv(outputs_dir / "predictions.csv")
    projection = _read_csv(outputs_dir / "projection_90d.csv")
    return {
        "Summary": _summary_frame(cost, evaluation),
        "Forecast": forecast,
        "Projection": projection,
        "Evaluation": _evaluation_frame(evaluation),
        "Tariff": _tariff_frame(cost),
    }


def build_excel(
    outputs_dir: str | Path = DEFAULT_OUTPUTS,
    out_path: str | Path = DEFAULT_XLSX,
) -> Path:
    """Read pipeline outputs and write a multi-sheet Excel workbook."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheets = _load_sheets(outputs_dir)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            (frame if not frame.empty else pd.DataFrame({"info": ["no data"]})).to_excel(
                writer, sheet_name=name, index=False
            )
        _autofit(writer)
    return out_path


def _html_table(frame: pd.DataFrame, empty_msg: str = "no data") -> str:
    if frame is None or frame.empty:
        frame = pd.DataFrame({"info": [empty_msg]})
    return frame.to_html(index=False, border=0, classes="table", justify="left")


def render_html(sheets: dict[str, pd.DataFrame], source: str = "") -> str:
    """Render a dict of named tables into a standalone HTML document."""
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = f"Generated {generated}" + (f" &middot; source: <code>{source}</code>" if source else "")
    sections = "\n".join(
        f"<section><h2>{name}</h2>\n{_html_table(frame)}</section>"
        for name, frame in sheets.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BijliPredict AI report</title>
<style>{_HTML_STYLE}</style>
</head>
<body>
<h1>BijliPredict AI report</h1>
<p class="meta">{meta}</p>
{sections}
</body>
</html>
"""


def build_html(
    outputs_dir: str | Path = DEFAULT_OUTPUTS,
    out_path: str | Path = DEFAULT_HTML,
) -> Path:
    """Read pipeline outputs and write a standalone HTML report (no deps)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(_load_sheets(outputs_dir), source=str(Path(outputs_dir)))
    out_path.write_text(html)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the report from outputs.")
    p.add_argument("--outputs-dir", default=DEFAULT_OUTPUTS)
    p.add_argument("--out", default=DEFAULT_XLSX, help="Excel report path")
    p.add_argument("--html-out", default=DEFAULT_HTML, help="HTML report path (empty to skip)")
    return p


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()
    path = build_excel(args.outputs_dir, args.out)
    print(f"Excel report saved: {path}")
    if args.html_out:
        html_path = build_html(args.outputs_dir, args.html_out)
        print(f"HTML report saved: {html_path}")


if __name__ == "__main__":
    main()
