#!/usr/bin/env python3
"""
Streamlit dashboard for the psychological safety survey export.

Run locally:
    streamlit run insights_dashboard.py

Upload the CSV inside the app or point the sidebar text box at the file that
already lives on disk (the sample export is auto-detected if it sits next to
this script).
"""
from __future__ import annotations

import io
import math
import os
import subprocess
from datetime import datetime
from html import escape
from importlib import import_module
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
sync_playwright = None  # lazy loaded via get_sync_playwright()
PLAYWRIGHT_INSTALL_ATTEMPTED = False
_default_browser_cache = Path(
    os.getenv("PLAYWRIGHT_BROWSERS_PATH") or (Path.cwd() / ".cache" / "ms-playwright")
).resolve()
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(_default_browser_cache))
_default_browser_cache.mkdir(parents=True, exist_ok=True)

LIKERT_ORDER = [
    "Never",
    "Rarely",
    "Some of the time",
    "Most of the time",
    "All of the time",
]

COLOR_MAP = {
    "Never": "#c0392b",
    "Rarely": "#e67e22",
    "Some of the time": "#f1c40f",
    "Most of the time": "#82c91e",
    "All of the time": "#2d8638",
}

HTML_CLASS_MAP = {
    "Never": "red",
    "Rarely": "orange",
    "Some of the time": "yellow",
    "Most of the time": "green",
    "All of the time": "dark-green",
}

VALUE_MAP = {
    "Never": 0,
    "Rarely": 25,
    "Some of the time": 50,
    "Most of the time": 75,
    "All of the time": 100,
}

ANSWER_REMAP = {
    "not at all": "Never",
    "never": "Never",
    "rarely": "Rarely",
    "some of the time": "Some of the time",
    "most of the time": "Most of the time",
    "all of the time": "All of the time",
}

SECTION_SPECS = [
    (
        "Risk Factors",
        slice(9, 21),
        "This dial represents your team's risk level – "
        "lower scores mean lower exposure to negative factors.",
        True,
    ),
    (
        "Inside My Team",
        slice(21, 31),
        "This dial reflects how psychologically safe people feel inside the team.",
        False,
    ),
    (
        "Outside My Team",
        slice(31, 39),
        "This dial reflects how the team feels when collaborating outside the team.",
        False,
    ),
]

NAV_TABS = [spec[0] for spec in SECTION_SPECS]

class PlaywrightUnavailable(RuntimeError):
    """Raised when the styled PDF cannot be produced because Playwright is missing."""


def get_sync_playwright():
    """Load Playwright lazily so installs after the first run are picked up."""
    global sync_playwright  # noqa: WPS420
    if sync_playwright is None:
        try:
            module = import_module("playwright.sync_api")
            sync_playwright = module.sync_playwright
        except ImportError as exc:  # pragma: no cover
            raise PlaywrightUnavailable(
                "Install Playwright (pip install playwright) and run "
                "`playwright install chromium` to enable styled PDF downloads."
            ) from exc
    return sync_playwright

DEFAULT_DATA_FILE = Path(__file__).with_name(
    "PT. Puma Jaya Utama Geoscience - Team Excel-Cleaned.csv"
)


def normalize_response(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        lowered = candidate.lower()
        if lowered in {"n/a", "na"}:
            return None
        return ANSWER_REMAP.get(lowered, candidate)

    return None


def load_clean_dataframe(csv_source: Path | str | io.BytesIO | io.StringIO) -> pd.DataFrame:
    """Read the csv, promote the first data row to headers and normalise answers."""
    raw = pd.read_csv(csv_source, encoding="utf-8-sig")
    question_row = raw.iloc[0]
    data = raw.iloc[1:].reset_index(drop=True)

    renamed_columns: List[str] = []
    duplicates: Dict[str, int] = {}

    for original, question_text in zip(raw.columns, question_row):
        if isinstance(question_text, str) and question_text.strip():
            candidate = question_text.strip()
        else:
            candidate = original

        count = duplicates.get(candidate, 0)
        duplicates[candidate] = count + 1
        if count:
            candidate = f"{candidate} ({count + 1})"
        renamed_columns.append(candidate)

    data.columns = renamed_columns
    for column in data.columns[9:]:
        data[column] = data[column].apply(normalize_response)

    if "Start Date" in data.columns:
        data["Start Date"] = pd.to_datetime(data["Start Date"], errors="coerce")
    if "End Date" in data.columns:
        data["End Date"] = pd.to_datetime(data["End Date"], errors="coerce")

    return data


def build_section_payload(df: pd.DataFrame) -> Dict[str, Dict]:
    summaries: Dict[str, Dict] = {}
    last_updated = None
    if "End Date" in df.columns:
        last_date = df["End Date"].max()
        if pd.notna(last_date):
            last_updated = last_date.date().isoformat()

    for section_name, column_slice, description, reverse_gauge in SECTION_SPECS:
        columns = list(df.columns[column_slice])
        section_frame = df[columns]
        values_frame = section_frame.map(lambda x: VALUE_MAP.get(x))
        row_scores = values_frame.mean(axis=1, skipna=True)
        if reverse_gauge:
            row_scores = 100 - row_scores
        section_score = (
            float(row_scores.mean()) if row_scores.notna().any() else 0.0
        )

        distributions = []
        for question in columns:
            responses = section_frame[question].dropna()
            total = len(responses)
            breakdown = {answer: 0.0 for answer in LIKERT_ORDER}
            if total:
                for answer in LIKERT_ORDER:
                    breakdown[answer] = round(
                        (responses == answer).sum() * 100 / total, 1
                    )
            distributions.append({"question": question, **breakdown})

        trend_dates: List[str] = []
        trend_scores: List[float] = []
        if "Start Date" in df.columns:
            trend_frame = pd.DataFrame(
                {"date": df["Start Date"], "score": row_scores}
            ).dropna()
            if not trend_frame.empty:
                grouped = (
                    trend_frame.groupby(trend_frame["date"].dt.date)["score"]
                    .mean()
                    .reset_index()
                )
                trend_dates = grouped["date"].astype(str).tolist()
                trend_scores = grouped["score"].round(1).tolist()

        summaries[section_name] = {
            "description": description,
            "score": round(section_score, 1),
            "reverse_gauge": reverse_gauge,
            "distributions": distributions,
            "trend": {"dates": trend_dates, "scores": trend_scores},
            "respondents": int(section_frame.dropna(how="all").shape[0]),
            "last_updated": last_updated,
        }
    return summaries


def gauge_steps(reverse: bool = False) -> List[Dict]:
    colors = ["#c0392b", "#f4d03f", "#2ecc71"]
    if reverse:
        colors = list(reversed(colors))
    ranges = [(0, 33.3), (33.3, 66.6), (66.6, 100)]
    return [{"range": rng, "color": color} for rng, color in zip(ranges, colors)]


def _gauge_coordinates(value: float, radius: float = 90.0) -> tuple[float, float]:
    clamped = max(0.0, min(100.0, float(value)))
    angle = math.pi + (clamped / 100.0) * math.pi
    return (
        100 + radius * math.cos(angle),
        100 + radius * math.sin(angle),
    )


def build_svg_gauge(score: float, reverse: bool) -> str:
    clamped = max(0.0, min(100.0, float(score or 0.0)))

    def arc_path(start_value: float, end_value: float) -> str:
        start_x, start_y = _gauge_coordinates(start_value)
        end_x, end_y = _gauge_coordinates(end_value)
        large_arc = 1 if (end_value - start_value) > 50 else 0
        return (
            f"M {start_x:.2f} {start_y:.2f} "
            f"A 90 90 0 {large_arc} 1 {end_x:.2f} {end_y:.2f}"
        )

    track_path = arc_path(0.0, 100.0)
    colored_arcs = "\n".join(
        f"<path d='{arc_path(start, end)}' stroke='{entry['color']}' "
        f"stroke-width='14' fill='none' stroke-linecap='round' />"
        for entry in gauge_steps(reverse)
        for start, end in [entry["range"]]
    )

    needle_x, needle_y = _gauge_coordinates(clamped, radius=80.0)
    svg_markup = f"""
<svg viewBox="0 0 200 120" role="img" aria-label="Gauge score {clamped:.1f}%">
    <path d="{track_path}" stroke="#e2e8f0" stroke-width="14" fill="none" stroke-linecap="round" />
    {colored_arcs}
    <line x1="100" y1="100" x2="{needle_x:.2f}" y2="{needle_y:.2f}" stroke="#0f172a" stroke-width="3" stroke-linecap="round" />
    <circle cx="100" cy="100" r="6" fill="#0f172a" stroke="#ffffff" stroke-width="3" />
</svg>
""".strip()
    return svg_markup


def build_gauge_figure(section: str, score: float, reverse: bool) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2b2b2b"},
                "steps": gauge_steps(reverse),
                "threshold": {
                    "line": {"color": "#2b2b2b", "width": 4},
                    "value": score,
                },
            },
            title={"text": section},
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=20),
        height=320,
        template="plotly_white",
    )
    return fig


def build_distribution_figure(
    section_name: str, distributions: List[Dict]
) -> go.Figure:
    frame = pd.DataFrame(distributions)
    bars = []
    for answer in LIKERT_ORDER:
        labels = [
            f"{value:.0f}%"
            if value and not pd.isna(value) and value >= 1
            else ""
            for value in frame[answer]
        ]
        bars.append(
            go.Bar(
                name=answer,
                y=frame["question"],
                x=frame[answer],
                orientation="h",
                marker=dict(color=COLOR_MAP[answer]),
                text=labels,
                textposition="inside",
                textfont=dict(color="white", size=12),
                hovertemplate=f"<b>%{{y}}</b><br>{answer}: %{{x:.1f}}%<extra></extra>",
            )
        )

    fig = go.Figure(bars)
    fig.update_layout(
        barmode="stack",
        title=f"{section_name} – question breakdown",
        xaxis=dict(title="Respondent share (%)", range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        height=max(400, 40 * len(frame)),
        margin=dict(l=350, r=40, t=80, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.18,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
        ),
        template="plotly_white",
    )
    return fig


def build_trend_figure(section_name: str, trend: Dict[str, List]) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=trend["dates"],
            y=trend["scores"],
            mode="lines+markers",
            line=dict(color="#34495e"),
        )
    )
    fig.update_layout(
        title=f"{section_name} trend",
        yaxis=dict(range=[0, 100], title="Score"),
        xaxis=dict(title="Date"),
        margin=dict(l=60, r=30, t=60, b=40),
        height=280,
        template="plotly_white",
    )
    return fig


def build_pdf_bytes(section: str, payload: Dict) -> bytes:
    frame = pd.DataFrame(payload["distributions"])
    pdf = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "scatter"}],
            [{"type": "bar", "colspan": 2}, None],
        ],
        row_heights=[0.4, 0.6],
        column_widths=[0.45, 0.55],
        vertical_spacing=0.12,
    )

    pdf.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=payload["score"],
            number={"suffix": " %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2b2b2b"},
                "steps": gauge_steps(payload["reverse_gauge"]),
                "threshold": {
                    "line": {"color": "#2b2b2b", "width": 4},
                    "value": payload["score"],
                },
            },
            title={"text": section},
        ),
        row=1,
        col=1,
    )

    trend_dates = payload["trend"]["dates"]
    trend_scores = payload["trend"]["scores"]
    if trend_dates and trend_scores:
        pdf.add_trace(
            go.Scatter(
                x=trend_dates,
                y=trend_scores,
                mode="lines+markers",
                line=dict(color="#34495e"),
                name="Trend",
            ),
            row=1,
            col=2,
        )

    for idx, answer in enumerate(LIKERT_ORDER):
        labels = [
            f"{value:.0f}%"
            if value and not pd.isna(value) and value >= 1
            else ""
            for value in frame[answer]
        ]
        pdf.add_trace(
            go.Bar(
                name=answer,
                y=frame["question"],
                x=frame[answer],
                orientation="h",
                marker=dict(color=COLOR_MAP[answer]),
                showlegend=idx == 0,
                text=labels,
                textposition="inside",
                textfont=dict(color="white", size=12),
            ),
            row=2,
            col=1,
        )

    pdf.update_xaxes(
        title_text="Respondent share (%)", range=[0, 100], row=2, col=1
    )
    pdf.update_yaxes(autorange="reversed", row=2, col=1)
    pdf.update_xaxes(title_text="Date", row=1, col=2)
    pdf.update_yaxes(title_text="Score", range=[0, 100], row=1, col=2)

    pdf.update_layout(
        barmode="stack",
        height=max(950, 60 * len(frame)),
        title={
            "text": f"{section} – snapshot",
            "x": 0.01,
            "xanchor": "left",
        },
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=120, r=60, t=90, b=60),
        template="plotly_white",
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#f8f9fa",
        font=dict(family="Segoe UI, sans-serif"),
    )
    pdf.add_annotation(
        text=(
            f"Responses analysed: {payload['respondents']} | "
            f"Last updated: {payload.get('last_updated') or 'N/A'}"
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=1.08,
        showarrow=False,
        font=dict(size=12, color="#2f3640"),
    )
    return pdf.to_image(format="pdf")


def build_html_snapshot(section: str, payload: Dict) -> str:
    snapshot_date = payload.get("last_updated") or datetime.now().strftime(
        "%d %b %Y"
    )
    respondents = payload.get("respondents", 0)
    score = round(payload.get("score", 0))
    gauge_svg = build_svg_gauge(payload.get("score", 0), payload.get("reverse_gauge", False))
    nav_html = "".join(
        f'<span class="{"active" if tab == section else ""}">{escape(tab)}</span>'
        for tab in NAV_TABS
    )
    def build_segments(entry: Dict) -> str:
        pieces = []
        for answer in LIKERT_ORDER:
            pct = entry.get(answer) or 0
            pct = max(0.0, float(pct))
            if pct <= 0.0:
                continue
            label = f"{pct:.0f}%"
            pieces.append(
                f"<span class='segment {HTML_CLASS_MAP[answer]}' "
                f"style='width:{pct}%;'>{label}</span>"
            )
        if not pieces:
            pieces.append(
                "<span class='segment empty' style='width:100%;'>No data</span>"
            )
        return "".join(pieces)

    bar_blocks = "".join(
        f"""
        <div class="question-row">
            <div class="question-title">{escape(entry["question"])}</div>
            <div class="bar">{build_segments(entry)}</div>
        </div>
        """
        for entry in payload.get("distributions", [])
    )

    legend_html = "".join(
        f"<span><i class='{HTML_CLASS_MAP[level]}'></i>{escape(level)}</span>"
        for level in LIKERT_ORDER
    )

    description = escape(payload.get("description", "")).replace("&#x27;", "’")
    today_label = datetime.now().strftime("%d %b %Y")
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>{escape(section)} snapshot</title>
    <style>
        @font-face {{
            font-family: 'Montserrat';
            font-style: normal;
            font-weight: 400;
            font-display: swap;
            src: url(https://fonts.gstatic.com/s/montserrat/v31/JTUHjIg1_i6t8kCHKm4532VJOt5-QNFgpCtr6Ew-.ttf) format('truetype');
        }}
        body {{
            margin: 0;
            padding: 24px 0;
            font-family: 'Montserrat', sans-serif;
            background: #f2f4f7;
            color: #1f2933;
        }}
        .page {{
            width: 794px;
            min-height: 1123px;
            margin: 0 auto;
            padding: 40px 56px 56px;
            background: #ffffff;
            box-sizing: border-box;
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
        }}
        .nav {{
            display: flex;
            gap: 24px;
            margin-bottom: 20px;
            padding: 0 0 12px 12px;
            border-bottom: 2px solid #eef1f6;
        }}
        .nav span {{
            font-size: 12px;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #98a2b3;
        }}
        .nav span.active {{
            color: #7f56d9;
            font-weight: 600;
        }}
        h1 {{
            margin: 12px 0 10px;
            padding: 4px 0 0 12px;
            font-size: 34px;
            font-weight: 600;
            color: #0f172a;
        }}
        .muted {{
            color: #667085;
            font-size: 13px;
            margin: 6px 0 32px;
            padding-left: 12px;
        }}
        .summary-grid {{
            display: flex;
            justify-content: center;
            margin-bottom: 32px;
        }}
        .gauge {{
            width: 260px;
            height: 140px;
            border-radius: 260px 260px 0 0;
            background: #f8fafc;
            position: relative;
            overflow: hidden;
            margin: 0 auto 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .gauge svg {{
            width: 100%;
            height: auto;
        }}
        .gauge-value {{
            text-align: center;
            font-size: 56px;
            font-weight: 600;
            color: #0f172a;
        }}
        .legend {{
            display: flex;
            gap: 24px;
            margin: 16px auto 28px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
        }}
        .legend span {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #475467;
        }}
        .legend i {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            display: inline-block;
            border: 1px solid rgba(15, 23, 42, 0.08);
        }}
        .legend .red {{ background:#f0524f; }}
        .legend .orange {{ background:#f68f44; }}
        .legend .yellow {{ background:#fbbf24; }}
        .legend .green {{ background:#4ade80; }}
        .legend .dark-green {{ background:#16a34a; }}
        .bars {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .question-row {{
            background: #fdfdfd;
            border-radius: 14px;
            padding: 14px 18px;
            border: 1px solid #eef2f7;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        }}
        .question-title {{
            font-size: 15px;
            margin-bottom: 8px;
            color: #0f172a;
            font-weight: 600;
        }}
        .bar {{
            display: flex;
            height: 22px;
            border-radius: 11px;
            overflow: hidden;
            background: #e4e7ec;
        }}
        .segment {{
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 600;
            color: #0f172a;
        }}
        .segment.red {{ background: #f0524f; color: #fff; }}
        .segment.orange {{ background: #f68f44; color: #fff; }}
        .segment.yellow {{ background: #fef08a; color: #3f3f46; }}
        .segment.green {{ background: #4ade80; color: #065f46; }}
        .segment.dark-green {{ background: #16a34a; color: #fff; }}
        .segment.empty {{ background: #e4e7ec; color:#667085; }}
        .footer-note {{
            margin-top: 30px;
            text-align: right;
            font-size: 12px;
            color: #98a2b3;
        }}
        @media print {{
            body, .page {{
                margin: 0;
                padding: 0;
                background: #ffffff;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <div class="nav">
            {nav_html}
        </div>
        <h1>Insights Summary</h1>
        <div class="muted">{escape(snapshot_date)} · Respondents analysed: {respondents}</div>
        <div class="summary-grid">
            <div>
                <div class="gauge">
                    {gauge_svg}
                </div>
                <div class="gauge-value">{score}</div>
                <div class="muted" style="text-align:center;">{description}</div>
            </div>
        </div>
        <div class="legend">
            {legend_html}
        </div>
        <div class="bars">
            {bar_blocks}
        </div>
        <div class="footer-note">
            Snapshot created {today_label}
        </div>
    </div>
</body>
</html>
"""
    return html_content


def convert_html_to_pdf_bytes(html_content: str) -> bytes:
    global PLAYWRIGHT_INSTALL_ATTEMPTED  # noqa: WPS420
    if os.getenv("DISABLE_STYLED_PDF", "").lower() in {"1", "true", "yes"}:
        raise PlaywrightUnavailable(
            "Styled PDF downloads are disabled via DISABLE_STYLED_PDF."
        )
    factory = get_sync_playwright()
    with factory() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            if not PLAYWRIGHT_INSTALL_ATTEMPTED:
                PLAYWRIGHT_INSTALL_ATTEMPTED = True
                try:
                    subprocess.run(
                        ["playwright", "install", "chromium"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    browser = playwright.chromium.launch()
                except Exception as install_exc:  # noqa: BLE001
                    raise PlaywrightUnavailable(
                        "Styled PDF rendering needs the Playwright Chromium browser. "
                        "Streamlit Cloud usually installs it automatically, but "
                        "if the download fails you can disable the button by "
                        "setting DISABLE_STYLED_PDF=1 in the app's secrets."
                    ) from install_exc
            else:
                raise PlaywrightUnavailable(
                    "Styled PDF rendering needs the Playwright Chromium browser. "
                    "Install it by running `playwright install chromium` "
                    "during deployment, or disable PDF downloads by setting "
                    "DISABLE_STYLED_PDF=1."
                ) from exc
        page = browser.new_page(viewport={"width": 1100, "height": 1600})
        page.set_content(html_content, wait_until="networkidle")
        pdf_bytes = page.pdf(
            width="794px",
            height="1123px",
            print_background=True,
            margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"},
            prefer_css_page_size=True,
        )
        browser.close()
    return pdf_bytes


def build_styled_pdf(section: str, payload: Dict) -> bytes:
    html_content = build_html_snapshot(section, payload)
    return convert_html_to_pdf_bytes(html_content)


def render_key() -> None:
    badges = "".join(
        (
            "<span style='display:inline-flex;align-items:center;margin-right:18px;'>"
            f"<span style='width:18px;height:18px;background:{COLOR_MAP[level]};"
            "display:inline-block;margin-right:6px;border-radius:3px;'></span>"
            f"{level}</span>"
        )
        for level in LIKERT_ORDER
    )
    st.markdown(badges, unsafe_allow_html=True)


def render_section(section_name: str, payload: Dict) -> None:
    st.subheader(section_name)
    st.caption(payload["description"])
    st.write(
        f"Responses analysed: {payload['respondents']} | "
        f"Last updated: {payload.get('last_updated') or 'N/A'}"
    )

    gauge = build_gauge_figure(
        section_name, payload["score"], payload["reverse_gauge"]
    )
    trend = build_trend_figure(section_name, payload["trend"])
    distribution = build_distribution_figure(
        section_name, payload["distributions"]
    )

    st.plotly_chart(gauge, width="stretch")
    st.plotly_chart(trend, width="stretch")
    st.plotly_chart(distribution, width="stretch")

    try:
        styled_pdf = build_styled_pdf(section_name, payload)
        st.download_button(
            label="Download snapshot as PDF",
            data=styled_pdf,
            file_name=f"{section_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )
    except PlaywrightUnavailable as exc:
        st.info(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not generate styled PDF: {exc}")


def main() -> None:
    st.set_page_config(
        page_title="Insights Summary Dashboard", layout="wide"
    )
    st.title("Insights Summary")
    st.write(
        "Upload the latest survey CSV or point the sidebar to a file on disk "
        "to refresh the dashboard."
    )

    st.sidebar.header("Data source")
    uploaded = st.sidebar.file_uploader(
        "Upload CSV export", type="csv", accept_multiple_files=False
    )

    default_path = DEFAULT_DATA_FILE if DEFAULT_DATA_FILE.exists() else None
    default_text = str(default_path) if default_path else ""
    manual_path = st.sidebar.text_input(
        "…or load from a path on disk", value=default_text
    )

    data = None
    source_label = ""

    if uploaded is not None:
        data = load_clean_dataframe(uploaded)
        source_label = uploaded.name
    elif manual_path:
        candidate = Path(manual_path).expanduser()
        if candidate.exists():
            data = load_clean_dataframe(candidate)
            source_label = candidate.name
        else:
            st.sidebar.error(f"No file found at {candidate}")

    if data is None:
        st.info(
            "Provide a CSV via the uploader or sidebar path to populate the dashboard."
        )
        return

    sections = build_section_payload(data)
    if not sections:
        st.warning("No section data detected in the CSV.")
        return

    st.caption(f"Source file: {source_label or 'uploaded file'}")
    render_key()

    tabs = st.tabs(list(sections.keys()))
    for tab, (name, payload) in zip(tabs, sections.items()):
        with tab:
            render_section(name, payload)


if __name__ == "__main__":
    main()
