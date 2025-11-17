#!/usr/bin/env python3
"""
Utility script that mirrors `invoice_1.html` style rendering: point at an HTML file
and capture a polished PDF using Playwright/Chromium.

Usage:
    python convert_to_pdf.py \
        --html insights_summary.html \
        --output insights_summary.pdf
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def convert(html_path: Path, output_path: Path) -> None:
    url = html_path.resolve().as_uri()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")
        time.sleep(1)
        page.pdf(
            path=str(output_path),
            format="A4",
            print_background=True,
            margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"},
            prefer_css_page_size=False,
        )
        browser.close()
    print(f"PDF created successfully: {output_path}")


def parse_args() -> argparse.Namespace:
    default_html = Path("insights_summary.html")
    default_pdf = Path(default_html.stem + ".pdf")
    parser = argparse.ArgumentParser(description="Render HTML files to PDF.")
    parser.add_argument("--html", type=Path, default=default_html, help="HTML file path.")
    parser.add_argument("--output", type=Path, default=default_pdf, help="Desired PDF path.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not arguments.html.exists():
        raise SystemExit(f"HTML file not found: {arguments.html}")
    convert(arguments.html, arguments.output)
