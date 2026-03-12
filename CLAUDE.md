# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Welly is a Flask web application for analyzing bacterial growth data (OD600nm) from 96-well and 384-well microplate readers. It generates growth curves, maximum growth rates, area under curve (AUC), and heatmaps. Published in Bioinformatics Advances (2025).

Live demo: http://synbioexplorer.pythonanywhere.com

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run (serves on http://127.0.0.1:5000)
python app.py
```

There is no test suite, linter, or CI/CD pipeline configured.

## Architecture

**Single-file Flask app** (`app.py`, ~735 lines) with Jinja2 templates. No database — uses filesystem-based caching with per-user UUID sessions.

### Request Flow

1. `GET /` — Upload page (`templates/index.html`)
2. `POST /plate_selection` — Parse CSV/Excel, detect plate type (96 or 384), store DataFrame in cache
3. `GET|POST /edit` — Display OD heatmap, allow well-name customization (`edit_96.html` / `edit_384.html`)
4. `GET|POST /results` — Show growth curves with color picker (`results.html`)
5. `GET /download_report` — Generate comprehensive HTML report
6. `GET /download/<filename>` — Download processed CSV
7. `GET /download_heatmap/<filename>` — Download heatmap HTML

### Session & Caching

- Per-user UUID stored in Flask session (`session['user_id']`)
- Cache keys follow pattern: `data_{user_id}`, `plate_type_{user_id}`, `sample_colors_{user_id}`, `well_name_mapping_{user_id}`
- Cache directory: `cache-directory/`, timeout: 4000 seconds

### Key Functions in app.py

- `parse_time()` — Normalizes time formats (HH:MM:SS, Excel serial dates, floats, Timestamps) to minutes
- `calculate_max_growth_rate_per_hour()` — Max dOD/dTime per well
- `calculate_auc()` — Trapezoidal integration via `scipy.integrate.trapz`
- `group_replicates_and_calculate_mean_std()` — Aggregates replicate statistics
- `create_plot()` — Plotly line graphs with error bars (mean +/- std)
- `string_to_pastel_color()` — Deterministic MD5-based pastel color generation

### Templates

Located in `templates/`. Uses Bootstrap 3.3.7 and Plotly CDN for client-side rendering.

### Static Assets

Located in `Static/` (capital S). Contains background images and example data (`Static/examples/example_data.csv`).

## Key Conventions

- Well labels use standard microplate notation (A1-H12 for 96-well, A1-P24 for 384-well)
- Time is converted to minutes internally, displayed as hours
- Duplicate sample names get `_1`, `_2` suffixes for replicate tracking
- Growth rates are clipped to non-negative values
- Generated HTML files (heatmap, plots, reports) are written to the working directory

## Dependencies

The `requirements.txt` is a full environment dump and includes many packages not used by Welly. The actual core dependencies are: Flask, Flask-Caching, Flask-Session, pandas, numpy, scipy, plotly, openpyxl, xlrd, beautifulsoup4, kaleido, chart-studio.
