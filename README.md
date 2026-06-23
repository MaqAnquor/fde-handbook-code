# The FDE Handbook — Companion Code

This is the runnable **portfolio codebase** that accompanies *The FDE Handbook* by
Abhijeet Verma. It contains the real CinemaStream project built across the book —
data pipelines, SQL, ML models, a RAG assistant, a Streamlit dashboard, dbt,
Kubernetes manifests, and more — so you can run, inspect, and adapt every piece.

> This repository contains **code, datasets, and configuration only**. The book
> text (chapters) is sold separately. See [the book](#get-the-book).

## Quickstart

```bash
git clone https://github.com/MaqAnquor/fde-handbook-code.git
cd fde-handbook-code
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scriptsctivate
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

Then run anything in the portfolio, e.g.:

```bash
python -m cinemastream.scripts.cs_plot_style     # plotting style module
python cinemastream/scripts/load_orders.py       # example data script
```

Datasets live in `cinemastream/data/*.csv`. Code imports as the `cinemastream`
package (run commands from the repo root).

### Running on Google Colab

No local install needed — see [COLAB_SETUP.md](COLAB_SETUP.md). Platform-specific
guides: [macOS](MACOS_SETUP.md) · [Windows (WSL)](WINDOWS_WSL_GUIDE.md) ·
[Windows](WINDOWS_GUIDE.md) · [all platforms](RUNNING_THE_CODE.md).

## What's inside

| Path | What it is |
|---|---|
| `cinemastream/data/` | CinemaStream datasets (movies, users, watch events, ratings, subscriptions, tickets) |
| `cinemastream/scripts/` | Data loading, currency API, synthetic data generation, plot style |
| `cinemastream/ml/churn/` | Churn prediction model + FastAPI serving |
| `cinemastream/ml/content_tagging/` | Content tagging model + movie RAG pipeline |
| `cinemastream/data_hub/` | Multi-source query layer |
| `cinemastream/pipelines/` | Airflow DAGs |
| `cinemastream/dbt_cinemastream/` | dbt project (warehouse models) |
| `cinemastream/streamlit_app/` | Internal dashboard |
| `cinemastream/harness/` | Verification harness |
| `cinemastream/sql/` | SQL schemas and queries |
| `cinemastream/k8s/` | Kubernetes manifests |
| `cinemastream/tests/` | Tests |

## Get the book

*The FDE Handbook* — 176 chapters from `print("hello world")` to production AI
systems. **Link coming soon.**

## License

See [LICENSE.txt](LICENSE.txt). You may use and adapt this code in your own
private or commercial projects; you may **not** repackage or resell it as a
competing educational product, course, or book.

---
*Published by AV Press · © 2026 Abhijeet Verma*
