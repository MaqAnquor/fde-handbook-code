# CinemaStream Portfolio Repository

This is the hands-on portfolio codebase built alongside *The Forward Deployed Engineer's Handbook*. It represents a production-grade data and AI system for CinemaStream — a fictional Southeast Asian streaming platform — built chapter by chapter across Parts 0–10 of the book.

**Hiring managers:** Every file here was written as a teaching artifact, not generated boilerplate. The code is explained in the book chapter that introduced it.

---

## What this repo demonstrates

| Skill area | Artifact | Book chapters |
|---|---|---|
| Data engineering | Airflow DAG, dbt project, data contracts | Ch 046–059 |
| ML (churn prediction) | XGBoost pipeline, MLflow tracking, FastAPI serving | Ch 067–077 |
| ML (content tagging) | CNN + transfer learning, RAG pipeline | Ch 080–085 |
| AI systems | RAG assistant, harness, eval suite | Ch 085a–085j |
| Platform engineering | Docker, Kubernetes, CI/CD | Ch 017a, 021a |
| Observability | Runbook, data dictionary, monitoring | Ch 052–053 |
| Consulting delivery | Capstone projects × 4 | Ch 093–093c |

---

## Repository structure

```
cinemastream/
│
├── data/                        ← Datasets (schema-validated, contract-governed)
│   ├── movies.csv               ← 300 movies (multilingual, Southeast Asian titles)
│   ├── users.csv                ← 100 users across SG, MY, ID, PH, TH, VN, IN
│   ├── watch_events.csv         ← 381 streaming events (Ch 055: CDC migration in progress)
│   ├── ratings.csv              ← 500 explicit ratings
│   ├── subscriptions.csv        ← 100 subscription records
│   ├── support_tickets.csv      ← 150 support tickets
│   └── README.md                ← Data dictionary summary
│
├── ml/
│   ├── churn/                   ← Churn prediction system (Ch 067–077)
│   │   ├── train.py             ← XGBoost training pipeline
│   │   ├── features.py          ← Feature engineering
│   │   ├── data_quality.py      ← Great Expectations checks
│   │   ├── tracking.py          ← MLflow experiment logging
│   │   ├── hyperparameter_tuning.py  ← Optuna search
│   │   ├── cross_validation.py  ← Stratified CV
│   │   ├── metrics.py           ← AUC-ROC, precision/recall, log-loss
│   │   ├── model_comparison.py  ← Bias-variance diagnostic
│   │   ├── serialize.py         ← pickle / joblib / ONNX export
│   │   └── serve.py             ← FastAPI prediction endpoint
│   │
│   └── content_tagging/         ← Content tagging system (Ch 080–085)
│       ├── poster_cnn.py        ← CNN poster classifier (ResNet transfer learning)
│       ├── transfer_learning.py ← Feature extraction vs fine-tuning
│       ├── augment.py           ← Image augmentation pipeline
│       ├── text_genre.py        ← Text-based genre classifier
│       ├── multilingual_genre.py ← Multilingual model (SEA languages)
│       ├── nn_intro.py          ← Introductory NN demo
│       ├── training_dynamics.py ← Loss curves, learning rate schedulers
│       ├── mini_llm.py          ← nanoGPT-style LM from scratch (Ch 084a)
│       ├── movie_rag.py         ← RAG pipeline (LlamaIndex + Chroma, Ch 085)
│       └── advanced_rag.py      ← Hybrid RAG + reranking (Ch 085a)
│
├── streamlit_app/               ← Internal analytics dashboard (Ch 060–062)
│   ├── app.py                   ← Main Streamlit app
│   ├── requirements.txt         ← App-specific dependencies
│   └── Dockerfile               ← Container for Streamlit Cloud / Cloud Run
│
├── pipelines/                   ← Orchestrated data pipelines (Ch 046–059)
│   ├── watch_events_dag.py      ← Airflow DAG for watch events ingestion
│   └── subscription_sync_flow.py ← Prefect flow for subscription sync
│
├── dbt_cinemastream/            ← dbt transformation project (Ch 049)
│   ├── dbt_project.yml          ← Project config
│   └── contracts/               ← Data contracts (versioned JSON)
│
├── harness/                     ← Verification harness (Ch 085e–085g)
│   └── harness.py               ← CI gates, golden fixtures, cost controls
│
├── data_hub/                    ← Multi-source query layer (Ch 093)
│   └── hub.py                   ← DuckDB + Polars + API unified interface
│
├── scripts/                     ← Utility and support scripts
│   ├── generate_data.py         ← Synthetic data generator (Ch 024a)
│   ├── cs_plot_style.py         ← CinemaStream matplotlib theme
│   ├── local_llm.py             ← Ollama local LLM setup (Ch 127)
│   ├── api_client.py            ← CinemaStream API client (Ch 019a)
│   ├── async_enrichment.py      ← Async API enrichment (Ch 019b)
│   └── logging_setup.py         ← Structured logging (Ch 013a)
│
├── sql/                         ← SQL query library (Ch 032–041)
│   └── queries/                 ← Reusable analytical queries
│
├── docs/                        ← Living documentation
│   ├── data_dictionary.md       ← Schema reference (updated Ch 054)
│   └── runbook.md               ← On-call runbook (Ch 057)
│
├── images/                      ← Generated plots and illustrations
│   └── 020_data_viz/            ← Matplotlib charts (Ch 020)
│
├── k8s/                         ← Kubernetes manifests (Ch 021a)
├── tests/                       ← Test suite
├── docker-compose.yml           ← Local stack (app + db + Airflow)
├── Dockerfile                   ← Production container
└── pyproject.toml               ← Project metadata and dependencies
```

---

## Running the churn model

```bash
# From repo root, with .venv activated
python cinemastream/ml/churn/train.py          # train + log to MLflow
python cinemastream/ml/churn/serve.py          # start FastAPI endpoint
curl localhost:8000/predict -d '{"user_id": 42}'
```

## Running the RAG pipeline

```bash
python cinemastream/ml/content_tagging/movie_rag.py   # builds index + answers queries
```

## Running the Streamlit dashboard

```bash
streamlit run cinemastream/streamlit_app/app.py
```

## Running the data hub

```bash
python cinemastream/data_hub/hub.py
```

---

## Data notes

- All data is synthetic — generated by `scripts/generate_data.py` with a fixed seed for reproducibility
- `watch_events` has an active data contract migration in progress (Ch 055): the `completed` field is deprecated in favour of `completed_status`; both are valid until the migration window closes
- Row counts: 300 movies, 100 users, 381 watch events, 500 ratings, 100 subscriptions, 150 support tickets

---

## Extending the portfolio

The capstone chapters (Ch 093–093c) show four worked extensions:

1. **Ch 093** — Streamlit data hub + LLM assistant + churn model integrated dashboard
2. **Ch 093a** — Production AI assistant (MCP + RAG + Observability)
3. **Ch 093b** — Autonomous legacy migration pipeline
4. **Ch 093c** — Fully harness-engineered AI system with CI/eval loop

After Part 11 (library reference), **Ch 132** revisits the capstone and shows surgical upgrades using DuckDB, Polars, Optuna, and production LangChain/Qdrant.

The repo is yours to extend. Good starting points:

- Replace the naive RAG in `streamlit_app/app.py` with the hybrid RAG from Ch 085a
- Add a Qdrant collection (Ch 121) and compare retrieval quality vs Chroma
- Wire Evidently (Ch 112) into the churn model's serving layer for drift detection
- Add the Prefect flow (Ch 047a) as an alternative to the Airflow DAG
