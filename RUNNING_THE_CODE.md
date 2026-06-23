# Running the Code — Mac, Windows & Google Colab

This book ships with **runnable code**: 13 Jupyter notebooks (one per part) and a full portfolio repo under `cinemastream/`. This page is the single place that explains **how to run it on any platform** and **why the dependencies are pinned the way they are**, so you can reproduce every example exactly.

> **Where to get the code.** The portfolio code, datasets, and `requirements.txt` are open-source at **[github.com/MaqAnquor/fde-handbook-code](https://github.com/MaqAnquor/fde-handbook-code)**. Clone it and work from the repo root:
> ```bash
> git clone https://github.com/MaqAnquor/fde-handbook-code.git
> cd fde-handbook-code
> ```
> The per-part notebooks come with your copy of the book; run them against this cloned repo.

> **TL;DR** — There is **one** `requirements.txt`. The install command is the **same on every platform**. pip automatically downloads the right build for your OS and hardware (Apple Silicon, Windows x86, Colab's Linux+GPU). The only differences are *where* you run it and one or two platform quirks, documented below.

---

## The one command

```bash
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

That flag is not optional — see [Why the legacy resolver?](#why-the-legacy-resolver) below. It is needed identically on Mac, Windows, and Colab.

**Supported Python: 3.11, 3.12, or 3.13.** Not 3.14 — the pinned compiled wheels (torch, faiss, duckdb, catboost) lag new Python releases and would fall back to failing source builds.

---

## Pick your platform

| Platform | How to set up |
|---|---|
| 🍎 **macOS** (Apple Silicon or Intel) | `python3.13 -m venv .venv` → activate → run the one command. GPU = Apple **MPS** (auto-detected by torch). |
| 🪟 **Windows** (WSL2 — recommended) | Run inside Ubuntu/WSL exactly like Linux. GPU = **CUDA** if you have an NVIDIA card. |
| 🪟 **Windows** (native) | `py -3.13 -m venv .venv` → activate → run the one command. |
| ☁️ **Google Colab** (no install on your machine) | Upload a notebook, run the install cell, **Runtime → Restart**, run all. Free GPU for the deep-learning parts. |

If you just want to *read and run without installing anything*, **Colab is the fastest path** — upload a notebook and run the install cell.

---

## What actually differs between platforms

Everything below is handled automatically by pip choosing the right wheel — you do **not** change `requirements.txt`. This table is so you understand *what's happening*, not steps you perform.

| Concern | macOS | Windows | Google Colab |
|---|---|---|---|
| Python source | `python3.13` (Homebrew/python.org) | `py -3.13` or WSL | Pre-installed (3.11/3.12) |
| Isolation | `.venv` virtualenv | `.venv` virtualenv | The Colab runtime *is* the sandbox |
| GPU backend (torch) | Apple **MPS** | **CUDA** (NVIDIA) or CPU | **CUDA** (free T4) |
| `torch` install | CPU/MPS wheel, auto | CUDA or CPU wheel, auto | Already present; reinstall is expected |
| `vllm` (Ch 128) | ❌ not supported (Linux+CUDA only) | ✅ in WSL+CUDA | ✅ on a GPU runtime |
| After install | activate venv, launch Jupyter | activate venv, launch Jupyter | **must restart the runtime once** |

Check your GPU backend from Python:

```python
import torch
print("CUDA :", torch.cuda.is_available())          # Windows/Colab with NVIDIA
print("MPS  :", torch.backends.mps.is_available())   # Apple Silicon
```

---

## Why the dependencies are pinned (the honest version)

Every version in `requirements.txt` is pinned to an exact number. That is deliberate: it makes a run on your machine in 2027 behave like a run on the author's machine in 2026. But a few pins exist to work around **genuine, irreconcilable upstream conflicts** in the 2026 package ecosystem. If you ever bump these, you will reintroduce the conflict — so here is *why* each one is where it is:

### Why the legacy resolver?

The modern pip resolver (`pip>=20.3`) treats the dependency graph below as unsolvable and aborts with `resolution-too-deep`. The `--use-deprecated=legacy-resolver` flag tells pip to install the explicit pins we provide instead of trying to re-derive the whole graph. The pins are already correct; the flag just stops pip from second-guessing them.

### The four pins that matter

| Pin | The conflict it resolves |
|---|---|
| `gradio==4.44.1` | Gradio 5/6 requires `huggingface-hub>=1.0`, but `transformers<5` requires `huggingface-hub<1.0`. They **cannot coexist**. The book teaches the `transformers` 4.x pipeline API, so Gradio is held at the last 4.x release. |
| `huggingface-hub==0.36.2` | The last `0.x` release. Without this explicit pin, even the legacy resolver pulls `1.x`, which breaks `transformers 4.57.6` (needs `<1.0`). |
| `tokenizers==0.22.2` | `transformers 4.57.6` requires `tokenizers<=0.23.0`. Version `0.23.0` was **never released** (the project jumped `0.22.2 → 0.23.1`), and `0.23.1` breaks the constraint. `0.22.2` is the last compatible build. |
| `torchvision==0.27.1` | Must match `torch==2.12.1` **exactly** — a mismatched pair fails to load shared CUDA/MPS symbols at import time. |

### One package is deliberately *not* installed

`llama-index-question-gen-openai` has **no release** compatible with `llama-index-core>=0.14` (every version of it caps at `core<0.13`). Since the book uses `llama-index==0.14.22`, this package is left out of `requirements.txt`. The one place that needs it — **Chapter 125 (LlamaIndex)** — installs it inline with `--no-deps`, which bypasses the broken constraint:

```python
!pip install llama-index-question-gen-openai --no-deps
```

This is already in the notebook; you don't add it yourself.

### Platform-specific exclusions

- **`vllm` (Ch 128)** is not in `requirements.txt` — it requires Linux + CUDA. Install it separately on a GPU box: `pip install vllm`. On Mac it is skipped entirely.
- **`faiss-cpu`** ships wheels for Windows, macOS (incl. Apple Silicon), and Linux — no source build needed.

---

## Reproducibility snapshot

These examples were authored and verified against this exact stack. If you reproduce on it, every `Output:` block in the book matches.

| | Value |
|---|---|
| Snapshot date | 2026-06-20 |
| Python | 3.13.7 (also valid: 3.11, 3.12) |
| Resolver | pip legacy resolver (`--use-deprecated=legacy-resolver`) |
| Dependency manifest | `requirements.txt` (single source of truth, all platforms) |
| Notebooks | `notebooks/part_00…part_12` — generated from `chapters/` by `build_notebooks.py` |

If a future package release breaks something, **the fix is to install this snapshot, not to upgrade.** The pins are the contract.

---

## Verifying your environment

After installing, confirm the stack is consistent:

```bash
python tools/check_data_contract.py     # datasets match the schema
python tools/check_notebooks_sync.py    # notebooks match the chapters
python tools/qc_notebooks.py --part 9   # execute a part's code end-to-end
```

All three should report success. If `qc_notebooks.py` fails on an import, you almost certainly skipped `--use-deprecated=legacy-resolver` — recreate the venv and reinstall with the flag.
