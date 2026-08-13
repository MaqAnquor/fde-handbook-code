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
| **macOS** (Apple Silicon or Intel) | `python3.13 -m venv .venv` → activate → run the one command. GPU = Apple **MPS** (auto-detected by torch). |
| **Windows** (WSL2 — recommended) | Run inside Ubuntu/WSL exactly like Linux. GPU = **CUDA** if you have an NVIDIA card. |
| **Windows** (native) | `py -3.13 -m venv .venv` → activate → run the one command. |
| **Google Colab** (no install on your machine) | Upload a notebook, run the install cell, **Runtime → Restart**, run all. Free GPU for the deep-learning parts. |

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
| `vllm` (Ch 128) | not supported (Linux+CUDA only) | supported in WSL+CUDA | supported on a GPU runtime |
| After install | activate venv, launch Jupyter | activate venv, launch Jupyter | **must restart the runtime once** |

Check your GPU backend from Python:

```python
import torch
print("CUDA :", torch.cuda.is_available())          # Windows/Colab with NVIDIA
print("MPS  :", torch.backends.mps.is_available())   # Apple Silicon
```

### Newer NVIDIA GPUs (Ada / Blackwell) — the `cu128` wheel trap

If you have a **recent NVIDIA card — especially a 50-series (Blackwell, compute capability
`sm_120`)** — read this before you fight a confusing error. It is the single most common GPU
setup failure, and it is easy to avoid once you know the rule.

**The trap.** pip's *default* torch wheel does not include compiled kernels for the newest GPUs.
On such a card `torch.cuda.is_available()` returns `True`, and then the first real GPU operation
dies with:

```
CUDA error: no kernel image is available for execution on the device
```

`is_available()` is necessary but **not sufficient** — it only checks a driver exists, not that
your torch build has kernels for your specific card.

**The fix — install the CUDA build that matches your card.** First read your compute capability:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv     # e.g. "RTX 5060 Ti, 12.0" → sm_120
```

Then install torch from the matching CUDA index. For any NVIDIA GPU from **Turing (RTX 20xx)
through Blackwell (RTX 50xx)**, the `cu128` build (PyTorch ≥ 2.7) has the kernels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Prove it works — run a real operation, not just `is_available()`:**

```python
import torch
print("built for :", torch.cuda.get_arch_list())        # must contain your sm_XXX (e.g. 'sm_120')
a = torch.randn(1024, 1024, device="cuda")
print("real op   :", (a @ a).sum().item())               # if THIS runs, the GPU truly works
```

| Your GPU | Example cards | `sm_` | What to install |
|---|---|---|---|
| Turing / Ampere / Ada | RTX 20xx–40xx, A100 | 75–89 | `cu128` index (default wheel often works too) |
| **Blackwell** | **RTX 50xx**, B200 | **120 / 100** | **`cu128` index — required** |
| Pascal & older | GTX 10xx and down | ≤ 61 | Too old for recent torch → use **Colab** |
| No NVIDIA GPU | Mac, most laptops | — | Colab (free T4) or CPU — see `COLAB_SETUP.md` |

> **Note on the pinned `torch==2.12.1`.** That pin gives you correct, reproducible **CPU** output on
> every platform. If you want to *use* an NVIDIA GPU, install the `cu128` build over the top as shown
> above — it may resolve to a slightly different torch version (e.g. 2.11.x), and that is fine: the
> deep-learning / RAG / LLM outputs are **representative, not byte-reproducible** (your numbers will
> differ by card and driver — reproduce the *shape* of the result, not the last decimal).

---

## Why the dependencies are pinned (the honest version)

Every version in `requirements.txt` is pinned to an exact number. That is deliberate: it makes a run on your machine in 2027 behave like a run on the author's machine in 2026. But a few pins exist to work around **genuine, irreconcilable upstream conflicts** in the 2026 package ecosystem. If you ever bump these, you will reintroduce the conflict — so here is *why* each one is where it is:

### Why the legacy resolver?

The modern pip resolver (`pip>=20.3`) treats the dependency graph below as unsolvable and aborts with `resolution-too-deep`. The `--use-deprecated=legacy-resolver` flag tells pip to install the explicit pins we provide instead of trying to re-derive the whole graph. The pins are already correct; the flag just stops pip from second-guessing them.

### The five pins that matter

| Pin | The conflict it resolves |
|---|---|
| `transformers==5.15.0` | The 5.0 release removed the seq2seq pipeline tasks (`summarization`, `translation`, `text2text-generation`). The book uses the explicit `AutoTokenizer` + `AutoModelForSeq2SeqLM` + `.generate()` API (Ch 123 §2), which is unaffected — so it can stay on 5.x and take the security fixes. |
| `gradio==6.22.0` | Gradio 5/6 requires `huggingface-hub>=1.0`, which `transformers` 4.x forbade. With `transformers` on 5.x that conflict is gone, so Gradio runs current. Ch 110 is executed against this version. |
| `huggingface-hub==1.27.0` | `transformers 5.15.0` requires `>=1.5.0,<2.0`; Gradio 6 also needs `1.x`. Pinned so the legacy resolver cannot drift it. |
| `tokenizers==0.22.2` | `transformers 5.15.0` requires `tokenizers>=0.22.0,<=0.23.0`. Version `0.23.0` was **never released** (the project jumped `0.22.2 → 0.23.1`), and `0.23.1` breaks the constraint. `0.22.2` is the last compatible build. |
| `torchvision==0.27.1` | Must match `torch==2.12.1` **exactly** — a mismatched pair fails to load shared CUDA/MPS symbols at import time. |

### One package is deliberately *not* installed

`llama-index-question-gen-openai` has **no release** compatible with `llama-index-core>=0.14` (every version of it caps at `core<0.13`). Since the book uses `llama-index==0.14.22`, this package is left out of `requirements.txt`. The one place that needs it — **Chapter 125 (LlamaIndex)** — installs it inline with `--no-deps`, which bypasses the broken constraint:

```python
!pip install llama-index-question-gen-openai --no-deps
```

This is already in the notebook; you don't add it yourself.

### Platform-specific exclusions

- **`vllm` (Ch 128)** is not in `requirements.txt` — it requires Linux + CUDA. Install it separately on a GPU box: `pip install vllm`. On Mac it is skipped entirely.
- **`timesfm` / `tabfm` (Ch 077c)** are not in `requirements.txt` — the foundation-model chapter runs in its **own fresh venv** (`python3.13 -m venv .venv-fm` → `pip install timesfm==2.0.1 tabfm==1.0.0 torch xgboost scikit-learn pandas`). Their dependency trees (newer numpy/pandas/huggingface-hub) conflict with the book's pinned stack; do not install them into the main `.venv`. The TabFM checkpoint download is ~12 GB.
- **`faiss-cpu`** ships wheels for Windows, macOS (incl. Apple Silicon), and Linux — no source build needed.

---

## Reproducibility snapshot

These examples were authored and verified against this exact stack. If you reproduce on it, every `Output:` block in the book matches.

| | Value |
|---|---|
| Snapshot date | 2026-06-20 |
| Python | 3.13.9 (also valid: 3.11, 3.12) |
| Resolver | pip legacy resolver (`--use-deprecated=legacy-resolver`) |
| Dependency manifest | `requirements.txt` (single source of truth, all platforms) |
| Exact lock | `requirements.lock` — all 385 packages hash-pinned (macOS arm64 / CPython 3.13.9) |
| Notebooks | `notebooks/part_00…part_12` — generated from `chapters/` by `build_notebooks.py` |

If a future package release breaks something, **the fix is to install this snapshot, not to upgrade.** The pins are the contract.

For a **byte-identical** reproduction of the author's environment (Apple Silicon),
install the hash-pinned lock instead of `requirements.txt`:

```bash
pip install --require-hashes -r requirements.lock
```

On Linux/Windows, install from `requirements.txt` (some locked wheels are
platform-specific); the pins still guarantee output-matching versions.

### Matching the book's output exactly (determinism)

A few chapters print **sets/dictionaries** (whose order depends on Python's hash
seed) or run multi-threaded math. To get output that matches the book character-for-
character, set these two environment variables before running — they're how the book's
outputs were captured:

```bash
export PYTHONHASHSEED=0      # fixes set/dict iteration order
export OMP_NUM_THREADS=1     # deterministic reduction order for NumPy/XGBoost/LightGBM
```

Without them your code still runs correctly — only the *order* of a printed set, or the
last digit of some floating-point results, may differ. (Outputs in the **deep-learning,
RAG, and LLM** chapters depend on models, embeddings, or live services and are
illustrative — they are not expected to reproduce byte-for-byte.)

### Exact reproduction with Docker (optional, zero setup)

If you'd rather not touch your own Python at all, run inside the book's **canonical
image** — Linux/amd64, the exact pinned dependencies, and the determinism settings
already baked in. Your **code and data come from this repo** (mounted); the
**environment comes from the image**, so the gated chapters reproduce byte-for-byte.

```bash
# 1. clone this companion repo and cd into it (if you haven't already)
git clone https://github.com/MaqAnquor/fde-handbook-code.git && cd fde-handbook-code

# 2. pull the image (the --platform flag is required on Apple Silicon Macs)
docker pull --platform linux/amd64 ghcr.io/maqanquor/fde-handbook:latest

# 3. launch JupyterLab on the repo, then open http://localhost:8888
docker run --rm --platform linux/amd64 -p 8888:8888 -v "$PWD":/work ghcr.io/maqanquor/fde-handbook:latest
```

Open any notebook under `notebooks/` and run it — `PYTHONHASHSEED=0` and
`OMP_NUM_THREADS=1` are already set inside the container, so set/dict ordering and
numeric output match the book.

**Why `--platform linux/amd64`?** The image is `linux/amd64` only — that is the
canonical architecture the book's outputs are gated against, so an emulated amd64 run
on Apple Silicon reproduces the book's numbers exactly (a native arm64 build would
not). Without the flag, an Apple Silicon Mac fails with *"no matching manifest for
linux/arm64"*. Intel Macs, Linux, and Windows/WSL are amd64 already, so the flag is a
harmless no-op there — keep it in the command and it works everywhere.

**Platform tested.** Verified end-to-end on **macOS (Apple Silicon) with Docker
Desktop** — anonymous `pull`, `run`, and a gated chapter reproduced byte-for-byte
through amd64 emulation. The image is built and published by GitHub Actions (a Linux
runner), so it's identical wherever you pull it. On the one Windows/NVIDIA machine used
during this book's own development, Docker Desktop's WSL2 backend would not start at
all, so Docker on Windows is an **untested path** for this book, not a confirmed one.
If `docker run` fails on Windows, that's very likely a Docker Desktop/WSL2 setup issue
on your machine rather than anything wrong with the image — the native `pip install`
path above doesn't need Docker at all and is the more proven route on Windows.

**Scope:** this lean image covers **Parts 0–7 and the classic-ML chapters** (the
deterministic, byte-reproducible core). The **deep-learning, RAG, and LLM** chapters
need heavier dependencies (PyTorch/transformers) or live services and are illustrative —
run those from `requirements.txt` on a machine with the right hardware.

---

## Verifying your environment

After installing, a quick sanity check:

```bash
python -c "import numpy, pandas, sklearn, torch; print('core imports OK')"
```

Then open any notebook under `notebooks/` and run it top to bottom. If the printed
output matches that chapter's `Output:` blocks in the book, your environment is
reproducing the deterministic core correctly.

Two things **not** to worry about:

- `pip check` will report a handful of version conflicts (`websockets`, `typer`).
  That is expected — those are the same declared-but-harmless conflicts the legacy
  resolver exists to step around (see "Why the legacy resolver?" above). The pinned
  versions work together in practice.
- If an **import** fails, you almost certainly installed without
  `--use-deprecated=legacy-resolver` — delete the venv and reinstall with the flag.
