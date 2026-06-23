# CinemaStream Book — Google Colab Setup Guide

**For readers who want to run the code with zero local installation.** Colab gives you a free Linux machine with a GPU in the browser — ideal for the deep-learning parts (Part 8) and capstones (Part 11) if you don't have an NVIDIA card.

Read [RUNNING_THE_CODE.md](RUNNING_THE_CODE.md) first for the cross-platform overview, then use this page for Colab specifics.

---

## What is different on Colab

| Topic | Local (Mac/Windows) | Google Colab |
|---|---|---|
| Install location | A `.venv` you create | The Colab runtime itself (ephemeral) |
| Python | You install 3.11–3.13 | Pre-installed (3.11/3.12) |
| GPU | Apple MPS / NVIDIA CUDA | Free **NVIDIA T4 (CUDA)** — must enable it |
| `torch` | pip installs the wheel | Already present; our install reinstalls it (expected, ~2 min) |
| After install | activate venv | **Restart the runtime once** (see below) |
| Persistence | Permanent on disk | **Wiped when the session ends** — reinstall each session |
| `vllm` (Ch 128) | Mac ❌ / Windows-WSL ✅ | ✅ works on a GPU runtime |

The `requirements.txt` and all notebook code are **identical** to local — you do not edit anything.

---

## Step 1 — Get the notebook into Colab

The part notebooks (`part_XX_*.ipynb`) come with your copy of the book. To run one:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook**
3. Choose the `part_XX_*.ipynb` for the part you want, e.g. `part_08_applied_ml_for_deployment.ipynb`

> The companion **code and datasets** are open-source at
> [github.com/MaqAnquor/fde-handbook-code](https://github.com/MaqAnquor/fde-handbook-code)
> — Step 3 and the notes below pull from there. The notebooks themselves ship with
> the book.

---

## Step 2 — Enable the GPU (for Parts 8 & 11 only)

Parts 8 (deep learning) and 11 (vLLM/local LLMs) need a GPU. The data/SQL/ML parts (0–7, 9, 10) run fine on the default CPU runtime.

**Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**

---

## Step 3 — Install the dependencies

The repo isn't on the Colab machine, so first pull `requirements.txt`, then install with the **same command and flag** used everywhere else. Paste this into the first cell:

```python
# Pull the pinned manifest, then install with the legacy resolver (required — see RUNNING_THE_CODE.md)
!wget -q https://raw.githubusercontent.com/MaqAnquor/fde-handbook-code/main/requirements.txt
!pip install -r requirements.txt --use-deprecated=legacy-resolver
```

This takes ~5 minutes. It is normal to see pip reinstall `torch` and a few packages Colab ships — the pins guarantee the versions match the book.

> **Lighter alternative:** if you only need one part, you can skip the full install and let each notebook's own inline `!pip install` cells pull just what that part needs. The full manifest is the safest path for exact reproducibility; the inline path is faster if you're just exploring one chapter.

---

## Step 4 — RESTART the runtime (do not skip)

After a big install that downgrades packages Colab pre-loaded (here: `huggingface-hub`, `tokenizers`, `gradio`), Colab is still holding the **old** versions in memory. You must restart so the new ones load:

**Runtime → Restart session** (or `Ctrl/Cmd+M .`)

Then run your notebook cells from the top — **but do not re-run the install cell.** The packages are already on disk for this session.

If you ever see an import error or a version-mismatch warning mid-notebook, the fix is almost always: **Restart session, run again.**

---

## Step 5 — Verify

```python
import torch, transformers, xgboost, duckdb, polars, langchain, llama_index
print("All packages OK")
print("CUDA available:", torch.cuda.is_available())   # True if you enabled the T4 GPU
```

---

## Colab-specific notes

- **Sessions are ephemeral.** When Colab disconnects (idle timeout or you close the tab), the install is wiped. Next session: re-run Step 3 + Step 4. Save any outputs you care about to Google Drive.
- **The portfolio repo (`cinemastream/`)** isn't on Colab by default. If a notebook imports from `cinemastream.scripts...`, clone the repo first:
  ```python
  !git clone https://github.com/MaqAnquor/fde-handbook-code.git
  %cd fde-handbook-code
  ```
- **Datasets:** the notebooks reference `cinemastream/data/*.csv`. Cloning the repo (above) brings them along.
- **`question-gen` in Ch 125:** the notebook already installs it with `--no-deps` inline — no action needed. See [RUNNING_THE_CODE.md](RUNNING_THE_CODE.md#one-package-is-deliberately-not-installed) for why.
- **Free-tier limits:** Colab free GPUs have usage caps. If you hit one, the CPU runtime still runs every part except the heaviest DL training (which will just be slow).

---

## Quick reference

```python
# === Cell 1: install (run once per session) ===
!wget -q https://raw.githubusercontent.com/MaqAnquor/fde-handbook-code/main/requirements.txt
!pip install -r requirements.txt --use-deprecated=legacy-resolver
# === then: Runtime → Restart session ===

# === Cell 2 (after restart): bring in repo code + data if the part needs it ===
!git clone https://github.com/MaqAnquor/fde-handbook-code.git
%cd fde-handbook-code
```
