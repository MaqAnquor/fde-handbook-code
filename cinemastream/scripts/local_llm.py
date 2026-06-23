#!/usr/bin/env python3
"""
local_llm.py — a FREE, no-API-key LLM for readers without an Anthropic/OpenAI key.

Why this exists
---------------
Most LLM chapters only need "an LLM that completes a prompt." Paid APIs (Claude,
GPT) give the best quality, but not every reader can get a key. This module wraps
a small, fully open HuggingFace instruct model so every chapter's *mechanics*
(prompting, RAG, tool/agent loops) run with zero keys and zero cost.

- **Default model:** ``Qwen/Qwen2.5-0.5B-Instruct`` — ~0.5B params, Apache-2.0,
  not gated, runs on CPU (slow-ish) and is fast on a Colab GPU runtime.
- **Trade-off (be honest with readers):** a 0.5B model is far weaker than Claude
  or GPT-4. Answers are rougher and it will sometimes be wrong. That's fine for
  *learning the plumbing* — the prompt structure, retrieval, and agent control
  flow are identical; only the answer quality differs. For production, swap in a
  hosted model.

Usage
-----
    from cinemastream.scripts.local_llm import complete
    print(complete("Name three churn-reduction tactics for a streaming service."))

    # With a system prompt:
    complete("Which country churns most?", system="You are a data analyst. Be terse.")

First call downloads the model (~1 GB) to the HF cache; later calls reuse it.
On Colab, set HF_HOME to a mounted Drive folder to persist across sessions.
"""
from functools import lru_cache

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@lru_cache(maxsize=2)
def _pipeline(model_name: str = DEFAULT_MODEL):
    """Build (once) and cache a text-generation pipeline. Uses a GPU if present."""
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-generation",
        model=model_name,
        dtype=torch.float32,
        device=device,
    )


def complete(prompt: str, system: str | None = None,
             max_new_tokens: int = 256, temperature: float = 0.7,
             model_name: str = DEFAULT_MODEL) -> str:
    """Return the model's text completion for ``prompt``.

    A drop-in stand-in for a paid chat API: pass a user ``prompt`` (and optional
    ``system`` instruction); get back the assistant's text. Deterministic-ish
    with ``temperature=0``.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    out = _pipeline(model_name)(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=_pipeline(model_name).tokenizer.eos_token_id,
    )
    # chat pipelines return the full message list; the last item is the reply
    return out[0]["generated_text"][-1]["content"].strip()


if __name__ == "__main__":
    print("Loading free local model:", DEFAULT_MODEL)
    print(complete("In one sentence, what is customer churn?",
                   system="You are CinemaStream's data analyst.",
                   max_new_tokens=60, temperature=0.0))
