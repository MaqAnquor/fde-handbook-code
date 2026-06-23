"""
mini_llm.py — a nano-GPT-style character-level language model, built from
scratch in pure NumPy (no PyTorch, no autograd). Companion to Chapter 084a.

What this demonstrates, end to end:
  1. A character tokenizer (text <-> integer ids).
  2. Next-character training batches (x, y).
  3. A single causal self-attention block (the Chapter 084 mechanism) plus a
     token+position embedding and an output head.
  4. A hand-written training loop: forward -> cross-entropy loss ->
     analytic backward pass -> Adam optimizer step.
  5. Autoregressive sampling (the model generates new text one char at a time).

It trains on two corpora:
  - a tiny toy "colors" corpus (Section 2) — converges to near-zero loss and
    reproduces the pattern exactly, proving the machinery works;
  - CinemaStream's 300 movie titles (Section 3) — learns the *style* of a
    title and invents new ones ("The Silent Signal", "The Last Harbor").

Everything is seeded and deterministic. Run: `python mini_llm.py`
"""
import csv
import time
import numpy as np


# --------------------------------------------------------------------------- #
# 1. Tokenizer
# --------------------------------------------------------------------------- #
class CharTokenizer:
    """Maps every distinct character to an integer id, and back."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# --------------------------------------------------------------------------- #
# 2. Model parameters
# --------------------------------------------------------------------------- #
def init_model(vocab_size, d_model, block_size, seed=0):
    """All trainable weights of the mini-GPT, randomly initialized."""
    rng = np.random.default_rng(seed)
    s = 1.0 / np.sqrt(d_model)
    return {
        "tok_emb": rng.normal(0, 0.02, (vocab_size, d_model)),   # token embedding
        "pos_emb": rng.normal(0, 0.02, (block_size, d_model)),   # position embedding
        "Wq": rng.normal(0, s, (d_model, d_model)),              # query projection
        "Wk": rng.normal(0, s, (d_model, d_model)),              # key projection
        "Wv": rng.normal(0, s, (d_model, d_model)),              # value projection
        "Wo": rng.normal(0, s, (d_model, d_model)),              # attention output
        "Whead": rng.normal(0, s, (d_model, vocab_size)),        # logits head
    }


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --------------------------------------------------------------------------- #
# 3. Forward pass: embed -> causal self-attention -> residual -> logits
# --------------------------------------------------------------------------- #
def forward(p, x, cache=None):
    B, T = x.shape
    D = p["tok_emb"].shape[1]
    h0 = p["tok_emb"][x] + p["pos_emb"][None, :T, :]        # (B,T,D)
    Q = h0 @ p["Wq"]; K = h0 @ p["Wk"]; V = h0 @ p["Wv"]    # (B,T,D)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(D)          # (B,T,T)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)        # forbid the future
    scores = np.where(mask[None], -1e9, scores)
    A = softmax(scores, axis=-1)                            # attention weights
    attn = A @ V                                            # (B,T,D)
    out = attn @ p["Wo"]
    h1 = h0 + out                                           # residual connection
    logits = h1 @ p["Whead"]                                # (B,T,vocab)
    if cache is not None:
        cache.update(dict(x=x, h0=h0, Q=Q, K=K, V=V, A=A, attn=attn, h1=h1))
    return logits


# --------------------------------------------------------------------------- #
# 4. Loss + analytic backward pass (the chain rule, by hand)
# --------------------------------------------------------------------------- #
def loss_and_grads(p, x, y):
    B, T = x.shape
    D = p["tok_emb"].shape[1]
    c = {}
    logits = forward(p, x, cache=c)
    P = softmax(logits, axis=-1)
    ll = -np.log(P[np.arange(B)[:, None], np.arange(T)[None], y] + 1e-12)
    loss = ll.mean()

    # gradient of cross-entropy w.r.t. logits: softmax(logits) - onehot(y)
    dlogits = P.copy()
    dlogits[np.arange(B)[:, None], np.arange(T)[None], y] -= 1.0
    dlogits /= (B * T)

    g = {k: np.zeros_like(v) for k, v in p.items()}
    g["Whead"] = np.einsum("btd,btv->dv", c["h1"], dlogits)
    dh1 = dlogits @ p["Whead"].T
    dh0 = dh1.copy()                       # residual splits the gradient
    dout = dh1
    g["Wo"] = np.einsum("btd,bte->de", c["attn"], dout)
    dattn = dout @ p["Wo"].T
    dA = dattn @ c["V"].transpose(0, 2, 1)
    dV = c["A"].transpose(0, 2, 1) @ dattn
    # softmax backward (row-wise)
    dscores = c["A"] * (dA - (dA * c["A"]).sum(axis=-1, keepdims=True))
    dscores /= np.sqrt(D)
    dQ = dscores @ c["K"]
    dK = dscores.transpose(0, 2, 1) @ c["Q"]
    g["Wq"] = np.einsum("btd,bte->de", c["h0"], dQ)
    g["Wk"] = np.einsum("btd,bte->de", c["h0"], dK)
    g["Wv"] = np.einsum("btd,bte->de", c["h0"], dV)
    dh0 += dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T
    g["pos_emb"][:T] = dh0.sum(axis=0)
    np.add.at(g["tok_emb"], x, dh0)        # scatter-add into embedding rows
    return loss, g


# --------------------------------------------------------------------------- #
# 5. Adam optimizer
# --------------------------------------------------------------------------- #
class Adam:
    def __init__(self, params, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# --------------------------------------------------------------------------- #
# 6. Batching + sampling
# --------------------------------------------------------------------------- #
def get_batch(data, block_size, batch_size, rng):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def generate(p, tok, block_size, n_new, seed_text="", temperature=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    ids = tok.encode(seed_text) if seed_text else [int(rng.integers(0, tok.vocab_size))]
    for _ in range(n_new):
        ctx = np.array(ids[-block_size:])[None, :]    # last block_size tokens
        logits = forward(p, ctx)
        probs = softmax(logits[0, -1] / temperature)  # next-char distribution
        nxt = int(rng.choice(tok.vocab_size, p=probs))
        ids.append(nxt)
    return tok.decode(ids)


# --------------------------------------------------------------------------- #
# Demo: reproduces the canonical outputs used in Chapter 084a
# --------------------------------------------------------------------------- #
def _train(corpus, block_size, d_model, steps, lr, batch=32, log_every=None):
    tok = CharTokenizer(corpus)
    data = np.array(tok.encode(corpus))
    p = init_model(tok.vocab_size, d_model, block_size, seed=0)
    opt = Adam(p, lr=lr)
    rng = np.random.default_rng(1)
    for step in range(1, steps + 1):
        xb, yb = get_batch(data, block_size, batch, rng)
        loss, g = loss_and_grads(p, xb, yb)
        opt.step(p, g)
        if log_every and (step % log_every == 0 or step == 1):
            print(f"  step {step:>4}: loss {loss:.4f}")
    return p, tok


if __name__ == "__main__":
    t0 = time.time()

    # ---- Section 2: toy "colors" corpus ----
    corpus2 = "red green blue yellow " * 200
    tok2 = CharTokenizer(corpus2)
    print("== toy corpus ==")
    print("vocab_size:", tok2.vocab_size)
    print("encode('red'):", tok2.encode("red"))
    p2, _ = _train(corpus2, block_size=16, d_model=32, steps=1200, lr=5e-3,
                   log_every=200)
    print("sample:", repr(generate(p2, tok2, 16, 60, seed_text="red ",
                                    temperature=0.5, rng=np.random.default_rng(7))))

    # ---- Section 3: CinemaStream movie titles ----
    titles = []
    with open("cinemastream/data/movies.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            titles.append(row["title"].strip())
    corpus3 = "\n".join(titles) + "\n"
    tok3 = CharTokenizer(corpus3)
    print("\n== CinemaStream titles ==")
    print("titles:", len(titles), "vocab_size:", tok3.vocab_size)
    p3, _ = _train(corpus3, block_size=32, d_model=64, steps=3000, lr=5e-3,
                   log_every=500)
    print("generated titles:")
    for k in range(6):
        g = generate(p3, tok3, 32, 40, seed_text="The ",
                     temperature=0.6, rng=np.random.default_rng(100 + k))
        print("   ", repr(g.split("\n")[0]))
    print(f"[total {time.time() - t0:.1f}s]")
