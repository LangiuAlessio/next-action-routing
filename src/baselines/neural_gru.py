"""
GRU next-action predictor --- a learned (neural) baseline, OUT of the router
pool, to answer "does a learned sequence model beat the classical pool?".

Design choices, kept deliberately fair to the paper's "untuned" ethos:
  * GRU4Rec-style minimal recurrent LM (Hidasi et al. 2016): embedding -> GRU
    -> linear softmax over the action alphabet.
  * ONE fixed configuration used identically on every corpus (no per-corpus
    tuning), mirroring the router, which is also untuned.
  * Same BOUNDED depth-D context as every other expert: predict(ctx) encodes
    only the last D tokens (here D is the window already passed by the driver),
    so the GRU gets no longer history than CTW/PST/Markov.
  * Trained once per LOUO fold on the training users, then FROZEN during
    evaluation (update() is a no-op): a learned model is realistically trained
    offline and deployed; the classical experts keep their online-update
    advantage, so this is a conservative setup for the neural baseline.

Interface matches the other baselines: fit / predict / predict_top / update.
"""
from __future__ import annotations
from typing import Iterable

import torch
import torch.nn as nn

# Fixed configuration (identical across corpora; no tuning).
EMB_DIM = 64
HIDDEN_DIM = 128
N_LAYERS = 1
LR = 1e-3
EPOCHS = 5
BATCH = 256
MAX_CTX = 6           # bounded context depth (same D as the pool experts)
SEED = 0

_BOS = "<bos>"
_UNK = "<unk>"


class _GRULM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM)
        self.gru = nn.GRU(EMB_DIM, HIDDEN_DIM, N_LAYERS, batch_first=True)
        self.out = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, x):                      # x: (B, T) long
        e = self.emb(x)                        # (B, T, E)
        _, h = self.gru(e)                     # h: (layers, B, H)
        return self.out(h[-1])                 # (B, vocab)


class NeuralGRU:
    """Learned recurrent next-action predictor (out-of-pool baseline)."""

    def __init__(self, max_depth: int = MAX_CTX, device: str = "cpu"):
        self.max_depth = max_depth
        self.device = torch.device(device)
        self.model: _GRULM | None = None
        self.stoi: dict[str, int] = {}
        self.itos: list[str] = []
        self._real_idx: list[int] = []         # indices of real actions (not BOS/UNK)

    # ------------------------------------------------------------------
    def _encode_ctx(self, context: list[str]) -> list[int]:
        ctx = context[-self.max_depth:]
        ids = [self.stoi.get(_BOS)]
        for tok in ctx:
            ids.append(self.stoi.get(tok, self.stoi[_UNK]))
        return ids

    def fit(self, sequences: Iterable[list[str]]) -> "NeuralGRU":
        torch.manual_seed(SEED)
        seqs = [s for s in sequences if s]
        vocab = sorted({a for s in seqs for a in s})
        self.itos = [_UNK, _BOS] + vocab
        self.stoi = {a: i for i, a in enumerate(self.itos)}
        self._real_idx = [self.stoi[a] for a in vocab]

        # Build bounded-context training pairs, matching the driver's
        # (ctx = seq[max(0,t-D):t], truth = seq[t+1]) convention exactly.
        X, Y = [], []
        for s in seqs:
            for t in range(len(s) - 1):
                ctx = s[max(0, t - self.max_depth):t]
                X.append(self._encode_ctx(ctx))
                Y.append(self.stoi[s[t + 1]])
        if not X:
            return self

        self.model = _GRULM(len(self.itos)).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        lossf = nn.CrossEntropyLoss()
        pad = self.stoi[_BOS]
        idx = list(range(len(X)))
        g = torch.Generator().manual_seed(SEED)
        self.model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(len(idx), generator=g).tolist()
            for b in range(0, len(perm), BATCH):
                rows = [X[perm[j]] for j in range(b, min(b + BATCH, len(perm)))]
                ys = [Y[perm[j]] for j in range(b, min(b + BATCH, len(perm)))]
                w = max(len(r) for r in rows)
                xb = torch.tensor([r + [pad] * (w - len(r)) for r in rows],
                                  dtype=torch.long, device=self.device)
                yb = torch.tensor(ys, dtype=torch.long, device=self.device)
                opt.zero_grad()
                loss = lossf(self.model(xb), yb)
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, context: list[str]) -> dict[str, float]:
        if self.model is None:
            return {}
        ids = self._encode_ctx(context)
        xb = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.model(xb)[0]                        # (vocab,)
        real = torch.tensor(self._real_idx, device=self.device)
        probs = torch.softmax(logits[real], dim=0).tolist()
        return {self.itos[real[i].item()]: probs[i] for i in range(len(probs))}

    def predict_top(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        return sorted(self.predict(context).items(), key=lambda kv: -kv[1])[:k]

    def update(self, context: list[str], next_action: str) -> None:
        # Static per fold: trained offline, frozen at deployment. No-op.
        return
