#!/usr/bin/env python3
"""
Transformer + Safety-LSTM pipeline
Prints and logs metrics for horizons 10 … 100 (step 10)
"""

import os, sys, time, math, torch, pandas as pd, numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

# ─── project imports ────────────────────────────────────────────────────
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from vae import VAE  # noqa: E402
import evaluator.lstm as evaluator  # noqa: E402
from data_loader import load_embed_label_list  # per-file loader  (memory-safe)

# ─── constants ──────────────────────────────────────────────────────────
LATENT_DIM = 32
SEQ_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = "./reliability_results"
os.makedirs(RESULT_DIR, exist_ok=True)

TRAIN_FILES = range(1, 201)
EVAL_FILES = range(201, 300)


# ─── data streaming helper ─────────────────────────────────────────────
def stream_records(file_range, seq_len, horizon):
    for file_no in file_range:
        try:
            rec = load_embed_label_list(file_no)
        except FileNotFoundError:
            print(f"[warn] {file_no}.npz not found — skipping")
            continue
        if len(rec) < seq_len + horizon:
            continue

        for i in range(len(rec) - seq_len - horizon):
            past = rec[i : i + seq_len]
            fut = rec[i + horizon : i + seq_len + horizon]

            yield (
                torch.stack([d["embedding"] for d in past]),  # [T, 32]
                torch.stack([d["embedding"] for d in fut]),  # [T, 32]
                [d["label"] for d in fut],  # list[bool]
            )
        del rec


# ─── Transformer model ────────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim=LATENT_DIM,
        num_heads=4,
        num_layers=4,
        mlp_hidden=512,
        dropout=0.05,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_hidden,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        return self.encoder(x)


# ─── training ----------------------------------------------------------
def train_one_epoch(model, optimizer, horizon):
    crit = nn.MSELoss(reduction="sum")
    model.train()
    tot, steps = 0.0, 0
    for past, fut_e, _ in stream_records(TRAIN_FILES, SEQ_LEN, horizon):
        past = past.unsqueeze(0).to(DEVICE)
        fut_e = fut_e.unsqueeze(0).to(DEVICE)

        out = model(past)
        loss = crit(out, fut_e)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot += loss.item()
        steps += 1
    return tot / max(steps, 1)


def train_model(model, horizon, epochs=30, lr=1e-3):
    model.to(DEVICE)
    optim_ = optim.Adam(model.parameters(), lr=lr)
    best = math.inf
    for ep in range(1, epochs + 1):
        loss = train_one_epoch(model, optim_, horizon)
        if loss < best:
            best = loss
            torch.save(
                model.state_dict(), f"./weights/transformer_weights{horizon}.pth"
            )
        print(f"[h={horizon:3}] Epoch {ep:02}/{epochs}  loss {loss:.4f}")
    print(f"[h={horizon:3}] Best loss {best:.4f}")


# ─── evaluation --------------------------------------------------------
@torch.no_grad()
def run_eval(model, safety_lstm, horizon):
    model.eval()
    safety_lstm.eval()

    preds, labels = [], []
    for past, _, fut_lbl in stream_records(EVAL_FILES, SEQ_LEN, horizon):
        past = past.unsqueeze(0).to(DEVICE)
        prob = safety_lstm(model(past))[0, -1, 0].item()
        preds.append(1 if prob > 0.5 else 0)
        labels.append(int(fut_lbl[-1]))

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)

    return acc, f1, fpr, fnr, prec, rec


# ─── MAIN LOOP (horizon sweep) ─────────────────────────────────────────
if __name__ == "__main__":
    # load pre-trained safety LSTM
    safety = evaluator.LSTM().to(DEVICE)
    safety.load_state_dict(
        torch.load("../evaluator/weights/lstm_weights.pth", map_location=DEVICE)
    )

    horizons = range(10, 101, 10)  # 10, 20, …, 100
    seq_len = SEQ_LEN
    epochs = 100

    log_file = os.path.join(RESULT_DIR, "accuracy.txt")
    with open(log_file, "a") as log:
        for h in horizons:
            print(f"\n════════ Horizon {h} ─ SeqLen {seq_len} ════════")

            transformer = Transformer().to(DEVICE)
            # --- TRAIN ---------------------------------------------------
            # train_model(transformer, horizon=h, epochs=epochs)

            # (use pre-trained weights if you already have them)
            train_model(transformer, horizon=h, epochs=epochs)

            # --- EVALUATE -----------------------------------------------
            acc, f1, fpr, fnr, p, r = run_eval(transformer, safety, horizon=h)

            # console
            print(
                f"ACC {acc:.4f} | F1 {f1:.4f} | P {p:.4f} | R {r:.4f} "
                f"| FPR {fpr:.4f} | FNR {fnr:.4f}"
            )

            # append to log file
            log.write(
                f"\n════════ Horizon {h} ─ SeqLen {seq_len} ════════"
                f"H={h:3}  Seq={seq_len}  "
                f"ACC {acc:.4f}  F1 {f1:.4f}  "
                f"P {p:.4f}  R {r:.4f}  "
                f"FPR {fpr:.4f}  FNR {fnr:.4f}\n"
            )
