import gc
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH  = "riddles.csv"
Q_COL      = "Riddle"
A_COL      = "Answer"
MODEL_NAME = "t5-small"
SEEDS      = [0, 1, 2]

TRAIN_FRAC = 0.75
DEV_FRAC   = 0.125      # rest of dataset is test

MAX_INPUT  = 128
MAX_OUTPUT = 16
BATCH_SIZE = 16
LR         = 5e-4
MAX_EPOCHS = 10
PATIENCE   = 3

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def exact_match(gold, pred):
    return float(normalize(gold) == normalize(pred))


def token_f1(gold, pred):
    g, p = normalize(gold).split(), normalize(pred).split()
    if not g or not p:
        return float(g == p)
    common = sum((Counter(g) & Counter(p)).values())
    if common == 0:
        return 0.0
    return 2 * common / (len(g) + len(p))


def score(golds, preds):
    return {
        "EM": np.mean([exact_match(g, p) for g, p in zip(golds, preds)]),
        "F1": np.mean([token_f1(g, p)    for g, p in zip(golds, preds)]),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RiddleDataset(Dataset):
    def __init__(self, questions, answers, tokenizer):
        self.questions = questions
        self.answers   = answers
        self.tok       = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        src = self.tok(f"riddle: {self.questions[idx]}",
                       max_length=MAX_INPUT, truncation=True,
                       padding="max_length", return_tensors="pt")
        tgt = self.tok(normalize(self.answers[idx]),
                       max_length=MAX_OUTPUT, truncation=True,
                       padding="max_length", return_tensors="pt")
        labels = tgt["input_ids"].squeeze()
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids":      src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ---------------------------------------------------------------------------
# Training / inference
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = model(input_ids      = batch["input_ids"].to(device),
                     attention_mask = batch["attention_mask"].to(device),
                     labels         = batch["labels"].to(device)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


def eval_loss(model, loader):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            loss = model(input_ids      = batch["input_ids"].to(device),
                         attention_mask = batch["attention_mask"].to(device),
                         labels         = batch["labels"].to(device)).loss
            total += loss.item()
    return total / len(loader)


def predict(model, tokenizer, questions, batch_size=32):
    model.eval()
    preds = []
    for i in range(0, len(questions), batch_size):
        inputs = tokenizer([f"riddle: {q}" for q in questions[i:i+batch_size]],
                           max_length=MAX_INPUT, truncation=True,
                           padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_OUTPUT)
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Load & split ──────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)[[Q_COL, A_COL]].dropna().reset_index(drop=True)
    print(f"Loaded {len(df)} examples from {DATA_PATH}")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n        = len(df)
    n_train  = int(n * TRAIN_FRAC)
    n_dev    = int(n * DEV_FRAC)
    train_df = df.iloc[:n_train]
    dev_df   = df.iloc[n_train:n_train + n_dev]
    test_df  = df.iloc[n_train + n_dev:].reset_index(drop=True)
    print(f"Split: {len(train_df)} train / {len(dev_df)} dev / {len(test_df)} test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gold_test = test_df[A_COL].tolist()
    test_q    = test_df[Q_COL].tolist()

    # ── Zero-shot baseline ────────────────────────────────────────────────
    print("\nRunning zero-shot baseline...")
    zs_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    zs_preds = predict(zs_model, tokenizer, test_q)
    print(f"Zero-shot  : {score(gold_test, zs_preds)}")
    del zs_model; torch.cuda.empty_cache(); gc.collect()

    # ── Fine-tuning across seeds ──────────────────────────────────────────
    all_preds = []

    for seed in SEEDS:
        set_seed(seed)
        print(f"\n=== Seed {seed} ===")

        train_loader = DataLoader(
            RiddleDataset(train_df[Q_COL].tolist(), train_df[A_COL].tolist(), tokenizer),
            batch_size=BATCH_SIZE, shuffle=True)
        dev_loader = DataLoader(
            RiddleDataset(dev_df[Q_COL].tolist(), dev_df[A_COL].tolist(), tokenizer),
            batch_size=BATCH_SIZE)

        model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        total_steps = MAX_EPOCHS * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = int(0.1 * total_steps),
            num_training_steps = total_steps)

        best_dev_loss, patience_count, best_state = float("inf"), 0, None

        for epoch in range(1, MAX_EPOCHS + 1):
            tr = train_epoch(model, train_loader, optimizer, scheduler)
            dv = eval_loss(model, dev_loader)
            print(f"  Epoch {epoch:2d} | train={tr:.4f} | dev={dv:.4f}")
            if dv < best_dev_loss:
                best_dev_loss  = dv
                patience_count = 0
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print(f"  Early stop at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        preds = predict(model, tokenizer, test_q)
        all_preds.append(preds)
        print(f"  Seed {seed}: {score(gold_test, preds)}")

        del model; torch.cuda.empty_cache(); gc.collect()

    # ── Aggregate & save ──────────────────────────────────────────────────
    avg_em = np.mean([[exact_match(g, p) for g, p in zip(gold_test, ps)] for ps in all_preds], axis=0)
    avg_f1 = np.mean([[token_f1(g, p)    for g, p in zip(gold_test, ps)] for ps in all_preds], axis=0)
    print(f"\nFine-tuned (avg over {len(SEEDS)} seeds): EM={avg_em.mean():.4f} | F1={avg_f1.mean():.4f}")

    pd.DataFrame({
        "question":       test_q,
        "gold":           gold_test,
        "zeroshot_pred":  zs_preds,
        "finetuned_pred": all_preds[0],
        "avg_em":         avg_em,
        "avg_f1":         avg_f1,
    }).to_csv("results.csv", index=False)
    print("Results saved to results.csv")


if __name__ == "__main__":
    main()