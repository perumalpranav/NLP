import re
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from nltk import Tree
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SEED = 16
nlp = spacy.load("en_core_web_sm")

# ---- Complexity Measures ----

def clause_density(sentence: str) -> float:
    if not sentence:
        return 0.0
    clause_markers = [
        "that", "which", "who", "whom", "because", "although",
        "when", "while", "if", "since", "after", "before"
    ]
    words = sentence.lower().split()
    clause_count = sum(1 for w in words if w in clause_markers)
    return clause_count / max(len(words), 1)

def subject_verb_distance(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            verb = token
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    return abs(verb.i - child.i)
    return None

def dependency_tree_depth(sentence):
    doc = nlp(sentence)
    def subtree_depth(token):
        if not list(token.children):
            return 1
        return 1 + max(subtree_depth(child) for child in token.children)
    roots = [token for token in doc if token.dep_ == "ROOT"]
    if not roots:
        return 0
    return subtree_depth(roots[0])

# ---- Perturbation Methods ----

def perturb_relative_clause(record: dict, field: str = "sentence1") -> str:
    """'The tall man ran' → 'The man who was tall ran'"""
    sentence = record[field]
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "ADJ" and token.dep_ == "amod":
            noun = token.head
            original = f"{token.text} {noun.text}"
            replacement = f"{noun.text} who was {token.text}"
            if original in sentence:
                return sentence.replace(original, replacement, 1)
    return sentence

def perturb_topicalization(record: dict, field: str = "sentence1") -> str:
    """Moves the object NP to the front of the sentence."""
    sentence = record[field].rstrip(". ")
    parse = record[field + "_parse"]
    vp_block = re.search(r"\(VP\s+(\(.*?\)\s*)+\)", parse)
    if not vp_block:
        return record[field]
    np_in_vp = re.search(r"\(NP\s+(\(.*?\)\s*)+\)", vp_block.group(0))
    if not np_in_vp:
        return record[field]
    obj_words = re.findall(r"\(\w+\s+(\w+)\)", np_in_vp.group(0))
    obj_text = " ".join(obj_words)
    if not obj_text or obj_text not in sentence:
        return record[field]
    remainder = sentence.replace(obj_text, "").strip().rstrip(",")
    return f"{obj_text.capitalize()}, {remainder[0].lower() + remainder[1:]}."

def perturb_cleft(record: dict, field: str = "sentence1") -> str:
    sentence = record[field].rstrip(". ")
    parse = record[field + "_parse"]
    np_block = re.search(r"\(NP\s+(\(.*?\)\s*)+\)", parse)
    if not np_block:
        return record[field]
    subj_words = re.findall(r"\(\w+\s+(\w+)\)", np_block.group(0))
    subject = " ".join(subj_words)
    if not subject or subject not in sentence:
        return record[field]
    rest = sentence[sentence.index(subject) + len(subject):].strip()
    return f"It was {subject.lower()} that {rest}."

# ---- Dataset ----

matched_df = pd.read_json("./multinli_1.0_dev_matched.jsonl", lines=True)
mismatched_df = pd.read_json("./multinli_1.0_dev_mismatched.jsonl", lines=True)

matched_df["split"] = "matched"
mismatched_df["split"] = "mismatched"

N = 8000
matched_sample = matched_df.sample(n=N, random_state=SEED)
mismatched_sample = mismatched_df.sample(n=N, random_state=SEED)

combined_df = pd.concat([matched_sample, mismatched_sample], ignore_index=True)

combined_df["sentence1_rel"]   = combined_df.apply(lambda x: perturb_relative_clause(x, "sentence1"), axis=1)
combined_df["sentence2_rel"]   = combined_df.apply(lambda x: perturb_relative_clause(x, "sentence2"), axis=1)
combined_df["sentence1_cleft"] = combined_df.apply(lambda x: perturb_cleft(x, "sentence1"), axis=1)
combined_df["sentence2_cleft"] = combined_df.apply(lambda x: perturb_cleft(x, "sentence2"), axis=1)
combined_df["sentence1_topic"] = combined_df.apply(lambda x: perturb_topicalization(x, "sentence1"), axis=1)
combined_df["sentence2_topic"] = combined_df.apply(lambda x: perturb_topicalization(x, "sentence2"), axis=1)

# ---- Complexity Metrics ----

rows = []
for _, row in tqdm(combined_df.iterrows()):
    new_row = {}
    for col in ["sentence1", "sentence1_rel", "sentence1_cleft", "sentence1_topic",
                "sentence2", "sentence2_rel", "sentence2_cleft", "sentence2_topic"]:
        sentence = row[col]
        new_row[f"{col}_clause_density"]  = clause_density(sentence)
        new_row[f"{col}_dep_depth"]       = dependency_tree_depth(sentence)
        new_row[f"{col}_subj_verb_dist"]  = subject_verb_distance(sentence)
    rows.append(new_row)

complexity_df = pd.DataFrame(rows)
complexity_df.index = combined_df.index

column_means = complexity_df.mean(numeric_only=True)
print(column_means)

# ---- Plot 1: Complexity Bar Charts ----

metrics        = ["clause_density", "dep_depth", "subj_verb_dist"]
metric_labels  = ["Clause Density", "Dependency Tree Depth", "Subject-Verb Distance"]
bases          = ["sentence1", "sentence2"]
variants       = ["", "_rel", "_cleft", "_topic"]
variant_labels = ["base", "rel", "cleft", "topic"]
colors         = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, axes = plt.subplots(3, 2, figsize=(13, 13))
for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
    for j, base in enumerate(bases):
        ax = axes[i, j]
        cols   = [f"{base}{v}_{metric}" for v in variants]
        values = column_means[cols]
        bars   = ax.bar(variant_labels, values, color=colors)
        ax.set_title(f"{base.replace('sentence','Sentence ')} — {mlabel}", fontsize=11)
        ax.set_ylabel(mlabel)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

plt.suptitle("Complexity Metrics Before and After Perturbation", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("plot1_complexity_bars.png", bbox_inches="tight", dpi=150)
plt.close()

# ---- Plot 2: Complexity Delta Table (printed + saved as figure) ----

delta_rows = []
for metric in metrics:
    for base in bases:
        base_col = f"{base}_{metric}"
        for v, vl in zip(variants[1:], variant_labels[1:]):
            pert_col = f"{base}{v}_{metric}"
            delta    = column_means[pert_col] - column_means[base_col]
            delta_rows.append({"sentence": base, "metric": metric, "perturbation": vl, "delta": delta})

delta_df = pd.DataFrame(delta_rows)
print("\nComplexity Deltas (perturbed − base):")
print(delta_df.pivot_table(index=["sentence","metric"], columns="perturbation", values="delta").to_string())

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
pivot = delta_df.pivot_table(index=["sentence","metric"], columns="perturbation", values="delta")
tbl   = ax.table(
    cellText=pivot.round(4).values,
    rowLabels=[f"{r[0]} / {r[1]}" for r in pivot.index],
    colLabels=pivot.columns.tolist(),
    cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)
ax.set_title("Complexity Delta (Perturbed − Base)", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("plot2_complexity_delta_table.png", bbox_inches="tight", dpi=150)
plt.close()

# ---- Model Inference ----

device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "deberta_mnli":  "cross-encoder/nli-deberta-v3-xsmall",
    "tiny_bert_mnli":"prajjwal1/bert-tiny-mnli",
    "electra_mnli":  "howey/electra-small-mnli",
    "nli_miniLM":    "cross-encoder/nli-MiniLM2-L6-H768",
}

VARIANTS = {
    "base":  ("sentence1",       "sentence2"),
    "rel":   ("sentence1_rel",   "sentence2_rel"),
    "cleft": ("sentence1_cleft", "sentence2_cleft"),
    "topic": ("sentence1_topic", "sentence2_topic"),
}

LABEL_MAPS = {
    "electra_mnli":    {0: "contradiction", 1: "entailment",  2: "neutral"},
    "bart_mnli":       {0: "contradiction", 1: "neutral",      2: "entailment"},
    "mobilebert_mnli": {0: "entailment",    1: "neutral",      2: "contradiction"},
    "nli_miniLM":      {0: "contradiction", 1: "entailment",  2: "neutral"},
    "distilbert_mnli": {0: "entailment",    1: "neutral",      2: "contradiction"},
    "albert_mnli":     {0: "entailment",    1: "neutral",      2: "contradiction"},
    "deberta_mnli":    {0: "contradiction", 1: "entailment",  2: "neutral"},
    "roberta_mnli":    {0: "contradiction", 1: "entailment",  2: "neutral"},
    "squeezebert_mnli":{0: "entailment",    1: "neutral",      2: "contradiction"},
    "tiny_bert_mnli":  {0: "entailment",    1: "neutral",      2: "contradiction"},
}

def get_predictions(model, tokenizer, s1_list, s2_list, label_map, batch_size=32):
    all_preds = []
    for i in range(0, len(s1_list), batch_size):
        s1_batch = s1_list[i:i+batch_size]
        s2_batch = s2_list[i:i+batch_size]
        inputs = tokenizer(s1_batch, s2_batch, return_tensors="pt",
                           truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        all_preds.extend([label_map[idx] for idx in logits.argmax(dim=-1).tolist()])
    return all_preds

def accuracy(gold, pred):
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)

# Filter valid labels
df_valid          = combined_df[combined_df["gold_label"].isin(["entailment", "neutral", "contradiction"])].copy()
df_valid          = df_valid.reset_index(drop=False)   # keep original index as 'index' column
complexity_valid  = complexity_df.loc[df_valid["index"]].reset_index(drop=True)
df_valid          = df_valid.reset_index(drop=True)
gold_labels       = df_valid["gold_label"].tolist()

accuracy_table    = {}   # model → variant → accuracy
predictions_store = {}   # model → variant → list of pred labels
split_accuracy    = {}   # model → variant → split → accuracy

matched_mask    = df_valid["split"] == "matched"
mismatched_mask = df_valid["split"] == "mismatched"

for model_name, model_path in models.items():
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nLoading {model_name}...")
    tokenizer  = AutoTokenizer.from_pretrained(model_path)
    model_obj  = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model_obj.eval()

    label_map = LABEL_MAPS.get(model_name)
    assert label_map, f"No label map for {model_name}"

    accuracy_table[model_name]    = {}
    predictions_store[model_name] = {}
    split_accuracy[model_name]    = {}

    for variant, (s1_col, s2_col) in VARIANTS.items():
        preds = get_predictions(
            model_obj, tokenizer,
            df_valid[s1_col].tolist(),
            df_valid[s2_col].tolist(),
            label_map
        )
        predictions_store[model_name][variant] = preds
        accuracy_table[model_name][variant]    = accuracy(gold_labels, preds)

        preds_arr   = np.array(preds)
        gold_arr    = np.array(gold_labels)
        split_accuracy[model_name][variant] = {
            "matched":    accuracy(gold_arr[matched_mask].tolist(),    preds_arr[matched_mask].tolist()),
            "mismatched": accuracy(gold_arr[mismatched_mask].tolist(), preds_arr[mismatched_mask].tolist()),
        }

    del model_obj, tokenizer
    torch.cuda.empty_cache()

accuracy_df = pd.DataFrame(accuracy_table).T
print("\nAccuracy Table:")
print(accuracy_df.to_string())

# ---- Plot 3: Accuracy Table as Figure ----

fig, ax = plt.subplots(figsize=(9, 3))
ax.axis("off")
tbl = ax.table(
    cellText=accuracy_df.round(4).values,
    rowLabels=accuracy_df.index.tolist(),
    colLabels=accuracy_df.columns.tolist(),
    cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.3, 1.8)
ax.set_title("Model Accuracy by Perturbation Variant", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("plot3_accuracy_table.png", bbox_inches="tight", dpi=150)
plt.close()

# ---- Plot 4: Accuracy Grouped Bar (models grouped, variants colored) ----

fig, ax = plt.subplots(figsize=(14, 6))
x         = range(len(accuracy_table))
n_variants = len(VARIANTS)
width      = 0.8 / n_variants
variant_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for i, variant in enumerate(VARIANTS):
    offsets = [xi + i * width for xi in x]
    values  = [accuracy_table[m].get(variant, 0) for m in accuracy_table]
    ax.bar(offsets, values, width=width, label=variant, color=variant_colors[i])

ax.set_xticks([xi + (n_variants - 1) * width / 2 for xi in x])
ax.set_xticklabels(list(accuracy_table.keys()), rotation=15, ha="right")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("NLI Model Accuracy Before and After Perturbation")
ax.legend(title="Perturbation", loc="lower right")
plt.tight_layout()
plt.savefig("plot4_accuracy_grouped_bar.png", bbox_inches="tight", dpi=150)
plt.close()

# ---- Plot 5: Matched vs Mismatched Accuracy ----

fig, axes = plt.subplots(1, len(VARIANTS), figsize=(16, 5), sharey=True)
splits      = ["matched", "mismatched"]
split_cols  = ["#5b9bd5", "#ed7d31"]

for ax, (variant, _) in zip(axes, VARIANTS.items()):
    model_names = list(split_accuracy.keys())
    x           = np.arange(len(model_names))
    bar_w       = 0.35
    for k, (split, col) in enumerate(zip(splits, split_cols)):
        vals = [split_accuracy[m][variant][split] for m in model_names]
        ax.bar(x + k * bar_w, vals, bar_w, label=split, color=col)
    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
    ax.set_title(f"Variant: {variant}", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)

plt.suptitle("Matched vs Mismatched Accuracy by Model and Perturbation", fontsize=13)
plt.tight_layout()
plt.savefig("plot5_matched_vs_mismatched.png", bbox_inches="tight", dpi=150)
plt.close()

# ---- Plot 6: Accuracy Drop (base → each variant) ----

fig, ax = plt.subplots(figsize=(13, 5))
pert_variants = [v for v in VARIANTS if v != "base"]
x             = np.arange(len(pert_variants))
bar_w         = 0.8 / len(models)
model_colors  = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

for i, model_name in enumerate(models):
    drops = [
        accuracy_table[model_name]["base"] - accuracy_table[model_name][v]
        for v in pert_variants
    ]
    ax.bar(x + i * bar_w, drops, bar_w, label=model_name, color=model_colors[i])

ax.set_xticks(x + (len(models) - 1) * bar_w / 2)
ax.set_xticklabels(pert_variants)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("Accuracy Drop (base − perturbed)")
ax.set_title("Accuracy Drop per Perturbation Type by Model")
ax.legend(title="Model", loc="upper right")
plt.tight_layout()
plt.savefig("plot6_accuracy_drop.png", bbox_inches="tight", dpi=150)
plt.close()

# ---- Plot 7: Error Analysis — base-correct / perturbed-wrong heatmap ----

gold_arr  = np.array(gold_labels)
error_matrix = {}

for model_name in models:
    base_preds  = np.array(predictions_store[model_name]["base"])
    base_correct = base_preds == gold_arr
    error_matrix[model_name] = {}

    for variant in pert_variants:
        pert_preds   = np.array(predictions_store[model_name][variant])
        pert_wrong   = pert_preds != gold_arr
        flip_count   = np.sum(base_correct & pert_wrong)
        error_matrix[model_name][variant] = flip_count / np.sum(base_correct)

error_heatmap_df = pd.DataFrame(error_matrix).T   # models × variants

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(error_heatmap_df.values, aspect="auto", cmap="Reds", vmin=0, vmax=0.4)
ax.set_xticks(range(len(pert_variants)))
ax.set_xticklabels(pert_variants)
ax.set_yticks(range(len(error_heatmap_df)))
ax.set_yticklabels(error_heatmap_df.index)
plt.colorbar(im, ax=ax, label="Flip Rate (base correct → perturbed wrong)")
for i in range(len(error_heatmap_df)):
    for j in range(len(pert_variants)):
        ax.text(j, i, f"{error_heatmap_df.values[i,j]:.3f}", ha="center", va="center", fontsize=9)
ax.set_title("Error Flip Rate: Correct on Base, Wrong on Perturbed")
plt.tight_layout()
plt.savefig("plot7_error_heatmap.png", bbox_inches="tight", dpi=150)
plt.close()

# ---- Plot 8: Matched vs Mismatched Flip Rate ----

flip_split = {}  # model → variant → split → flip_rate

for model_name in models:
    base_preds   = np.array(predictions_store[model_name]["base"])
    base_correct = base_preds == gold_arr
    flip_split[model_name] = {}

    for variant in pert_variants:
        pert_preds = np.array(predictions_store[model_name][variant])
        pert_wrong = pert_preds != gold_arr
        flip       = base_correct & pert_wrong
        flip_split[model_name][variant] = {
            "matched":    np.sum(flip & matched_mask.values) / max(np.sum(base_correct & matched_mask.values), 1),
            "mismatched": np.sum(flip & mismatched_mask.values) / max(np.sum(base_correct & mismatched_mask.values), 1),
        }

fig, axes = plt.subplots(1, len(pert_variants), figsize=(15, 5), sharey=True)
for ax, variant in zip(axes, pert_variants):
    model_names = list(flip_split.keys())
    x           = np.arange(len(model_names))
    bar_w       = 0.35
    for k, (split, col) in enumerate(zip(splits, split_cols)):
        vals = [flip_split[m][variant][split] for m in model_names]
        ax.bar(x + k * bar_w, vals, bar_w, label=split, color=col)
    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
    ax.set_title(f"Variant: {variant}", fontsize=11)
    ax.set_ylabel("Flip Rate")
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8)

plt.suptitle("Error Flip Rate: Matched vs Mismatched by Model and Perturbation", fontsize=13)
plt.tight_layout()
plt.savefig("plot8_flip_matched_vs_mismatched.png", bbox_inches="tight", dpi=150)
plt.close()

# ---- Plot 9: Scatter — Complexity Metric vs Model Error Rate (binned) ----

# For each sentence in df_valid, whether each model/variant prediction was correct
# Use sentence1 complexity metrics only, averaged across pair

scatter_metrics = {
    "clause_density": "Clause Density",
    "dep_depth":      "Dependency Tree Depth",
    "subj_verb_dist": "Subject-Verb Distance",
}

for metric_key, metric_label in scatter_metrics.items():
    fig, axes = plt.subplots(1, len(pert_variants), figsize=(16, 5), sharey=True)

    for ax, variant in zip(axes, pert_variants):
        col_name = f"sentence1_{'' if variant == 'base' else variant + '_'}{metric_key}".replace("sentence1__", "sentence1_")
        # Handle base case
        if variant == "base":
            col_name = f"sentence1_{metric_key}"
        else:
            col_name = f"sentence1_{variant}_{metric_key}"

        x_vals = complexity_valid[col_name].dropna()
        valid_idx = x_vals.index

        bins = pd.qcut(x_vals, q=8, duplicates="drop")

        for model_name, color in zip(models, model_colors):
            preds    = np.array(predictions_store[model_name][variant])
            correct  = (preds == gold_arr).astype(float)
            correct_series = pd.Series(correct, index=range(len(correct))).iloc[valid_idx]

            bin_acc = correct_series.groupby(bins).mean()
            bin_mid = [interval.mid for interval in bin_acc.index]
            ax.plot(bin_mid, bin_acc.values, marker="o", label=model_name, color=color)

        ax.set_title(f"Variant: {variant}", fontsize=10)
        ax.set_xlabel(metric_label)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    plt.suptitle(f"Accuracy vs {metric_label} (binned, sentence1)", fontsize=13)
    plt.tight_layout()
    fname = f"plot9_scatter_{metric_key}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {fname}")

# ---- Plot 10: Error Examples Table ----

print("\n=== Error Analysis Examples (base correct → perturbed wrong) ===")
for model_name in models:
    base_preds   = np.array(predictions_store[model_name]["base"])
    base_correct = base_preds == gold_arr

    for variant in pert_variants:
        pert_preds  = np.array(predictions_store[model_name][variant])
        pert_wrong  = pert_preds != gold_arr
        flip_idx    = np.where(base_correct & pert_wrong)[0][:3]

        examples = []
        for idx in flip_idx:
            s1_base = df_valid.iloc[idx]["sentence1"]
            s2_base = df_valid.iloc[idx]["sentence2"]
            s1_pert = df_valid.iloc[idx][f"sentence1_{variant}"]
            s2_pert = df_valid.iloc[idx][f"sentence2_{variant}"]
            gold    = gold_labels[idx]
            split   = df_valid.iloc[idx]["split"]
            examples.append({
                "model": model_name, "variant": variant, "split": split,
                "gold": gold,
                "base_pred": base_preds[idx], "pert_pred": pert_preds[idx],
                "s1_base": s1_base, "s1_pert": s1_pert,
                "s2_base": s2_base, "s2_pert": s2_pert,
            })
            print(f"\n[{model_name} | {variant} | {split}]")
            print(f"  Gold: {gold} | Base pred: {base_preds[idx]} | Pert pred: {pert_preds[idx]}")
            print(f"  S1 base: {s1_base}")
            print(f"  S1 pert: {s1_pert}")

print("\nAll plots saved.")