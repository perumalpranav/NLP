import re
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from nltk import Tree
import spacy
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt

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
    """
    Compute distance between subject and root verb
    """
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

# ---- Pertrubation Methods ----

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

    # Find first NP inside a VP (i.e., the object)
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
    parse    = record[field + "_parse"]
 
    np_block   = re.search(r"\(NP\s+(\(.*?\)\s*)+\)", parse)
    if not np_block:
        return record[field]
 
    subj_words = re.findall(r"\(\w+\s+(\w+)\)", np_block.group(0))
    subject    = " ".join(subj_words)
    if not subject or subject not in sentence:
        return record[field]
 
    rest = sentence[sentence.index(subject) + len(subject):].strip()
    return f"It was {subject.lower()} that {rest}."

# ---- Dataset Changes ----

matched_df = pd.read_json("./multinli_1.0_dev_matched.jsonl", lines=True)
mismatched_df = pd.read_json("./multinli_1.0_dev_mismatched.jsonl", lines=True)

matched_df["split"] = "matched"
mismatched_df["split"] = "mismatched"

N = 8000
matched_sample = matched_df.sample(n=N, random_state=SEED)
mismatched_sample = mismatched_df.sample(n=N, random_state=SEED)

combined_df = pd.concat([matched_sample, mismatched_sample], ignore_index=True)

combined_df["sentence1_rel"] = combined_df.apply(
    lambda x: perturb_relative_clause(x, "sentence1"),
    axis=1
)

combined_df["sentence2_rel"] = combined_df.apply(
    lambda x: perturb_relative_clause(x, "sentence2"),
    axis=1
)

combined_df["sentence1_cleft"] = combined_df.apply(
    lambda x: perturb_cleft(x, "sentence1"),
    axis=1
)

combined_df["sentence2_cleft"] = combined_df.apply(
    lambda x: perturb_cleft(x, "sentence2"),
    axis=1
)

combined_df["sentence1_topic"] = combined_df.apply(
    lambda x: perturb_topicalization(x, "sentence1"),
    axis=1
)

combined_df["sentence2_topic"] = combined_df.apply(
    lambda x: perturb_topicalization(x, "sentence2"),
    axis=1
)

rows = []

for _, row in tqdm(combined_df.iterrows()):
    new_row = {}

    for col in ["sentence1", "sentence1_rel", "sentence1_cleft", "sentence1_topic",
                "sentence2", "sentence2_rel", "sentence2_cleft", "sentence2_topic"]:

        sentence = row[col]

        new_row[f"{col}_clause_density"] = clause_density(sentence)
        new_row[f"{col}_dep_depth"] = dependency_tree_depth(sentence)
        new_row[f"{col}_subj_verb_dist"] = subject_verb_distance(sentence)

    rows.append(new_row)

complexity_df = pd.DataFrame(rows)

column_means = complexity_df.mean(numeric_only=True)
print(column_means)


metrics = ["clause_density", "dep_depth", "subj_verb_dist"]
bases = ["sentence1", "sentence2"]
variants = ["", "_rel", "_cleft", "_topic"]
variant_labels = ["base", "rel", "cleft", "topic"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # distinct colors for each variant

fig, axes = plt.subplots(3, 2, figsize=(12, 12))

for i, metric in enumerate(metrics):
    for j, base in enumerate(bases):
        ax = axes[i, j]

        cols = [f"{base}{v}_{metric}" for v in variants]
        values = column_means[cols]

        ax.bar(variant_labels, values, color=colors)
        ax.set_title(f"{base} - {metric}")

plt.tight_layout()
plt.savefig('complexity_plots.png')

device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "deberta_mnli": "cross-encoder/nli-deberta-v3-xsmall",
    "tiny_bert_mnli": "prajjwal1/bert-tiny-mnli",
    "electra_mnli": "howey/electra-small-mnli",
    "nli_miniLM": "cross-encoder/nli-MiniLM2-L6-H768"
}


VARIANTS = {
    "base":  ("sentence1",       "sentence2"),
    "rel":   ("sentence1_rel",   "sentence2_rel"),
    "cleft": ("sentence1_cleft", "sentence2_cleft"),
    "topic":    ("sentence1_topic",    "sentence2_topic"),
}

LABEL_MAPS = {
    "electra_mnli":    {0: "contradiction", 1: "entailment",   2: "neutral"},
    "bart_mnli":       {0: "contradiction", 1: "neutral",       2: "entailment"},
    "mobilebert_mnli": {0: "entailment",    1: "neutral",       2: "contradiction"},
    "nli_miniLM":      {0: "contradiction", 1: "entailment",   2: "neutral"},
    "distilbert_mnli": {0: "entailment",    1: "neutral",       2: "contradiction"},
    "albert_mnli":     {0: "entailment",    1: "neutral",       2: "contradiction"},
    "deberta_mnli":    {0: "contradiction", 1: "entailment",   2: "neutral"},
    "roberta_mnli":    {0: "contradiction", 1: "entailment",   2: "neutral"},
    "squeezebert_mnli":{0: "entailment",    1: "neutral",       2: "contradiction"},
    "tiny_bert_mnli":  {0: "entailment",    1: "neutral",       2: "contradiction"},
}

# ---- Prediction function ----
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

# ---- Accuracy ----
def accuracy(gold, pred):
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)

# ---- Run models ----
accuracy_table = {}

# Filter once outside all loops
df_valid = combined_df[combined_df["gold_label"].isin(["entailment", "neutral", "contradiction"])]
gold_labels = df_valid["gold_label"].tolist()

for model_name, model_path in models.items():
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    label_map = LABEL_MAPS.get(model_name)
    accuracy_table[model_name] = {}

    for variant, (s1_col, s2_col) in VARIANTS.items():
        pred_labels = get_predictions(
            model, tokenizer,
            df_valid[s1_col].tolist(),
            df_valid[s2_col].tolist(),
            label_map
        )
        accuracy_table[model_name][variant] = accuracy(gold_labels, pred_labels)

    del model
    torch.cuda.empty_cache()

# ---- Convert to DataFrame ----
accuracy_df = pd.DataFrame(accuracy_table).T
print("\nAccuracy Table:")
print(accuracy_df.to_string())

# ---- Plot ----
fig, ax = plt.subplots(figsize=(14, 6))
x = range(len(accuracy_table))
n_variants = len(VARIANTS)
width = 0.8 / n_variants
variant_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for i, variant in enumerate(VARIANTS):
    offsets = [xi + i * width for xi in x]
    values = [accuracy_table[model].get(variant, 0) for model in accuracy_table]
    ax.bar(offsets, values, width=width, label=variant, color=variant_colors[i])

ax.set_xticks([xi + (n_variants - 1) * width / 2 for xi in x])
ax.set_xticklabels(list(accuracy_table.keys()), rotation=15, ha="right")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("NLI Model Accuracy Before and After Perturbation")
ax.legend(title="Perturbation", loc="lower right")
plt.tight_layout()
plt.savefig('accuracy_plots.png')