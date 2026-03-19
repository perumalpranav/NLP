import re
import torch
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

def perturb_adj_to_relative(record: dict, field: str = "sentence1") -> str:
    sentence = record[field]
    parse = record[field + "_parse"]

    tree = Tree.fromstring(parse)

    # Find first NP of form: (NP (DT ...) (JJ ...) (NN/NNS ...))
    target_np = None
    for subtree in tree.subtrees(lambda t: t.label() == "NP"):
        if len(subtree) == 3:
            dt, jj, nn = subtree

            if (
                dt.label() == "DT"
                and jj.label() == "JJ"
                and nn.label() in ["NN", "NNS"]
            ):
                target_np = subtree
                break

    if target_np is None:
        return sentence

    # Extract words
    adj = target_np[1].leaves()[0]
    noun = target_np[2].leaves()[0]
    tag = target_np[2].label()

    pronoun = "which" if tag == "NNS" else "who"

    # Replace in sentence (simple string replacement, not regex)
    phrase = f"{adj} {noun}"
    replacement = f"{noun}, {pronoun} is {adj},"

    if phrase in sentence:
        return sentence.replace(phrase, replacement, 1)

    return sentence

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

def perturb_pp_fronting(record: dict, field: str = "sentence1") -> str:
    sentence = record[field].rstrip(". ")
    parse    = record[field + "_parse"]
 
    pp_block = re.search(r"\(PP\s+(\(.*?\)\s*)+\)", parse)
    if not pp_block:
        return record[field]
 
    pp_words = re.findall(r"\(\w+\s+(\w+)\)", pp_block.group(0))
    pp_text  = " ".join(pp_words)
    if not pp_text or pp_text not in sentence:
        return record[field]
 
    remainder = sentence.replace(pp_text, "").strip().rstrip(",")

    if not remainder:
        return record[field]

    return f"{pp_text.capitalize()}, {remainder[0].lower() + remainder[1:]}."
 
# ---- Dataset Changes ----

matched_df = pd.read_json("./multinli_1.0_dev_matched.jsonl", lines=True)
mismatched_df = pd.read_json("./multinli_1.0_dev_mismatched.jsonl", lines=True)

matched_df["split"] = "matched"
mismatched_df["split"] = "mismatched"

N = 10000
matched_sample = matched_df.sample(n=N, random_state=SEED)
mismatched_sample = mismatched_df.sample(n=N, random_state=SEED)

combined_df = pd.concat([matched_sample, mismatched_sample], ignore_index=True)

combined_df["sentence1_adj"] = combined_df.apply(
    lambda x: perturb_adj_to_relative(x, "sentence1"),
    axis=1
)

combined_df["sentence2_adj"] = combined_df.apply(
    lambda x: perturb_adj_to_relative(x, "sentence2"),
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

combined_df["sentence1_pp"] = combined_df.apply(
    lambda x: perturb_pp_fronting(x, "sentence1"),
    axis=1
)

combined_df["sentence2_pp"] = combined_df.apply(
    lambda x: perturb_pp_fronting(x, "sentence2"),
    axis=1
)

rows = []

for _, row in tqdm(combined_df.iterrows()):
    new_row = {}

    for col in ["sentence1", "sentence1_adj", "sentence1_cleft", "sentence1_pp",
                "sentence2", "sentence2_adj", "sentence2_cleft", "sentence2_pp"]:

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
variants = ["", "_adj", "_cleft", "_pp"]
variant_labels = ["base", "adj", "cleft", "pp"]
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
plt.show()


device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "distilbert_encoder": "typeform/distilbert-base-uncased-mnli",
    "squeezebert_encoder": "squeezebert/squeezebert-mnli",
    "bart_seq2seq": "facebook/bart-large-mnli",
    "xlmroberta_encoder": "symanto/xlm-roberta-base-snli-mnli-anli-xnli"
}

# ---- Model Performances ----

MODEL_LABEL_MAPS = {
    "distilbert_encoder":  {0: "contradiction", 1: "neutral", 2: "entailment"},
    "squeezebert_encoder": {0: "contradiction", 1: "neutral", 2: "entailment"},
    "bart_seq2seq":        {0: "contradiction", 1: "neutral", 2: "entailment"},
    "xlmroberta_encoder":  {0: "contradiction", 1: "neutral", 2: "entailment"},
}

VARIANTS = {
    "base":  ("sentence1",       "sentence2"),
    "adj":   ("sentence1_adj",   "sentence2_adj"),
    "cleft": ("sentence1_cleft", "sentence2_cleft"),
    "pp":    ("sentence1_pp",    "sentence2_pp"),
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Prediction function ----
def get_predictions(model, tokenizer, s1_list, s2_list, label_map):
    inputs = tokenizer(s1_list, s2_list, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return [label_map[idx] for idx in logits.argmax(dim=-1).tolist()]

# ---- Accuracy ----
def accuracy(gold, pred):
    return sum(g==p for g, p in zip(gold, pred)) / len(gold)

# ---- Run models ----
accuracy_table = {}

for model_name, model_path in models.items():
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    label_map = MODEL_LABEL_MAPS[model_name]
    accuracy_table[model_name] = {}

    # Ensure variant columns exist and differ from base
    for variant, (s1_col, s2_col) in VARIANTS.items():
        df_valid = combined_df[combined_df["gold_label"].isin(["entailment", "neutral", "contradiction"])]
        gold_labels = df_valid["gold_label"].tolist()
        pred_labels = get_predictions(model, tokenizer, df_valid[s1_col].tolist(), df_valid[s2_col].tolist(), label_map)
        accuracy_table[model_name][variant] = accuracy(gold_labels, pred_labels)

    del model
    torch.cuda.empty_cache()

# ---- Convert to DataFrame ----
accuracy_df = pd.DataFrame(accuracy_table).T
print("\nAccuracy Table:")
print(accuracy_df.to_string())

# ---- Plot ----
fig, ax = plt.subplots(figsize=(12,6))
x = range(len(VARIANTS))
n_models = len(accuracy_table)
width = 0.8 / n_models

for i, (model_name, variant_accs) in enumerate(accuracy_table.items()):
    offsets = [xi + i*width for xi in x]
    values = [variant_accs.get(v, 0) for v in VARIANTS]
    ax.bar(offsets, values, width=width, label=model_name)

ax.set_xticks([xi + (n_models-1)*width/2 for xi in x])
ax.set_xticklabels(list(VARIANTS.keys()))
ax.set_ylabel("Accuracy")
ax.set_ylim(0,1)
ax.set_title("NLI Model Accuracy Before and After Perturbation")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()