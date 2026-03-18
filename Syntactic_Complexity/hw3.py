import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from nltk import Tree
import spacy
import pandas as pd

SEED = 16
nlp = spacy.load("en_core_web_sm")

# ---- Complexity Measures ----

def cfg_tree_depth(parse_string):
    """
    Computes maximum depth of a CFG constituency parse tree
    """
    tree = Tree.fromstring(parse_string)
    return tree.height()

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
    return f"{pp_text.capitalize()}, {remainder[0].lower() + remainder[1:]}."
 
# ---- Dataset Changes ----

matched_df = pd.read_json("./multinli_1.0_dev_matched.jsonl", lines=True)
mismatched_df = pd.read_json("./multinli_1.0_dev_mismatched.jsonl", lines=True)

matched_df["split"] = "matched"
mismatched_df["split"] = "mismatched"

N = 2000
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


device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "bert_encoder": "textattack/bert-base-uncased-SST-2",
    "roberta_encoder": "roberta-base",
    "gpt2_decoder": "gpt2",
    "mlp_embedding": "sentence-transformers/all-MiniLM-L6-v2"
}






# ---- Model Performances ----