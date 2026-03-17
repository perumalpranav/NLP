# Load MultiNLI Dataset
"""
by sampling uniformly at random 
Additionally, you must keep track of the source split (matched vs. mismatched) for each pair throughout your evaluation.
"""

# Implement 3 Syntactic Complexity Metrics
"""
To get you started, here are two examples of structural metrics you can use:
• CFG Tree Depth: The maximum depth of the context-free grammar (CFG) constituency parse tree for a given sentence. Deeper trees generally indicate more nested
and complex sentence structures.
• Subject-Verb Distance: The number of words (or edges in a dependency parse) separating the main verb (V) of a clause and its subject noun (NP). For example, in the
sentence “The person in the observatory looked at the moons of Jupiter,” the main verb
is “looked” and its subject noun is “person,” and we can define the subject-verb distance
as the distance between “looked” and “person” in the parse tree of the sentence. Longer
dependencies often make sentences harder to process.
Your Task
Use a parsing library of your choice (e.g. SpaCy, Stanza), and justify them
"""

from nltk import Tree
import spacy

nlp = spacy.load("en_core_web_sm")

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



import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "bert_encoder": "textattack/bert-base-uncased-SST-2",
    "roberta_encoder": "roberta-base",
    "gpt2_decoder": "gpt2",
    "mlp_embedding": "sentence-transformers/all-MiniLM-L6-v2"
}

#Create Data Flow for Pertrubations
"""
Use metrics to show increase in complexity
 injecting new syntactic structures. Here are a few ways to do
this:
• Adding Relative Clauses: Identify adjectival modifiers attached to nouns, extract the
adjective, and automatically rewrite the phrase as a relative clause.
– Original: “The wealthy investor...”
– Perturbed: “The investor, who is wealthy, ...”
Another approach is to add new relative clauses that contains irrelevant information, such
as:
– Original: “The wealthy investor...”
– Perturbed: “The wealthy investor, who is related to Sam’s older sister, ...”
Where “Sam” is not mentioned elsewhere in the premise or hypothesis.
• Entity-Linked Appositive Insertion: Inject complexity by using a Named Entity
Recognition (NER) tagger combined with an external knowledge base (e.g., the Wikidata
API).
Original: “Purdue University announced a new contract with DoD.”
– Perturbed: “Purdue University, a public research university in West Lafayette, announced a new contract with DoD.”
"""

# flat NP pattern: (NP (DT x) (JJ adj) (NN noun))
_FLAT_NP = re.compile(r"\(NP \(\w+ \w+\) \(JJ (\w+)\) \((NNS?) (\w+)\)\)")
 

def perturb_adj_to_relative(record: dict, field: str = "sentence1") -> str:
    sentence = record[field]
    parse    = record[field + "_parse"]
 
    # Only search the portion of the parse before the first PP subtree
    pp_start = parse.find("(PP ")
    search_zone = parse[:pp_start] if pp_start != -1 else parse
 
    match = _FLAT_NP.search(search_zone)
    if not match:
        return sentence
 
    adj, tag, noun = match.group(1), match.group(2), match.group(3)
    pronoun = "which" if tag == "NNS" else "who"
    return re.sub(rf"\b{adj}\s+{noun}\b", f"{noun}, {pronoun} is {adj},", sentence, count=1)
 

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
 



# Run 4 pretrained models with different architectures on the two different datasets

#Encoder Transformer BERT-tiny-MNLI HuggingFace
#Decoder Transformer gpt2-finetuned-mnli HuggingFace



def evaluate_transformer(model, tokenizer, dataset):

    correct = 0

    for sample in dataset:
        inputs = tokenizer(
            sample["premise"],
            sample["hypothesis"],
            return_tensors="pt",
            truncation=True
        ).to(device)

        outputs = model(**inputs)

        pred = torch.argmax(outputs.logits).item()

        if pred == sample["label"]:
            correct += 1

    return correct / len(dataset)

from sklearn.linear_model import LogisticRegression
import numpy as np

def evaluate_mlp(model_name, dataset):

    embedder = SentenceTransformer(model_name)

    X = []
    y = []

    for sample in dataset:
        text = sample["premise"] + " " + sample["hypothesis"]
        emb = embedder.encode(text)

        X.append(emb)
        y.append(sample["label"])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    preds = clf.predict(X)

    return (preds == np.array(y)).mean()


results = {}

for name, model_name in models.items():

    if name == "mlp_embedding":

        accA = evaluate_mlp(model_name, dataset_A)
        accB = evaluate_mlp(model_name, dataset_B)

    else:

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3
        ).to(device)

        accA = evaluate_transformer(model, tokenizer, dataset_A)
        accB = evaluate_transformer(model, tokenizer, dataset_B)

    results[name] = (accA, accB)

for model, scores in results.items():

    print(model)
    print("Dataset A:", scores[0])
    print("Dataset B:", scores[1])
    print()






"""
Write a 3–4 page report detailing your methodology and findings. Your report must include:
1. A high-level introduction summarizing your approach and findings.
2. Definitions and justifications for your chosen complexity metrics. You must provide at
least one concrete example for each metric you implemented.
3. A description of your perturbation methodology. This should include your approach, a
comparison of complexity metrics before and after perturbation, and examples of original
versus perturbed text.
4. A comparative analysis of model performance. Discuss whether certain architectures
handled the added complexity better than others. You must also conduct an error analysis.
Error analysis goes beyond just reporting a drop in accuracy; it requires
you to isolate specific examples where a model correctly classified the original
text but failed on the perturbed version. Examine the linguistic differences
in these sentence pairs and hypothesize why the added syntactic complexity
broke the model’s understanding (e.g., did an added relative clause create a
long-distance dependency that the model could not resolve?). Furthermore,
leverage your tracking of the data splits: you should compare these failures
across the matched (in-domain) and mismatched (out-of-domain) sets to discuss
whether models are more vulnerable to syntactic perturbations when dealing
with out-of-domain text.
5. Scatter plots showing the relationship between your complexity metrics (X-axis) and the
model error rate or accuracy (Y-axis).
"""


#Reprodicable
"""
For reference, see: PyTorch Reproducibility. For
example, if your code relies on a random number generator, make sure to set the random
seed to a fixed value at the beginning of your program, so that all subsequent executions
of your code will produce the same output. The final output of your code should save
all results as two CSV files with the following columns (the numbers must exactly match
those shown in your report):
– perf.csv: model, perturbation method, performance
– complex.csv: perturbation method, metric type, value
"""

