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

# Run 4 pretrained models with different architectures on the two different datasets
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

