# Named Entity Recognition (NER): A Window into Semantic Categorization

This project uses spaCy to build a Named Entity Recognition system and then asks a second question on top of it: how does what the model does compare to what a human reader does when scanning the same text? Each technical step in the notebook has a short explanation that links it back to a result from cognitive science.

## Table of Contents

1. [Motivation](#motivation)
2. [What This Project Does](#what-this-project-does)
3. [Cognitive Science Framing](#cognitive-science-framing)
4. [Project Structure](#project-structure)
5. [Notebook Walkthrough](#notebook-walkthrough)
6. [Features](#features)
7. [Requirements](#requirements)
8. [Installation](#installation)
9. [How to Run](#how-to-run)
10. [Example Output](#example-output)
11. [Key Results](#key-results)
12. [Extending the Project](#extending-the-project)
13. [Troubleshooting](#troubleshooting)
14. [References](#references)
15. [Course Context](#course-context)

## Motivation

When you read a sentence like *"Alan Turing worked at Bletchley Park during WWII"*, you don't store it as a flat string of words. You break it into chunks (a person, a place, a time period) and drop each chunk into a category you already know about. That chunked version is what makes the sentence easy to remember and to connect with other things you've read.

NER does the same thing computationally. So this project treats NER as a small working model of categorization, and asks the following:

- How does a probabilistic model split continuous text into a fixed set of categories?
- What do the frequency and co-occurrence patterns of entities look like across a corpus?
- Where does the model agree with human judgments, and where does it diverge?
- What changes when we feed in expert vocabulary the model wasn't trained on?

## What This Project Does

The repository has one Jupyter notebook ([NamedEntityRecognition_demo.ipynb](NamedEntityRecognition_demo.ipynb)) that walks through eight sections:

1. Setup: environment, imports, model loading.
2. NER mechanics: pulling entities out of a sentence with offsets and context.
3. Corpus-level frequency analysis: bar chart of entity types and an entity-by-sentence heatmap.
4. Performance evaluation: token-level classification report and span-level CoNLL-style F1, plus a confusion matrix.
5. Custom domain rules: adding three cognitive-science labels using spaCy's `EntityRuler`.
6. Error and ambiguity analysis: a small taxonomy of false positives, false negatives, and type errors.
7. Co-occurrence graph: a weighted NetworkX graph showing which entities cluster together.
8. Discourse timeline: a Gantt-style chart showing which entities recur across the document.

## Cognitive Science Framing

The mapping between the code and the cognitive-science claims is summarized below, and then explained in more depth in the sub-sections that follow.

| Computational Step | Cognitive Phenomenon | Key Reference |
|---|---|---|
| Bottom-up entity tagging | Rapid semantic segmentation into schemas | Barsalou (1992) |
| Corpus frequency counts | Memory consolidation through repetition | McClelland et al. (1995) |
| Precision / recall / F1 | Meta-cognitive monitoring | Koriat (2000) |
| Custom `EntityRuler` patterns | Top-down lexical priming by expertise | Neely (1977) |
| Co-occurrence graph | Spreading activation in semantic networks | Collins & Loftus (1975) |
| Discourse timeline | Referential accessibility, given-new contract | Gernsbacher (1990) |

### 1. Entity tagging and rapid schematic perception

When you read *"Marie Curie won Nobel Prizes in Paris"*, you don't consciously decide that *Marie Curie* is a person and *Paris* is a city. The assignment happens in a few hundred milliseconds, below the threshold of awareness. The standard explanation (Barsalou, 1992; Rosch, 1978) is that a noun phrase activates the closest matching schema in long-term memory, and the rest of comprehension proceeds from there.

The `extract_entities(text)` helper does the same operation. A span of characters gets mapped to a label from a small fixed inventory (`PERSON`, `GPE`, `EVENT`, and so on). The inventory is coarse on purpose, which matches the fact that human categorization also compresses an open-ended world into a handful of types.

### 2. Frequency analysis and consolidation through repetition

The bar chart and heatmap in Section 2 are not just descriptive stats. They reflect the well-known result that repeated exposure builds durable memory. Complementary Learning Systems theory (McClelland, McNaughton & O'Reilly, 1995) describes the hippocampus storing individual episodes quickly and the neocortex extracting statistical regularities over many exposures. High-frequency entities in the heatmap are the items most likely to make that slow transition into stable long-term knowledge.

The frequency distribution is also Zipf-like: a few entities recur a lot, most appear once. This is the same shape that shows up in word-learning studies of children and second-language learners.

### 3. Precision and recall as meta-cognitive monitoring

A model with 90% recall and 60% precision is being over-eager: it labels too many things as entities. Human memory has the same trade-off. Under loose criteria we report many memories but make source-monitoring errors. Under tight criteria we miss real ones.

Koriat's (2000) work on the feeling of knowing treats this calibration as a meta-cognitive skill, the ability to evaluate your own outputs. The confusion matrix in Section 3 is the computational version of that view. We are looking at where the model thinks it is right and is actually wrong, which is the same thing a person does when they realise they confidently misremembered something.

### 4. EntityRuler as expert priming

A baseline spaCy model has never been told that *hippocampus* is a brain region. A neuroscientist has. Section 4 closes that gap by inserting an `EntityRuler` ahead of the statistical NER component, so the expert vocabulary gets the first pass at the text.

This is the computational version of top-down semantic priming (Neely, 1977). When a reader holds a strong prior expectation (a neuroscientist reading a neuroscience paper), that expectation reaches down into perception and changes how words are recognized. The string *Broca* is parsed differently by a neuroscientist than by a tourist, and the pipeline becomes "expert" the moment the ruler is added.

### 5. Error analysis and context-driven sense selection

The word *Apple* could be a fruit or a company. *Washington* could be a person, a city, or a state. Humans resolve these in real time using context and rarely even notice the ambiguity. Constraint-based parsing models describe this as parallel activation of multiple senses, with context quickly suppressing the wrong ones.

The error taxonomy in Section 5 picks out the cases where the model's context resolution fails in ways a human reader usually wouldn't. Looking at those failure modes is useful both for fixing the pipeline and for thinking about what kinds of context human readers actually use.

### 6. Co-occurrence graph and spreading activation

Collins and Loftus (1975) proposed that semantic memory is a graph. Concepts are nodes, associations are weighted edges, and activating one node sends activation along the edges to its neighbours. This is why hearing *doctor* speeds up recognition of *nurse*.

Section 6 builds that graph from text. Every pair of entities that show up in the same sentence becomes an edge, and repeated co-mentions raise the weight. The high-degree nodes that emerge are the same kind of hub concepts that dominate semantic-priming studies.

### 7. Discourse timeline and referential accessibility

Gernsbacher's (1990) Structure-Building Framework describes comprehension as the incremental construction of a mental representation. Some entities stay highly accessible because they are central to the discourse. Others fade because they are mentioned once and dropped. The given-new contract (Clark & Haviland, 1977) adds that speakers attach new information to entities the listener already has active.

The Gantt-style timeline in Section 7 visualizes that accessibility gradient. Entities with long horizontal lifelines are the anchors a reader keeps active. Single-mention entities are the "new" content being attached to those anchors.

### Why this framing matters

Treating NER as a window into categorization isn't decorative. It changes the questions we ask about the model. A pure engineering view only asks about F1. A cognitive view also asks:

- Are the model's confusions the same ones humans make (for example, mixing up person names and city names that share a form)?
- Does the model show the same frequency-driven advantages humans show on common entities?
- Does expert priming through the `EntityRuler` shift performance the same way expertise shifts human readers?
- Does the model's associative structure (the co-occurrence graph) line up with documented human semantic networks?

## Project Structure

```
NER/
├── NamedEntityRecognition_demo.ipynb   Main notebook, all eight sections
├── README.md                           This file
└── requirements.txt                    Python dependencies
```

The notebook is monolithic on purpose. Each section reuses variables defined earlier, so it should be run top-to-bottom in a fresh kernel.

## Notebook Walkthrough

### Section 0. Setup
Installs and imports `spacy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `networkx`, then loads the compact English pipeline `en_core_web_sm`. That pipeline provides tokenization, POS tagging, dependency parsing, and a pretrained NER component covering 18 entity types (`PERSON`, `ORG`, `GPE`, `DATE`, `EVENT`, and so on).

### Section 1. NER mechanics
Defines `extract_entities(text)`, a helper that wraps `nlp(text).ents` and returns structured records with the entity text, label, description, character offsets, sentence index, and a position ratio. The output is shown both as a pandas table and as an inline `displacy` visualization.

### Section 2. Corpus-level analysis
A 15-sentence corpus of historical and scientific facts is processed end-to-end. Two plots sit side-by-side: a bar chart of entity-type counts and a heatmap of entity-by-sentence occurrence.

### Section 3. Performance evaluation
Five sentences are hand-annotated with gold `(span, label)` pairs. Two metrics are computed:

- Token-level IO classification report, using `sklearn.metrics.classification_report`.
- Span-level exact-match F1 following the CoNLL standard. A prediction only counts if both the span boundaries and the label match.

A confusion-matrix heatmap shows systematic confusions (for example `ORG` vs `GPE`).

### Section 4. Custom domain rules
Out of the box, `en_core_web_sm` doesn't know cognitive-science vocabulary. We add an `EntityRuler` before the statistical NER component with case-insensitive token patterns for three new labels:

- `COGSCI_BRAIN`: hippocampus, prefrontal cortex, amygdala, and so on.
- `COGSCI_PROCESS`: working memory, attention, encoding.
- `COGSCI_DISORDER`: Alzheimer's, aphasia, dyslexia.

### Section 5. Error and ambiguity analysis
Every predicted entity is labeled as `correct`, `false_positive`, `false_negative`, or `type_error`, producing a colour-coded table. The section talks through lexical ambiguity cases like *Apple* (fruit vs company) and *Washington* (person, state, or city) and how statistical context resolves them.

### Section 6. Co-occurrence graph
For every pair of entities that show up in the same sentence, an edge is added (or its weight incremented) in a `networkx.Graph`. The graph is drawn with a spring layout, where highly-connected nodes sit at the centre and edge thickness encodes how often two entities co-occur.

### Section 7. Discourse timeline
A Gantt-style scatter plot maps every entity mention to its sentence index. Entities that recur across many sentences are thematically central. Entities that appear once are peripheral.

## Features

- Structured entity extraction with character offsets, sentence context, and a position-salience ratio (`start_char / len(text)`) as a lightweight prominence proxy.
- Side-by-side bar chart and heatmap in a single matplotlib figure.
- Dual-granularity evaluation: token-level IO report and span-level exact-match F1.
- Custom domain lexicon with three cogsci labels added via `EntityRuler`. The ruler is placed before the statistical NER so rules win on conflicts.
- Error taxonomy that buckets every model decision into `correct / false_positive / false_negative / type_error`, with row-level colour coding.
- Weighted co-occurrence graph built with `itertools.combinations` over each sentence's entity set, drawn with `nx.spring_layout` and edge widths proportional to co-occurrence counts.
- Gantt-style discourse timeline with sentence index on the x-axis, entity on the y-axis, and marker size proportional to mention count.

## Requirements

- Python 3.10 or newer (the notebook uses PEP-604 `list[dict]` syntax).
- A working Jupyter environment (JupyterLab, classic Notebook, or VS Code's notebook UI).
- About 500 MB of disk space for the spaCy model and its dependencies.

Python packages (also pinned in [requirements.txt](requirements.txt)):

```
spacy
pandas
matplotlib
seaborn
scikit-learn
networkx
```

## Installation

Clone the repository and install the dependencies in a virtual environment:

```bash
git clone <repo-url>
cd NER

python -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate             # Windows PowerShell

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If you'd rather not use `requirements.txt`:

```bash
pip install spacy pandas matplotlib seaborn scikit-learn networkx
python -m spacy download en_core_web_sm
```

## How to Run

1. Start Jupyter from the project root:

   ```bash
   jupyter lab        # or: jupyter notebook
   ```

2. Open [NamedEntityRecognition_demo.ipynb](NamedEntityRecognition_demo.ipynb).

3. Run all cells top-to-bottom. `Kernel > Restart & Run All` is the safest path, since later sections depend on variables defined earlier.

The first cell installs packages via `%pip install`. If you already installed them with `pip install -r requirements.txt`, that cell is a no-op.

## Example Output

Running Section 1 on the demo sentence

> *"Alan Turing worked at Bletchley Park during WWII and collaborated with MIT researchers in Cambridge, Massachusetts in 1936."*

produces the following entities:

| text | label | description |
|---|---|---|
| Alan Turing | PERSON | People, including fictional |
| Bletchley Park | FAC | Buildings, airports, highways, bridges |
| WWII | EVENT | Named hurricanes, battles, wars |
| MIT | ORG | Companies, agencies, institutions |
| Cambridge | GPE | Countries, cities, states |
| Massachusetts | GPE | Countries, cities, states |
| 1936 | DATE | Absolute or relative dates or periods |

The notebook also renders these inline using `spacy.displacy` with colour-coded highlights.

## Key Results

- The baseline `en_core_web_sm` model reliably handles canonical entity types (`PERSON`, `GPE`, `ORG`, `DATE`) in well-formed historical and scientific prose.
- The same model misses cognitive-science vocabulary like *hippocampus*, *working memory*, or *Alzheimer's disease*. It has no schema for them.
- Adding an `EntityRuler` with around 30 token patterns gives near-perfect coverage of the three new cogsci labels without retraining the underlying model.
- Span-level exact-match F1 comes out consistently lower than token-level F1, which confirms that token accuracy on its own overstates real-world performance.
- The co-occurrence graph turns up hub entities (for example *Einstein*, *NASA*) that anchor multiple narrative threads.
- Most entities are mentioned once, and a small group recur many times. The shape of that distribution matches what psycholinguists observe in human discourse.

## Extending the Project

A few ideas for follow-up work:

- Swap in `en_core_web_trf` (the Transformer pipeline) and compare span-F1 against `en_core_web_sm`.
- Add coreference resolution with `coreferee` or `fastcoref` to merge *Newton* / *Isaac Newton* / *he* into a single discourse entity.
- Replace the toy corpus with a real domain dataset (PubMed abstracts, news articles, historical biographies) and rerun the metrics.
- Train a custom NER head on annotated cogsci text instead of relying on `EntityRuler` patterns.
- Build a small Streamlit or Gradio dashboard where users paste text and see entities, graph, and timeline live.
- Compare the model's confusions against documented human categorization errors from the psycholinguistics literature.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `OSError: [E050] Can't find model 'en_core_web_sm'` | Model not downloaded into the active environment | `python -m spacy download en_core_web_sm` |
| `displacy` shows raw HTML instead of visuals | Running outside a Jupyter context | Open the file as a notebook, not as a plain `.py` |
| Plots look tiny or overlap | Inline backend defaults | Add `%matplotlib inline` or run Restart & Run All |
| Kernel hangs on first run | Background `pip install` cell still running | Wait for the install cell to finish before re-running |
| `ModuleNotFoundError: networkx` | Dependencies installed in a different interpreter | Confirm the Jupyter kernel matches your `.venv` |

## References

- Barsalou, L. W. (1992). *Cognitive Psychology: An Overview for Cognitive Scientists*. Lawrence Erlbaum.
- Clark, H. H., & Haviland, S. E. (1977). Comprehension and the given-new contract.
- Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. *Psychological Review, 82*(6), 407-428.
- Gernsbacher, M. A. (1990). *Language Comprehension as Structure Building*. Erlbaum.
- Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.
- Koriat, A. (2000). The feeling of knowing: Some metatheoretical implications for consciousness and control. *Consciousness and Cognition, 9*(2), 149-171.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review, 102*(3), 419-457.
- Neely, J. H. (1977). Semantic priming and retrieval from lexical memory. *Journal of Experimental Psychology: General, 106*(3), 226-254.
- Ratinov, L., & Roth, D. (2009). Design challenges and misconceptions in named entity recognition. *CoNLL 2009*.

## Course Context

- Course topic: Natural Language Processing for Cognitive Science
- Application domain: Computational modelling of semantic categorization and memory
- Toolchain: Python, spaCy, pandas, matplotlib, seaborn, scikit-learn, networkx, Jupyter
