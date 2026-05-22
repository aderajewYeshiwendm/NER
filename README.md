# Named Entity Recognition (NER): A Window into Semantic Categorization

---

## Overview

This project develops an NER system using **spaCy** that recognizes and classifies named entities (persons, organizations, locations, dates, and more) in text data. Beyond basic recognition, it models the **associations and context** between entities - connecting NLP techniques to theories from cognitive science such as spreading-activation memory networks, referential accessibility, and top-down lexical priming.

---

## Project Structure

```
NamedEntityRecognition_demo.ipynb       ← Main notebook (all sections)
README.md                               ← This file
```

### Notebook Sections

| Section | Topic | Cognitive Science Connection |
|---|---|---|
| 0 | Setup & environment | Experimental protocol standardization |
| 1 | NER mechanics & entity extraction | Rapid semantic segmentation (Barsalou schemas) |
| 2 | Corpus-level frequency analysis | Repetition-based memory consolidation |
| 3 | Performance evaluation (F1, confusion matrix) | Memory monitoring & source attribution |
| 4 | Domain-specific rules (EntityRuler) | Top-down lexical priming |
| 5 | Ambiguity resolution & error taxonomy | Context-driven sense selection |
| 6 | Entity co-occurrence graph | Spreading-activation semantic networks |
| 7 | Discourse entity timeline | Referential accessibility & given-new contract |

---

## Features

- **Entity extraction** with character offsets, sentence context, and position salience ratio
- **Corpus-level analysis** - type distribution bar chart + entity × sentence heatmap
- **Evaluation** - token-level IO classification report and span-level exact-match F1 (CoNLL standard)
- **Custom domain lexicon** - `COGSCI_BRAIN`, `COGSCI_PROCESS`, `COGSCI_DISORDER` labels via spaCy `EntityRuler` with case-insensitive token-pattern rules
- **Error taxonomy** - structured `correct / false_positive / false_negative / type_error` table with colour coding
- **Co-occurrence graph** - weighted NetworkX graph revealing hub concepts and associative structure
- **Discourse timeline** - Gantt-style entity mention map tracking prominence across a document

---

## Requirements

```
Python >= 3.10
spacy
pandas
matplotlib
seaborn
scikit-learn
networkx
```

Install all dependencies:

```bash
pip install spacy pandas matplotlib seaborn scikit-learn networkx
python -m spacy download en_core_web_sm
```

---


## Key Results

- The base `en_core_web_sm` model correctly identifies standard entities (persons, locations, organizations) but misses domain-specific cognitive science terminology.
- Adding `EntityRuler` patterns introduces `COGSCI_BRAIN`, `COGSCI_PROCESS`, and `COGSCI_DISORDER` labels, improving coverage on neuroscience text.
- The entity co-occurrence graph exposes associative relationships between named entities across sentences - directly modelling the "context and associations" goal of the project.
- Span-level exact-match F1 provides a stricter and more realistic benchmark than token-level accuracy alone.

---

## References

- Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. *Psychological Review, 82*(6), 407–428.
- Gernsbacher, M. A. (1990). *Language Comprehension as Structure Building*. Erlbaum.
- Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.
- Ratinov, L., & Roth, D. (2009). Design challenges and misconceptions in named entity recognition. *CoNLL 2009*.

---

## Course Context

**Topic:** Natural Language Processing (NLP)
**Application domain:** Cognitive Science
**Tools:** Python, spaCy, pandas, matplotlib, seaborn, scikit-learn, networkx
