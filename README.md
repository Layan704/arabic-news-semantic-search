# ArabicSeek
## An End-to-End Semantic Retrieval and Hybrid Summarization Platform for Arabic News Archives

ArabicSeek is an end-to-end Arabic NLP platform built to make searching and exploring Arabic news more meaningful than traditional keyword matching.

Instead of asking only whether an article contains the same words as a query, ArabicSeek represents both queries and news articles in a shared semantic space. This allows the system to retrieve articles that are conceptually related even when they use different wording.

The platform brings together Arabic text processing, word-boundary correction, semantic embeddings, topic discovery, ambiguity handling, semantic retrieval, extractive summarization, optional LLM refinement, evaluation, and an interactive bilingual Streamlit interface.

**Project Team:** Mohammed Almadhoun · Layan Khaddash · Shatha Al-Ghrair  
**Faculty of Information Technology · Middle East University · Amman, Jordan**

---

## Table of Contents

- [Project Overview](#project-overview)
- [The Problem](#the-problem)
- [What ArabicSeek Does](#what-arabicseek-does)
- [System Architecture](#system-architecture)
- [Dataset and Corpus](#dataset-and-corpus)
- [Arabic Text Processing](#arabic-text-processing)
- [Semantic Embeddings](#semantic-embeddings)
- [Topic Discovery](#topic-discovery)
- [Semantic Search](#semantic-search)
- [Ambiguity Handling](#ambiguity-handling)
- [Hybrid Summarization](#hybrid-summarization)
- [Evaluation](#evaluation)
- [Interactive Interface](#interactive-interface)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Testing and Development](#testing-and-development)
- [Data and Model Artifacts](#data-and-model-artifacts)
- [Technology Stack](#technology-stack)
- [Limitations and Design Decisions](#limitations-and-design-decisions)
- [Future Work](#future-work)
- [Team](#team)

---

# Project Overview

Arabic news archives grow quickly, but finding the right article is not always a simple keyword-search problem.

Arabic has rich morphology, orthographic variation, and a large amount of vocabulary variation. A user may describe an idea using words that never appear literally in the relevant article. A traditional lexical retrieval system such as BM25 is therefore limited by the surface form of the text.

ArabicSeek was designed around a simple idea:

> **Retrieve by meaning, not only by matching words.**

The system uses the multilingual E5 model (`intfloat/multilingual-e5-large`) to map Arabic queries and documents into a shared vector space. Search results are then ranked according to cosine similarity.

The semantic search layer is only one part of the platform. The same representations are reused throughout the system for topic discovery, ambiguity handling, sentence-level summarization, and exploration.

At a high level:

```text
                Arabic News Archive
                        │
                        ▼
             Scraping & Collection
                        │
                        ▼
              Cleaning / Word Fixing
                        │
                        ▼
            Arabic Linguistic Processing
                        │
                        ├───────────────┐
                        ▼               ▼
              Semantic Embeddings   Lemmatization
                        │               │
                        ▼               ▼
                Topic Discovery   Topic Analysis
                        │
                        ▼
                 Semantic Search
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Article Results       Summarization
             │                     │
             └──────────┬──────────┘
                        ▼
                 Streamlit UI
```

---

# The Problem

Traditional keyword retrieval can struggle with Arabic for several reasons.

### 1. Rich morphology

A single Arabic root can appear in many surface forms. A query term may therefore not occur literally in an article even when the article discusses the same concept.

### 2. Orthographic and linguistic variation

The same concept may be written using different forms or conventions, creating additional lexical variation.

### 3. Vocabulary mismatch

A user may formulate a natural-language query using words that are semantically related to the article but do not overlap strongly with its vocabulary.

This creates a gap between:

```text
What the user means
        ≠
The exact words appearing in the article
```

ArabicSeek addresses this gap using dense semantic retrieval.

---

# What ArabicSeek Does

The project combines several components into one pipeline:

| Component | Purpose |
|---|---|
| News scraping | Collect Arabic news articles |
| SimHash deduplication | Remove duplicate or near-duplicate articles |
| Word-boundary correction | Repair glued or incorrectly split Arabic tokens |
| Arabic preprocessing | Normalize and prepare text for downstream tasks |
| E5 embeddings | Represent articles and queries semantically |
| Topic discovery | Discover groups of related news articles |
| Semantic retrieval | Find articles by meaning |
| Word-sense disambiguation | Resolve ambiguous Arabic words using context |
| TextRank + MMR | Produce faithful extractive summaries |
| LLM refinement | Optional fluency and word-boundary refinement |
| Key-point extraction | Surface important bullet-style information |
| Streamlit interface | Provide an interactive bilingual user experience |
| Evaluation | Measure retrieval and summarization quality |

---

# System Architecture

ArabicSeek is organized into functional layers, with intermediate artifacts stored in JSONL, NumPy, and CSV formats so that individual stages can be rerun independently.

## 1. Acquisition and Cleaning

The system starts by collecting Arabic news and preparing the raw text.

This layer includes:

- sitemap/RSS-based collection
- article cleaning
- duplicate removal
- LLM-assisted word-boundary correction
- additional Arabic word-splitting and prefix-fixing utilities

## 2. Linguistic Processing

Each article is transformed into multiple representations.

Different downstream tasks need different versions of the same text, so ArabicSeek does not force one preprocessing representation to serve every model.

## 3. Semantic Representation

Articles are encoded using:

```text
intfloat/multilingual-e5-large
```

The resulting vectors are normalized and used for semantic retrieval and several other components.

## 4. Topic Discovery

The article embeddings are projected into a lower-dimensional space using UMAP and clustered using HDBSCAN.

## 5. Retrieval and Understanding

The semantic index supports query-to-document retrieval, while the ambiguity component uses the same embedding space to rank possible word senses.

## 6. Summarization and Interface

Retrieved articles can be summarized using the default extractive pipeline, optionally refined using an LLM, and presented through the Streamlit interface.

---

# Dataset and Corpus

The project uses Arabic news collected from **Al Jazeera Arabic**.

The collection and deduplication process resulted in:

```text
16,191 scraped articles
        │
        ▼
SimHash deduplication
        │
        ▼
8,635 unique articles
```

The corpus contains multiple content categories. The project presentation reports the following approximate distribution:

| Category | Share |
|---|---:|
| News (Akhbar) | 40.3% |
| Unspecified | 31.2% |
| Sports | 13.8% |
| Economy | 8.2% |
| Technology | 2.5% |
| Other | 3.9% |

The deployed search and topic index uses the **3,480-article News subset**.

---

# Arabic Text Processing

Arabic preprocessing is one of the central parts of ArabicSeek.

Rather than producing a single cleaned text field, the pipeline creates task-specific representations.

## `embedding_view`

Used for semantic embedding and search.

It preserves information that can be useful for semantic representation, including:

- digits
- decimals
- Latin text

Percent expressions are normalized into Arabic textual forms where appropriate.

## `clustering_view`

Used for topic discovery.

Numerical expressions are converted into typed tokens such as:

```text
YEAR_TOKEN
NUM_TOKEN
PCT_TOKEN
MONEY_TOKEN
```

This prevents specific numbers from dominating topic vocabulary and allows clustering to focus more on thematic information.

## `lemmas`

Used for topic analysis and class-based TF-IDF.

The pipeline uses:

- CAMeL Tools
- `MLELemmatizer`
- customized Arabic stopwords
- filtered token representations

The project also keeps structured date information for downstream tasks such as search and ambiguity handling.

---

# Word-Boundary Correction

Arabic text collected from web sources may contain incorrectly joined or separated words.

ArabicSeek therefore includes a dedicated cleaning stage for word-boundary problems.

A Gemini-powered word-boundary fixer is used to repair glued or split Arabic tokens at scale, with additional safety-net processing through Arabic prefix and word-splitting utilities.

This stage is important because errors in word boundaries can propagate into:

```text
Preprocessing
      ↓
Embeddings
      ↓
Search
      ↓
Topic discovery
      ↓
Summarization
```

Improving the text before these stages therefore improves the quality of the downstream representations.

---

# Semantic Embeddings

The core semantic representation model is:

```text
intfloat/multilingual-e5-large
```

ArabicSeek follows a bi-encoder approach in which queries and documents are encoded into the same vector space.

The model provides multilingual representation capabilities across 100+ languages, including Arabic, allowing the project to perform Arabic semantic retrieval without Arabic-specific fine-tuning.

Each article is represented as a:

```text
1,024-dimensional vector
```

For the full unique corpus, the embedding matrix has the shape:

```text
8,635 × 1,024
```

The vectors are L2-normalized before retrieval.

This makes cosine similarity equivalent to a dot product between the normalized query and document vectors.

---

# Topic Discovery

Topic discovery is performed on the **3,480-article News subset**.

The pipeline is:

```text
E5 Article Embeddings
        │
        ▼
UMAP
1024 dimensions → 5 dimensions
        │
        ▼
HDBSCAN
        │
        ▼
Topic Assignments
        │
        ▼
Class-based TF-IDF
        │
        ▼
Topic Labels
```

## UMAP

The project uses:

```text
Dimensions: 1024 → 5
Metric: cosine
```

The dimensionality reduction provides a compact representation suitable for density-based clustering and visualization.

## HDBSCAN

The clustering configuration includes:

```text
min_cluster_size = 15
min_samples      = 10
```

HDBSCAN was selected because it does not require the number of topics to be specified in advance and can explicitly identify articles that do not belong strongly to any cluster.

It also provides soft membership information that can be used by the application.

## Topic Results

The topic discovery stage produced:

- **20 coherent topics**
- **727 noise articles**
- **20.9% noise**
- **0.328 topic NPMI**
- **0.095 silhouette score**

Topic labels are generated using class-based TF-IDF and the top terms associated with each cluster.

---

# Semantic Search

The deployed search engine uses the E5 model to encode the user's Arabic query.

The retrieval flow is:

```text
User Query
    │
    ▼
"query:" Prefix
    │
    ▼
E5 Encoder
    │
    ▼
Normalized Query Vector
    │
    ▼
Cosine Similarity
    │
    ▼
3,480 News Vectors
    │
    ▼
Ranked Results
```

The query uses the E5 `query:` prefix before encoding.

The search engine compares the normalized query vector against the precomputed normalized article vectors using a NumPy dot product.

Each retrieved article can be enriched with:

- similarity score
- article title
- category
- date
- topic label
- topic membership information
- article snippet
- summary

An optional topic filter can also be used to browse within discovered topics.

## Search Performance

Measured on CPU:

```text
Mean query latency       0.19 seconds
Embedding matrix         34 MB
E5 model weights         ~2.1 GB
GPU requirement          None
```

For the current corpus size, exact cosine search is used rather than introducing approximate nearest-neighbor infrastructure. The project identifies FAISS as a natural future backend when scaling to much larger collections.

---

# Ambiguity Handling

Arabic contains many words whose meaning depends strongly on context.

ArabicSeek includes an embedding-based word-sense disambiguation component that reuses the E5 semantic representation.

The basic idea is:

```text
Ambiguous Word
      +
Context
      │
      ▼
Context Representation
      │
      ▼
Compare with Candidate Sense Glosses
      │
      ▼
Cosine Similarity
      │
      ▼
Ranked Candidate Senses
```

For example, the Arabic word:

```text
عين
```

can have different meanings depending on context.

The implementation can rank candidate senses based on the semantic relationship between the context and each sense description.

Example:

```python
from arnlp.ambiguity import EmbeddingDisambiguator

wsd = EmbeddingDisambiguator()

ranked = wsd.disambiguate_word(
    "عين",
    "تدفقت مياه العين من سفح الجبل"
)
```

The important idea is that ambiguity is treated as a contextual semantic problem rather than a simple string-matching problem.

---

# Hybrid Summarization

Summarization was designed around a specific requirement:

> **The default summary should remain faithful to the source article.**

The project evaluated several summarization approaches and selected a faithful extractive strategy as the default because the tested abstractive approaches, including mT5/AraT5 variants, showed hallucination issues on long news articles.

ArabicSeek therefore provides three complementary summarization modes.

---

## Tier 1 — TextRank + MMR

This is the default summarization method.

The article is divided into sentences and each sentence is represented using E5 embeddings.

A sentence similarity graph is constructed, and TextRank/PageRank identifies central sentences.

MMR then selects a diverse subset of those sentences.

```text
Article
   │
   ▼
Sentence Segmentation
   │
   ▼
E5 Sentence Embeddings
   │
   ▼
Similarity Graph
   │
   ▼
TextRank / PageRank
   │
   ▼
MMR Diversity Selection
   │
   ▼
Original Sentences
```

The MMR parameter is:

```text
λ = 0.7
```

Because the final summary is constructed from original article sentences, the default approach does not invent new facts in the way a free-form generative model can.

---

## Tier 2 — Optional LLM Refinement

An optional `gpt-4o-mini` summarizer is available for cases where additional fluency or word-boundary refinement is useful.

It can:

- repair residual word-boundary issues
- produce a fluent summary
- preserve the article's information
- operate only when explicitly enabled
- cache generated results

This tier requires an API key.

The LLM is therefore an optional refinement layer rather than the foundation of the retrieval system.

---

## Tier 3 — Key-Point Bullets

ArabicSeek also provides a lightweight key-point extraction approach.

A pattern-based scorer looks for several Arabic information families, including areas such as:

- speech
- events
- quantities
- other important factual patterns

The system can surface up to four bullet-style highlights.

---

# Evaluation

The project evaluates both the retrieval system and the summarization system.

---

## Retrieval Evaluation

The semantic retrieval experiment used:

- **30 Arabic queries**
- **5 domains**
- top-5 binary relevance judgments
- **2 annotators**
- a **third annotator for adjudication**

The dense retrieval system was compared against a BM25 lexical baseline.

### Overall Result

| Metric | ArabicSeek | BM25 |
|---|---:|---:|
| Mean NDCG@5 | **0.79** | 0.61 |
| Difference | **+0.18** | — |

A paired Wilcoxon signed-rank test reported:

```text
p < 0.01
```

### Query-Level Results

| Query / Domain | ArabicSeek | BM25 | Difference |
|---|---:|---:|---:|
| Artificial Intelligence · Technology | **0.91** | 0.55 | +0.36 |
| Infectious Diseases · Health | **0.74** | 0.48 | +0.26 |
| United Nations · Politics | **0.85** | 0.72 | +0.13 |
| Gaza War · Politics | **0.88** | 0.81 | +0.07 |
| **Mean across 30 queries** | **0.79** | **0.61** | **+0.18** |

The largest observed gains were reported for paraphrastic technology and health queries. Sports queries showed a smaller difference because sports queries often contain relatively stable lexical names.

---

# Summarization Evaluation

The summarization evaluation was performed on **60 articles**.

The deployed TextRank + MMR method was compared against a simpler extractive baseline.

| Metric | Baseline | TextRank + MMR |
|---|---:|---:|
| Number hallucination ↓ | 0.203 | **0.067** |
| Redundancy ↓ | 0.082 | **0.068** |
| Grounding ↑ | 0.552 | **0.560** |
| Distinct-2 ↑ | 0.918 | **0.932** |
| Length | 75.2 words | **58.9 words** |

The most notable change was the reduction in hallucinated numerical information:

```text
0.203 → 0.067
```

while the generated summaries also became shorter:

```text
75.2 → 58.9 words
```

and slightly improved in grounding and lexical diversity according to the reported metrics.

---

# Interactive Interface

ArabicSeek includes a bilingual Streamlit interface designed specifically around Arabic news exploration.

The interface supports RTL Arabic rendering and combines search, topic exploration, and summarization.

## Semantic Search Tab

The search interface presents ranked news cards containing information such as:

- article title
- semantic similarity score
- topic
- category
- date
- snippet
- one-click summary

The interface is intended to make semantic retrieval understandable rather than exposing only raw similarity scores.

## 3D Topic Galaxy

The second major interface component is an interactive Plotly `scatter3d` visualization of the topic space.

The visualization is based on the UMAP representation and allows users to explore:

- discovered topic clusters
- relationships between clusters
- article positions in semantic space
- topic titles through hover interactions

## Additional Interface Features

The presentation also includes:

- full RTL Arabic support
- bilingual interface elements
- editorial-style visual design
- live corpus statistics
- number of discovered topics
- mean search latency
- a real-time breaking-news ticker

---

# Repository Structure

```text
arabic-news-semantic-search/
│
├── app/
│   ├── streamlit_app.py
│   ├── search_engine.py
│   ├── cluster_view.py
│   ├── topic_labels.py
│   ├── theme.py
│   └── README.md
│
├── configs/
│   ├── default.yaml
│   ├── dev.yaml
│   └── prod.yaml
│
├── data/
│   └── clusters/
│       ├── clusters.jsonl
│       ├── cluster_assignments.csv
│       ├── topics_info.csv
│       ├── topic_labels.json
│       └── topic_stats.csv
│
├── models/
│   └── arabert_splitter/
│
├── reports/
│   └── summarization_eval.json
│
├── scripts/
│   ├── apply_mwe.py
│   ├── apply_word_splitter.py
│   ├── run_clustering.py
│   ├── run_embeddings.py
│   └── run_preprocess.py
│
├── src/
│   └── arnlp/
│       ├── cleaning/
│       ├── embeddings/
│       ├── scraping/
│       ├── utils/
│       ├── ambiguity.py
│       ├── clustering.py
│       ├── evaluation.py
│       ├── logging_setup.py
│       ├── preprocessing.py
│       └── summarization.py
│
├── tests/
│   ├── fixtures/
│   ├── integration/
│   └── unit/
│
├── pipeline.png
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Layan704/arabic-news-semantic-search.git
cd arabic-news-semantic-search
```

## 2. Install the project dependencies

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

## 3. Configure environment variables

Create a local `.env` file from the provided example:

```bash
cp .env.example .env
```

Configure the API keys required by the optional LLM-related components.

The default TextRank + MMR summarizer does not depend on an LLM API.

---

# Running the Pipeline

The project is designed so the major stages can be run independently.

## Step 1 — Preprocessing

For a small smoke test:

```bash
python scripts/run_preprocess.py --offset 5000 --limit 50
```

For the full preprocessing stage:

```bash
python scripts/run_preprocess.py \
    --input data/raw/articles_2026_05_fixed.jsonl \
    --output data/processed/preprocessed.jsonl
```

## Step 2 — Generate Embeddings

```bash
python scripts/run_embeddings.py \
    --input data/processed/preprocessed.jsonl
```

This produces the E5 embedding matrix used by the retrieval system.

## Step 3 — Run Topic Discovery

```bash
python scripts/run_clustering.py
```

This generates the topic and clustering artifacts.

## Step 4 — Launch the Search Interface

```bash
streamlit run app/streamlit_app.py
```

---

# Using the Python Components

The main functionality is exposed through the `arnlp` package.

## Word-Sense Disambiguation

```python
from arnlp.ambiguity import EmbeddingDisambiguator

disambiguator = EmbeddingDisambiguator()

ranked = disambiguator.disambiguate_word(
    "عين",
    "تدفقت مياه العين من سفح الجبل"
)

print(ranked[0])
```

## Extractive Summarization

```python
from arnlp.summarization import ExtractiveSummarizer

summarizer = ExtractiveSummarizer()

summary = summarizer.summarize(article_body)

print(summary)
```

---

# Testing and Development

The project includes unit and integration tests.

## Linting

```bash
ruff check .
```

## Formatting

```bash
ruff format .
```

## Type Checking

```bash
mypy src/arnlp
```

## Unit Tests

```bash
pytest tests/unit -q
```

## Full Test Suite

```bash
pytest
```

---

# Data and Model Artifacts

The complete corpus and large model artifacts are not committed to the GitHub repository because of their size.

The local project contains large artifacts under directories such as:

```text
data/raw/
data/processed/
data/model_training/
data/embeddings/
models/
```

The GitHub repository instead contains the source code, pipeline scripts, tests, configurations, lightweight clustering outputs, evaluation reports, and documentation required to understand the project.

This keeps the repository practical to clone while preserving the full working artifacts locally.

---

# Technology Stack

## Natural Language Processing

- Python
- Transformers
- PyTorch
- `intfloat/multilingual-e5-large`
- CAMeL Tools
- Arabic text normalization
- Arabic stopword processing
- Lemmatization

## Information Retrieval

- Dense semantic retrieval
- Bi-encoder architecture
- Cosine similarity
- NumPy vector operations
- BM25 baseline for evaluation

## Topic Modeling

- UMAP
- HDBSCAN
- BERTopic-style analysis
- Class-based TF-IDF
- Plotly 3D visualization

## Summarization

- TextRank
- PageRank
- Maximal Marginal Relevance
- E5 sentence embeddings
- Optional `gpt-4o-mini` refinement

## Application

- Streamlit
- Plotly
- RTL Arabic interface
- Bilingual UI

## Engineering

- pytest
- Ruff
- mypy
- YAML configuration
- JSONL
- NumPy
- CSV-based intermediate artifacts

---

# Limitations and Design Decisions

ArabicSeek was designed with several practical constraints in mind.

### CPU-first retrieval

The deployed semantic search path was designed to run on CPU, avoiding a mandatory GPU dependency.

### Exact search for the current corpus

The current index contains 3,480 searchable News articles, making exact cosine similarity practical. The project does not introduce FAISS prematurely for a corpus of this size.

### Faithfulness over aggressive generation

The default summarizer intentionally uses original sentences instead of relying entirely on abstractive generation.

This was a deliberate choice after evaluating summarization approaches and observing hallucination problems in long Arabic news articles.

### Large artifacts are kept outside Git

Raw data, processed corpora, embeddings, and large model checkpoints are kept locally rather than being committed to GitHub.

### Evaluation scope

The reported retrieval results come from a manually evaluated set of 30 Arabic queries across five domains, while summarization evaluation covers 60 articles. These results therefore describe the evaluated dataset and experimental setup rather than claiming universal performance across all Arabic news.

---

# Future Work

Several extensions were identified during the project:

### Arabic-specific semantic fine-tuning

Fine-tune the bi-encoder using Arabic domain-specific relevance pairs to improve retrieval beyond zero-shot multilingual embeddings.

### Cross-lingual retrieval

Support queries in one language retrieving documents in another, for example:

```text
English Query
      ↓
Arabic News
```

using approaches such as LaBSE.

### Neural topic labeling

Replace or complement class-based TF-IDF labels with neural Arabic topic-label generation.

### Multi-document summaries

Generate a digest from an entire discovered topic or cluster rather than summarizing articles individually.

### Large-scale retrieval

Introduce FAISS or another approximate nearest-neighbor backend when scaling the corpus to hundreds of thousands or millions of articles.

### Larger evaluation sets

Expand query domains, annotation volume, and evaluation coverage to obtain a broader measurement of retrieval quality.

---

# Project Takeaway

ArabicSeek is not just a semantic search demo.

It is a complete Arabic NLP pipeline that connects:

```text
Data Collection
      ↓
Arabic Cleaning
      ↓
Linguistic Processing
      ↓
Semantic Representation
      ↓
Topic Discovery
      ↓
Semantic Retrieval
      ↓
Ambiguity Handling
      ↓
Faithful Summarization
      ↓
Evaluation
      ↓
Interactive Exploration
```

The project demonstrates how multiple NLP techniques can be combined into one practical system for working with Arabic news archives.

The main experimental results show:

```text
8,635 unique articles
20 discovered topics
0.19 s mean CPU query latency

NDCG@5
ArabicSeek : 0.79
BM25       : 0.61

Number hallucination
Baseline   : 0.203
Deployed   : 0.067
```

The result is an Arabic news platform that can retrieve articles by semantic meaning, organize them into discoverable topics, handle contextual ambiguity, and generate concise summaries while keeping the default summarization process grounded in the original source text.

---

# Team

**Mohammed Almadhoun**  
**Layan Khaddash**  
**Shatha Al-Ghrair**

**Faculty of Information Technology**  
**Middle East University**  
**Amman, Jordan**

---

### Academic Project · 2026
