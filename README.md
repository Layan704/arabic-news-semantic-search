# Arabic News NLP — Ambiguity-Aware Semantic Retrieval

An end-to-end Arabic NLP pipeline over **Al Jazeera Arabic** news that
supports semantic search, BERTopic clustering, summarization, and an
explicit focus on **ambiguity handling**.

```
Scrape  →  LLM word-fix  →  Linguistic preprocessing  →  Embeddings  →  ┐
                                                                        ├──▶  Clustering / Search / Summarization
                                                            Ambiguity ─┘
```

## Quick start

```bash
# 1. install
pip install -r requirements.txt
pip install -r requirements-dev.txt          # optional, for tests / lint

# 2. configure secrets
cp .env.example .env                          # add OPENROUTER_API_KEY (+ OPENAI_API_KEY for LLM summaries)

# 3. run the Stage-2 preprocessor on a 50-article smoke sample
python scripts/run_preprocess.py --offset 5000 --limit 50

# 4. when you're confident, run the full corpus
python scripts/run_preprocess.py \
    --input  data/raw/articles_2026_05_fixed.jsonl \
    --output data/processed/preprocessed.jsonl

# 5. embed → cluster → launch the search UI
python scripts/run_embeddings.py --input data/processed/preprocessed.jsonl
python scripts/run_clustering.py
streamlit run app/streamlit_app.py
```

## Repository layout

```
.
├── src/arnlp/                # importable Python package (the code surface)
│   ├── scraping/             # Stage 0 — sitemap + RSS collectors (package)
│   ├── cleaning/             # Stages 1 & 3 — LLM fix, ال-prefix, splitter (package)
│   ├── preprocessing.py      # Stage 2 — Arabic linguistic preprocessing (module)
│   ├── embeddings/           # Stage 4 — E5 encoder (package)
│   ├── clustering.py         # Stage 5 — BERTopic over precomputed vectors (module)
│   ├── ambiguity.py          # Stage 6 — E5 word-sense disambiguation
│   ├── summarization.py      # Stage 7 — extractive (TextRank+MMR) + opt-in LLM
│   ├── evaluation.py         # Stage 8 — summarization metrics (ROUGE, grounding, …)
│   ├── utils/                # io helpers (read_jsonl / write_jsonl)
│   └── logging_setup.py      # logging configuration
├── app/                      # Stage 9 — Streamlit semantic-search UI
│   ├── streamlit_app.py      #   the RTL Arabic search app
│   └── search_engine.py      #   SemanticSearchEngine (E5 + cosine similarity)
├── scripts/                  # thin CLIs over src/arnlp
├── configs/                  # YAML config scaffolding (not yet wired to code)
├── tests/                    # pytest (unit + integration)
├── reports/                  # evaluation outputs (e.g. summarization_eval.json)
├── data/                     # gitignored (raw, processed, embeddings, clusters)
└── models/                   # gitignored (trained checkpoints)
```

> **Layout note:** stages are *packages* where they span multiple files
> (`scraping/`, `cleaning/`, `embeddings/`) and *single modules* where they
> don't (`preprocessing.py`, `clustering.py`, `ambiguity.py`,
> `summarization.py`, `evaluation.py`).

## Pipeline stages

| Stage | Module / entry point | Output |
|------:|----------------------|--------|
| 0 | `arnlp.scraping` | `data/raw/articles_*.jsonl` |
| 1 | `arnlp.cleaning.llm_word_fixer` | `data/raw/articles_*_fixed.jsonl` |
| 2 | `arnlp.preprocessing.ArabicTextPipeline` | `data/processed/preprocessed.jsonl` |
| 3 | `arnlp.cleaning.al_prefix_fix` + `word_splitter` | (optional safety nets) |
| 4 | `arnlp.embeddings` (`E5Encoder`) | `data/embeddings/e5_large.npy` |
| 5 | `arnlp.clustering` | `data/clusters/clusters.jsonl` |
| 6 | `arnlp.ambiguity.EmbeddingDisambiguator` | ranked word senses (E5 cosine) |
| 7 | `arnlp.summarization.ExtractiveSummarizer` | extractive summaries (TextRank + MMR) |
| 8 | `arnlp.evaluation` | summarization metrics (ROUGE, grounding, hallucination) |
| 9 | `app/` (Streamlit) | interactive semantic-search UI |

## The preprocessing pipeline

The Stage-2 pipeline emits **three views** of every article so each
downstream task gets the representation it needs:

| Field | Consumer | Notes |
|-------|----------|-------|
| `embedding_view`  | E5, semantic search | digits, decimals (`1.7`), Latin preserved; `%` → `بالمئة` |
| `clustering_view` | BERTopic                 | numbers masked (`YEAR_TOKEN`, `NUM_TOKEN`, `PCT_TOKEN`, `MONEY_TOKEN`); `أبريل/نيسان` correctly split |
| `lemmas`          | BERTopic class-TF-IDF    | bare (no diacritics), aligned 1:1 with `filtered_tokens`, post-filtered against stopwords |
| `dates`           | NER / ambiguity / search | structured `{raw, iso, kind}` records |

Lemmatization uses CAMeL's `MLELemmatizer` (fast, context-free).

## The search app

`app/` is a Streamlit UI that encodes an Arabic query with
`multilingual-e5-large` and ranks the news articles by cosine similarity
against the precomputed `data/embeddings/e5_large.npy`, showing each hit's
BERTopic topic name. It reads the repo's `data/` directly (paths resolve
relative to the repo root). See [`app/README.md`](app/README.md).

```bash
streamlit run app/streamlit_app.py
```

## Ambiguity & summarization

**Stage 6 — word-sense disambiguation** (`arnlp.ambiguity`) reuses the same
E5 + cosine approach as the search app: it ranks the candidate senses of an
ambiguous Arabic word by how close each sense's gloss sits to the context.

```python
from arnlp.ambiguity import EmbeddingDisambiguator

wsd = EmbeddingDisambiguator()                       # defaults to E5Encoder
ranked = wsd.disambiguate_word("عين", "تدفقت مياه العين من سفح الجبل")
print(ranked[0].sense.sense_id)                      # -> "spring"
```

**Stage 7 — extractive summarization** (`arnlp.summarization`) selects the most
central, non-redundant **original** sentences (TextRank + MMR over E5), so it
*cannot hallucinate*. An opt-in `LLMSummarizer` (OpenAI) is also available.

```python
from arnlp.summarization import ExtractiveSummarizer

summarizer = ExtractiveSummarizer()
print(summarizer.summarize(article_body))
```

## Development

```bash
# format + lint
ruff check .  &&  ruff format .

# type-check
mypy src/arnlp

# fast unit tests
pytest tests/unit -q

# full suite (loads CAMeL models)
pytest
```

## License / Citation

(TBD)
