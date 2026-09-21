# Arabic News — Semantic Search & 3D Topic Galaxy (Streamlit)

A dark, **RTL**, glassy "news dashboard" over the Al Jazeera Arabic corpus
(Stage 9 of the `arnlp` pipeline): a breaking-news ticker, three tabs, and a 3D
cluster map that **follows your search**.

- **🔎 Semantic search** — encodes the Arabic query with `multilingual-e5-large`
  and ranks news by cosine similarity. Each hit is a themed card with a score
  bar, a colored **topic chip** (clean label), and a category badge; expand it
  for two summary options — a **faithful extractive** summary (TextRank over E5,
  free/local) and an optional **OpenAI** summary that fixes merged words and
  summarizes — plus the **preprocessed/cleaned body**.
- **🌌 3D topic galaxy** — a UMAP→3D projection of the news embeddings, colored
  by topic. After a search, the result points light up as **gold stars** and
  their dominant **cluster is spotlighted** (everything else dims) so attention
  follows the query. Real JS **auto-rotate**, drag to orbit, scroll to zoom.
- **ℹ️ About** — how the pipeline fits together.

## Files

| File | Role |
|------|------|
| `streamlit_app.py` | UI: hero, ticker, tabs, cards, search↔galaxy wiring, caching |
| `theme.py` | Dark-glass RTL CSS, palette, category/topic colors, ticker, card builders |
| `cluster_view.py` | UMAP→3D (cached) + Plotly galaxy + JS auto-rotate (`galaxy_html`) |
| `search_engine.py` | `SemanticSearchEngine` — streamed data load + cosine search (E5 lazy) |
| `topic_labels.py` | Clean Arabic cluster names from c-TF-IDF (drops stopwords/placeholders) |

## Data it reads (from the repo's `data/`)

Paths resolve relative to the repo root, so it runs from any CWD:

- `data/processed/preprocessed.jsonl` — article source (**cleaned body**, order matches the embeddings); streamed, keeping only display fields
- `data/embeddings/e5_large.npy` — L2-normalized E5 vectors `(8635, 1024)`
- `data/clusters/clusters.jsonl` — `{url, topic, prob}` per article
- `data/clusters/topics_info.csv` — BERTopic representation (input to the labeler)
- `data/clusters/topic_labels.json` — **clean** topic names (auto-built if missing)
- `data/clusters/umap_3d.npy` — **cached** 3D coords (auto-computed on first run, ~30 s)

## Run

```bash
pip install -r requirements.txt        # streamlit, plotly, sentence-transformers, …
streamlit run app/streamlit_app.py     # from the repo root
```

Heavy models load **lazily**: data + galaxy are instant; the ~2 GB E5 model
loads on the **first search**. The **OpenAI** summary is on-demand and needs
`OPENAI_API_KEY` in the environment (and `pip install openai`); the extractive
summary reuses the already-loaded E5 (no extra model). Only `category == "أخبار"`
(news) articles are searched.

## Notes

- **Topic names:** `topic_labels.py` turns BERTopic's noisy `0_اسرائيل_في_num_token`
  into `إسرائيل · إيران · صواريخ` by reusing `arnlp.preprocessing` stopwords +
  placeholder tokens and collapsing near-duplicates. (An `OPENROUTER_API_KEY`
  could drive an LLM labeler for even cleaner phrases.)
- **Triton guard:** the app registers `triton` as unimportable at startup — on a
  CPU-only box `torch._dynamo` otherwise probes `triton` and segfaults (same
  guard as `arnlp.clustering._disable_triton`). Harmless on GPU.
- The 3D reduction reuses `arnlp.clustering.build_umap` (cosine), consistent with
  how the topics were actually clustered.

## Quick demo without the UI

`notebooks/04_search_demo.ipynb` exercises `SemanticSearchEngine.search()`
directly with sample Arabic queries.
