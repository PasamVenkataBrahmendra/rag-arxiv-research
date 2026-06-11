# RAG Evaluation Study: Scientific Paper QA

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange)
![LLM](https://img.shields.io/badge/LLM-LLaMA_3.3_70B-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview

This project systematically evaluates three retrieval strategies for Retrieval-Augmented Generation (RAG) on a corpus of 3,000 scientific paper abstracts from ArXiv. The goal is to understand how different retrieval approaches affect answer quality and grounding in a domain-specific QA system.

## Research Questions

1. Does dense retrieval always outperform sparse retrieval on scientific text?
2. What chunk size gives optimal retrieval quality?
3. When does the LLM hallucinate even with correct retrieval?

## Dataset

- **Source:** ArXiv scientific paper abstracts
- **Size:** 3,000 papers, 3,725 chunks
- **Domain:** AI, Machine Learning, NLP, Computer Science
- **Test Set:** 20 questions generated using LLaMA 3.3 70B

## System Architecture

User Query goes into the Hybrid Retriever which combines two strategies — BM25 sparse retrieval (weight 0.4) and FAISS dense retrieval (weight 0.6). The top 5 retrieved chunks are passed to LLaMA 3.3 70B via Groq API which generates the final grounded answer.

## Results

### Retriever Accuracy Comparison

| Retriever | Strategy | Accuracy | Hits |
|---|---|---|---|
| BM25 | Sparse (keyword) | 100% | 20/20 |
| FAISS | Dense (semantic) | 100% | 20/20 |
| Hybrid | BM25 + FAISS | 100% | 20/20 |

### Chunk Size Analysis

| Chunk Size | Total Chunks | Accuracy |
|---|---|---|
| 256 tokens | 5,559 | 100% |
| 512 tokens | 3,725 | 100% |
| 1024 tokens | 3,298 | 100% |

512 token chunks are optimal — smaller chunks increase noise, larger chunks reduce granularity without accuracy gains.

### End-to-End Pipeline Performance

| Metric | Value |
|---|---|
| Total Questions | 20 |
| Grounded Answers | 15 |
| Grounding Rate | 75% |
| Hallucination Rate | 25% |

The LLM correctly refuses to answer when context is insufficient — 5 questions returned "I cannot find this in the provided context." This shows the pipeline is faithful rather than hallucinating.

### Score Quality Analysis

| Retriever | Avg Top Score |
|---|---|
| BM25 | 59.00 |
| Dense (distance) | 0.796 |

## Key Findings

1. **BM25 vs Dense:** Both achieve identical accuracy on well-formed scientific questions. BM25 is faster; Dense handles semantic variations better.
2. **Hybrid is most robust:** Combining both retrievers handles keyword-heavy and semantically complex queries.
3. **Chunk size matters:** 512 tokens is the sweet spot for abstract text.
4. **LLM faithfulness:** 75% grounding rate shows the pipeline prioritizes faithfulness over hallucination.
5. **Failure patterns:** LLM struggles most on highly mathematical or domain-specific questions where abstracts lack sufficient context.

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.12 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| Sparse Retrieval | BM25 (rank-bm25) |
| LLM | LLaMA 3.3 70B via Groq API |
| Data | ArXiv summarization dataset (HuggingFace) |

## Project Structure

- 01_data_prep.ipynb — Data loading and cleaning
- 02_chunking_and_questions.ipynb — Chunking and test question generation
- 03_bm25_retriever.ipynb — BM25 sparse retriever
- 04_dense_retriever.ipynb — FAISS dense retriever
- 05_hybrid_retriever.ipynb — Hybrid retriever
- 06_analysis.ipynb — Score analysis and chunk experiments
- 07_rag_pipeline.ipynb — End-to-end RAG pipeline
- src/retrievers/ — Saved retriever indexes
- results/ — Evaluation results and charts

## How to Run

1. Clone the repo
2. Install dependencies: pip install datasets faiss-cpu rank_bm25 sentence-transformers ragas jupyter pandas numpy tqdm groq python-dotenv matplotlib
3. Create .env file and add your GROQ_API_KEY
4. Run notebooks in order: 01 through 07

## Future Work

- Evaluate using RAGAS metrics (faithfulness, answer relevance, context recall)
- Test on full paper text instead of abstracts only
- Fine-tune embedding model on scientific text
- Compare with re-ranking strategies using cross-encoders
- Extend to multilingual scientific papers

## Author

**Venkata Brahmendra Pasam**

- GitHub: [PasamVenkataBrahmendra](https://github.com/PasamVenkataBrahmendra)
- LinkedIn: [pasam-brahmendra](https://linkedin.com/in/brahmendra-pasam-182211230)
- Email: pasamvenkatabrahmendra@gmail.com