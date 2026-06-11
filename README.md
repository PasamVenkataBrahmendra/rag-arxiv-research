# RAG Evaluation Study: Retrieval-Augmented Generation for Scientific Paper QA

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange)
![LLM](https://img.shields.io/badge/LLM-LLaMA_3.3_70B-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview

This project systematically evaluates three retrieval strategies for 
Retrieval-Augmented Generation (RAG) on a corpus of 3,000 scientific 
paper abstracts from ArXiv. The goal is to understand how different 
retrieval approaches affect answer quality and grounding in a 
domain-specific QA system.

---

## Research Questions

1. Does dense retrieval always outperform sparse retrieval on 
   scientific text?
2. What chunk size gives optimal retrieval quality?
3. When does the LLM hallucinate even with correct retrieval?

---

## Dataset

- **Source:** ArXiv scientific paper abstracts
- **Size:** 3,000 papers, 3,725 chunks
- **Domain:** AI, Machine Learning, NLP, Computer Science
- **Test Set:** 20 questions generated from paper abstracts using 
  LLaMA 3.3 70B

---

## System Architecture