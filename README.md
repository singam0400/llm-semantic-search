# 🧠 LLM Semantic Search + QA with SBERT & DistilBERT

A compact, production-inspired **semantic search + question answering** system using **Sentence-BERT for dense retrieval** and **DistilBERT for local extractive QA**. 

---

## 🚀 What This Project Does

> Upload a `.txt` file → Ask a natural language question → Get an intelligent answer using local models.

✅ Sentence-BERT (`MiniLM-L6-v2`) generates dense embeddings  
✅ Custom cosine similarity returns top semantic matches  
✅ DistilBERT (SQuAD fine-tuned) generates an answer from top-k retrieved chunks  

---

## 🧩 Tech Stack

| Layer                | Component                              |
|----------------------|-----------------------------------------|
| 🔍 Embedding Model   | `sentence-transformers/paraphrase-MiniLM-L6-v2` |
| 📚 Search Engine     | NumPy-based cosine similarity            |
| 🧠 QA Model          | `distilbert-base-cased-distilled-squad` |
| ⚙️ Interface         | Google Colab (`.ipynb`)                  |
| 🧰 Utils             | Modular Python file (`utils.py`)         |

---

## 🔮 Potential Extensions

- Swap QA model with **open-weight LLM** like Mistral or TinyLlama  /  
- Support **multi-document RAG with citations**    
- Replace NumPy similarity with **FAISS or ChromaDB** for large-scale corpora

---
