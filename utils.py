# utils.py
import re
import numpy as np

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=100):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def cosine_similarity_manual(A, B):
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return np.dot(A_norm, B_norm.T)

# -------------------------------
# Evaluation Block (Recall@5 + F1)
# -------------------------------
if __name__ == "__main__":
    from semantic_search_qa import semantic_search, qa_pipeline, model, chunks, embeddings

    eval_data = [
        {
            "query": "What is artificial intelligence?",
            "answer": "the simulation of human intelligence processes by machines"
        },
        {
            "query": "What is machine learning?",
            "answer": "a subset of AI that involves the use of algorithms and statistical models"
        },
        {
            "query": "What is NLP used for?",
            "answer": "to understand and respond to human language"
        }
    ]

    def compute_token_f1(pred, truth):
        pred_tokens = pred.lower().split()
        truth_tokens = truth.lower().split()
        
        common = set(pred_tokens) & set(truth_tokens)
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    retrieval_hits = []
    predicted_answers = []
    ground_truths = []

    for item in eval_data:
        query = item["query"]
        ground_truth = item["answer"]

        top_chunks = semantic_search(query, top_k=5)
        found = any(ground_truth.lower() in chunk.lower() for chunk, _ in top_chunks)
        retrieval_hits.append(int(found))

        context = "\n".join([chunk for chunk, _ in top_chunks])
        result = qa_pipeline({"context": context, "question": query})
        pred = result["answer"]

        predicted_answers.append(pred)
        ground_truths.append(ground_truth)

    recall_at_5 = sum(retrieval_hits) / len(retrieval_hits)
    f1_scores = [compute_token_f1(p, t) for p, t in zip(predicted_answers, ground_truths)]
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print(f"\n Recall@5: {recall_at_5:.2f}")
    print(f" F1-score (token overlap): {avg_f1:.2f}")
