# semantic_search_qa.ipynb

# Upload .txt file
from google.colab import files
uploaded = files.upload()

# Import utilities
import utils
import pickle, os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

#  Load + chunk text
text_path = list(uploaded.keys())[0]
text = utils.load_text(text_path)
chunks = utils.chunk_text(text, chunk_size=100)

# Embed with SBERT
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = model.encode(chunks)

#  Save embeddings
os.makedirs("/content/vector_store", exist_ok=True)
with open("/content/vector_store/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
with open("/content/vector_store/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
print("\n Embeddings saved.")

#  Load for querying
with open("/content/vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
with open("/content/vector_store/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

#  Setup QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

#  Semantic Search function
def semantic_search(query, top_k=3):
    query_embedding = model.encode([query])
    sims = utils.cosine_similarity_manual(query_embedding, embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [(chunks[i], sims[i]) for i in top_indices]

#  Answer Generator
def answer_query(query):
    top_chunks = semantic_search(query)
    context = "\n".join([chunk for chunk, _ in top_chunks])
    result = qa_pipeline({"context": context, "question": query})
    return result["answer"]

#  Ask!
query = input("\n Ask a question: ")
print("\n Answer:", answer_query(query))
