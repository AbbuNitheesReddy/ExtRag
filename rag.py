# ============================================
# 🧠 RAG SYSTEM (Auto-load FAISS + GPU/CPU Switch + Metadata)
# ============================================

import os
import sys
import subprocess

# -------------------------------
# 🧩 Auto-install required libraries
# -------------------------------
required_libs = [
    "numpy",
    "torch",  # for GPU detection
    "sentence-transformers",
    "ollama"
]

for lib in required_libs:
    try:
        __import__(lib)
    except ImportError:
        print(f"📦 Installing missing library: {lib} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# Detect if GPU is available
import torch
gpu_available = torch.cuda.is_available()

# Install correct FAISS version
try:
    if gpu_available:
        print("💠 GPU detected — installing faiss-gpu ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
    else:
        print("⚙️ No GPU found — using faiss-cpu ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
except Exception as e:
    print(f"⚠️ Failed installing FAISS variant: {e}")

# -------------------------------
# 🔧 Import after install
# -------------------------------
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# -------------------------------
# ⚙ Configuration
# -------------------------------
TOP_K = 2
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:8b"
RAG_OUTPUT_DIR = r"/app/Extractor/ExtRag/rag_output"

print("[INFO] Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

# -------------------------------
# 🔹 Load FAISS Index + Metadata
# -------------------------------
faiss_files = [f for f in os.listdir(RAG_OUTPUT_DIR) if f.endswith(".faiss")]
metadata_files = [f for f in os.listdir(RAG_OUTPUT_DIR) if f.endswith(".json")]

if not faiss_files or not metadata_files:
    print("Files found in rag_output:", os.listdir(RAG_OUTPUT_DIR))
    raise FileNotFoundError(f"No FAISS index or metadata JSON found in '{RAG_OUTPUT_DIR}'.")

index_path = os.path.join(RAG_OUTPUT_DIR, faiss_files[0])
metadata_path = os.path.join(RAG_OUTPUT_DIR, metadata_files[0])

print(f"[INFO] Using FAISS index: {index_path}")
print(f"[INFO] Using metadata: {metadata_path}")

# Load index
faiss_index = faiss.read_index(index_path)

# Move to GPU if available
if gpu_available:
    try:
        print("⚡ Moving FAISS index to GPU...")
        res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
        print("✅ FAISS is now running on GPU.")
    except Exception as e:
        print(f"⚠️ GPU transfer failed, using CPU instead: {e}")
else:
    print("💻 Running FAISS on CPU.")

# Load metadata
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)
knowledge_base = [item['text'] for item in metadata]

# -------------------------------
# 🔍 Context Retrieval
# -------------------------------
def retrieve_context(query):
    query_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = faiss_index.search(query_emb, TOP_K)
    context = "\n".join([knowledge_base[i] for i in indices[0]])
    return context if context.strip() else "No documents indexed."

# -------------------------------
# 💬 Query Type Detection
# -------------------------------
def detect_query_type(query):
    detailed_keywords = ["explain", "describe", "how", "why", "derive", "difference", "steps", "process", "workflow", "detailed", "in detail"]
    return "detailed" if any(word in query.lower() for word in detailed_keywords) else "simple"

# -------------------------------
# 🤖 Generate Answer using Ollama
# -------------------------------
def generate_answer(context, question):
    if context == "No documents indexed.":
        return "I could not find relevant documents."

    query_type = detect_query_type(question)
    style_instruction = "Provide a concise answer (1–3 sentences)." if query_type=="simple" else "Provide a detailed explanation with examples."

    prompt = f"""
You are an AI tutor restricted to the given context.
Never use your general knowledge or external information.

### Context:
{context}

### Question:
{question}

### Instructions:
{style_instruction}

If the answer is not found in the context above, reply exactly:
"I could not find this in the given material."
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "Answer strictly from provided context."},
                {"role": "user", "content": prompt}
            ]
        )
        if isinstance(response, dict) and "message" in response:
            return response["message"].get("content", "").strip()
        elif hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content.strip()
        else:
            return str(response)
    except Exception as e:
        return f"⚠ Generation failed: {e}"

# -------------------------------
# 🧑‍💻 Interactive Loop
# -------------------------------
if __name__ == "__main__":
    print("\n📚 AI JSON Tutor (Connected to Extraction Output)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("🧑‍🎓 Ask a question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        context = retrieve_context(query)
        answer = generate_answer(context, query)
        print(f"\n🤖 AI: {answer}\n")
