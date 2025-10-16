# ============================================================
# 🧠 RAG-READY PDF EXTRACTOR (GPU/CPU Auto + Multi-PDF + Batch)
# ============================================================

import os
import sys
import subprocess

# -------------------------------
# 🧩 AUTO-INSTALL REQUIRED LIBRARIES
# -------------------------------
base_libs = [
    "pymupdf", "pdfplumber", "pytesseract", "Pillow", "tqdm",
    "numpy", "sentence-transformers", "langchain", "torch"
]

def install_missing_packages():
    for lib in base_libs:
        try:
            __import__(lib if lib != "pymupdf" else "fitz")
        except ImportError:
            print(f"📦 Installing missing dependency: {lib} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

install_missing_packages()

# -------------------------------
# ⚙ DETECT GPU + INSTALL FAISS VARIANT
# -------------------------------
import torch

if torch.cuda.is_available():
    print("⚡ GPU detected — installing faiss-gpu ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
    FAISS_GPU = True
else:
    print("💻 No GPU detected — installing faiss-cpu ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    FAISS_GPU = False

# -------------------------------
# 🔧 IMPORT AFTER INSTALL
# -------------------------------
import fitz
import pdfplumber
import pytesseract
from PIL import Image
import tempfile
import json
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ============================================================
# 🧰 UTILITY FUNCTIONS
# ============================================================
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def print_section(title):
    print("\n" + "=" * 70)
    print(f"🧩 {title}")
    print("=" * 70 + "\n")

# ============================================================
# 📑 COLUMN DETECTION HELPERS (PyMuPDF + PDFPlumber)
# ============================================================
def detect_two_column_layout_pymupdf(page, threshold=0.3):
    blocks = page.get_text("blocks")
    if not blocks: return False
    x_centers = [(b[0] + b[2]) / 2 for b in blocks]
    page_width = page.rect.width
    left_count = sum(1 for x in x_centers if x < page_width / 2)
    right_count = sum(1 for x in x_centers if x >= page_width / 2)
    return left_count > 1 and right_count > 1 and min(left_count, right_count) / max(left_count, right_count) > threshold

def extract_two_column_pymupdf(page):
    blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
    page_width = page.rect.width
    left_col, right_col = [], []
    for b in blocks:
        text = b[4].strip()
        if not text: continue
        if b[0] < page_width / 2: left_col.append(text)
        else: right_col.append(text)
    return "\n".join(left_col + right_col)

def extract_text_pymupdf(pdf_path, start_page=0, end_page=None):
    text_pages = {}
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    end_page = end_page or total_pages
    for i in range(start_page, end_page):
        page = doc[i]
        try:
            if detect_two_column_layout_pymupdf(page):
                text = extract_two_column_pymupdf(page)
            else:
                text = page.get_text("text")
            text_pages[f"page_{i+1}"] = text.strip()
        except Exception as e:
            text_pages[f"page_{i+1}"] = f"[Error extracting text: {e}]"
    doc.close()
    return text_pages

def detect_two_column_pdfplumber(page, threshold=0.3):
    words = page.extract_words(x_tolerance=2, y_tolerance=3)
    if not words: return False
    x_centers = [(float(w['x0']) + float(w['x1'])) / 2 for w in words]
    page_width = page.width
    left_count = sum(1 for x in x_centers if x < page_width / 2)
    right_count = sum(1 for x in x_centers if x >= page_width / 2)
    return left_count > 1 and right_count > 1 and min(left_count, right_count) / max(left_count, right_count) > threshold

def extract_two_column_pdfplumber(page):
    words = page.extract_words(x_tolerance=2, y_tolerance=3)
    if not words: return ""
    page_width = page.width
    left_words = [w for w in words if float(w['x0']) < page_width / 2]
    right_words = [w for w in words if float(w['x0']) >= page_width / 2]

    def words_to_text(word_list):
        lines, current_line, current_y = [], [], None
        for w in sorted(word_list, key=lambda w: (round(float(w['top']) / 3), float(w['x0']))):
            y = round(float(w['top']) / 3)
            if current_y is None or y != current_y:
                if current_line: lines.append(" ".join(current_line))
                current_line = [w['text']]
                current_y = y
            else: current_line.append(w['text'])
        if current_line: lines.append(" ".join(current_line))
        return "\n".join(lines)

    left_text = words_to_text(left_words)
    right_text = words_to_text(right_words)
    return left_text + "\n" + right_text

def extract_text_pdfplumber_clean(pdf_path, start_page=0, end_page=None):
    text_pages = {}
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        end_page = end_page or total_pages
        for i in range(start_page, end_page):
            page = pdf.pages[i]
            try:
                if detect_two_column_pdfplumber(page):
                    text = extract_two_column_pdfplumber(page)
                else:
                    text = page.extract_text()
                text = re.sub(r"\s{2,}", " ", text or "")
                text = re.sub(r"ﬁ", "fi", text)
                text = re.sub(r"ﬂ", "fl", text)
                text = re.sub(r"\(cid:[0-9]+\)", "", text)
                text_pages[f"page_{i+1}"] = text.strip()
            except Exception as e:
                text_pages[f"page_{i+1}"] = f"[Error: {e}]"
    return text_pages

# ============================================================
# 🔍 OCR FALLBACK (for scanned pages)
# ============================================================
def extract_text_ocr(pdf_path, lang="eng", start_page=0, end_page=None):
    ocr_pages = {}
    temp_dir = tempfile.mkdtemp()
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    end_page = end_page or total_pages
    for i in range(start_page, end_page):
        page = doc[i]
        try:
            pix = page.get_pixmap(dpi=200)
            img_path = os.path.join(temp_dir, f"page_{i+1}.png")
            pix.save(img_path)
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img, lang=lang)
            ocr_pages[f"page_{i+1}"] = text.strip()
        except Exception as e:
            ocr_pages[f"page_{i+1}"] = f"[OCR failed: {e}]"
    doc.close()
    return ocr_pages

# ============================================================
# 🧠 MAIN PIPELINE (Batch + FAISS + Multi-PDF)
# ============================================================
print_section("🧠 Loading Embedding Model (GPU/CPU Auto)")
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

def process_pdf_batch(pdf_path, out_dir="./rag_output", batch_size=100):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Failed to open PDF '{pdf_path}': {e}")
        return None, []

    total_pages = len(doc)
    all_chunks = []
    print_section(f"📘 Extracting Text in Batches from {Path(pdf_path).name}")

    # -------------------------------
    # Extract text in batches
    # -------------------------------
    for start in range(0, total_pages, batch_size):
        end = min(start + batch_size, total_pages)
        print(f"\nProcessing pages {start+1} to {end} ...")
        plumber_text = extract_text_pdfplumber_clean(pdf_path, start, end)
        batch_text = "\n".join(plumber_text.values())
        if not batch_text.strip():
            # If text is empty, fallback to OCR
            print("⚠ No text found, running OCR fallback...")
            ocr_text = extract_text_ocr(pdf_path, start_page=start, end_page=end)
            batch_text = "\n".join(ocr_text.values())
        if not batch_text.strip():
            print("⚠ Pages empty even after OCR, skipping this batch.")
            continue

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        chunks = splitter.split_text(batch_text)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("⚠ No text chunks extracted from PDF. Skipping FAISS indexing.")
        doc.close()
        return None, []

    # -------------------------------
    # Generate embeddings
    # -------------------------------
    print_section("🧠 Generating Offline Embeddings (GPU Accelerated if available)")
    embeddings = embedding_model.encode(
        all_chunks,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Ensure 2D (even if single chunk)
    embeddings = np.atleast_2d(embeddings)

    # -------------------------------
    # Create / Update FAISS Index
    # -------------------------------
    print_section("💾 Creating/Updating FAISS Vector Index")
    dim = embeddings.shape[1]
    index_file = Path(out_dir) / "rag_index.faiss"
    if index_file.exists():
        try:
            index = faiss.read_index(str(index_file))
            index.add(embeddings)
        except Exception as e:
            print(f"⚠ Failed to read existing FAISS index, creating new: {e}")
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
    faiss.write_index(index, str(index_file))

    # -------------------------------
    # Save / Update Metadata JSON
    # -------------------------------
    rag_data_file = Path(out_dir) / "rag_metadata.json"
    if rag_data_file.exists():
        with open(rag_data_file, "r", encoding="utf-8") as f:
            rag_data = json.load(f)
        start_id = len(rag_data)
    else:
        rag_data = []
        start_id = 0

    rag_data.extend([{"chunk_id": i + start_id, "text": all_chunks[i]} for i in range(len(all_chunks))])
    save_json(rag_data, rag_data_file)

    print_section("✅ PDF Processing DONE")
    print(f"FAISS Index: {index_file}")
    print(f"Metadata JSON: {rag_data_file}")
    doc.close()
    return index, rag_data


# ============================================================
# 🚀 INTERACTIVE MULTI-PDF MODE
# ============================================================
if __name__ == "__main__":
    out_dir = "./rag_output"
    Path(out_dir).mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("🚀 RAG PDF Extractor Ready (GPU/CPU Auto Mode)")
    print("Enter PDF file paths or folder paths (type 'exit' to quit).")
    print("=" * 70 + "\n")

    while True:
        path_input = input("📂 Enter PDF file or folder path: ").strip()
        if path_input.lower() == "exit":
            print("\n👋 Exiting extractor... Goodbye!\n")
            break
        if not os.path.exists(path_input):
            print("❌ Path not found. Please check and try again.\n")
            continue

        if os.path.isdir(path_input):
            pdf_files = list(Path(path_input).glob("*.pdf"))
            if not pdf_files:
                print("❌ No PDF files found in this folder.\n")
                continue
            print(f"📄 Found {len(pdf_files)} PDF(s) in folder. Processing...\n")
            for pdf_file in pdf_files:
                process_pdf_batch(str(pdf_file), out_dir=out_dir, batch_size=100)
        else:
            process_pdf_batch(path_input, out_dir=out_dir, batch_size=100)