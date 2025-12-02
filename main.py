import os
import pickle
import numpy as np
import ollama
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
from typing import List, Tuple, Dict
from tqdm import tqdm


# --- CONFIGURATION ---
class Config:
    # Paths (Use raw strings or forward slashes)
    TESSERACT_CMD = r"C:\Ahsan\Work\python open source files\tesseract.exe"
    POPPLER_PATH = r'C:\Ahsan\Work\python open source files\Release-25.11.0-0\poppler-25.11.0\Library\bin'

    # Input/Output
    PDF_PATH = Path('./data/CBRE_-_UAE_Real_Estate_Market_.pdf')
    CACHE_DIR = Path('./data/cache')

    # Models
    EMBED_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
    LLM_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

    # Parameters
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50  # Important for context continuity
    TOP_K = 3


# Apply Tesseract Config
pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD

# Ensure cache directory exists
Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)


class LocalRAG:
    def __init__(self):
        self.vector_db: List[Dict] = []
        self.embeddings_matrix = None

    # -----------------------------------------------------------
    # 1. OCR & Text Extraction (With Caching)
    # -----------------------------------------------------------
    def extract_text(self, pdf_path: Path) -> str:
        cache_path = Config.CACHE_DIR / f"{pdf_path.stem}.txt"

        # Check cache first
        if cache_path.exists():
            print("💾 Loading text from cache...")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()

        print("📄 Reading PDF & converting to images...")
        try:
            pages = convert_from_path(pdf_path, dpi=300, poppler_path=Config.POPPLER_PATH)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

        print("🔍 Performing OCR (this may take a while)...")
        extracted_text = []
        for page in tqdm(pages, desc="OCR Progress"):
            text = pytesseract.image_to_string(page)
            extracted_text.append(text)

        full_text = "\n".join(extracted_text)

        # Save to cache
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        return full_text

    # -----------------------------------------------------------
    # 2. Advanced Chunking (Window + Overlap)
    # -----------------------------------------------------------
    def chunk_text(self, text: str) -> List[str]:
        # Clean up excessive newlines/whitespace for better embedding
        text = " ".join(text.split())

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + Config.CHUNK_SIZE
            chunk = text[start:end]
            chunks.append(chunk)
            # Move forward by size minus overlap
            start += (Config.CHUNK_SIZE - Config.CHUNK_OVERLAP)

        return chunks

    # -----------------------------------------------------------
    # 3. Vector Database (With Persistence)
    # -----------------------------------------------------------
    def build_or_load_index(self, chunks: List[str], pdf_name: str):
        db_path = Config.CACHE_DIR / f"{pdf_name}_db.pkl"

        if db_path.exists():
            print("💾 Loading vector DB from disk...")
            with open(db_path, 'rb') as f:
                data = pickle.load(f)
                self.vector_db = data['chunks']
                self.embeddings_matrix = data['matrix']
            return

        print("🧠 Generating embeddings...")
        embeddings = []

        for chunk in tqdm(chunks, desc="Embedding"):
            response = ollama.embed(model=Config.EMBED_MODEL, input=chunk)
            # Handle API variations (sometimes returns list of lists, sometimes flat)
            emb = response['embeddings'][0] if 'embeddings' in response else []

            self.vector_db.append({"text": chunk, "embedding": emb})
            embeddings.append(emb)

        # Convert to numpy array for fast calculation
        self.embeddings_matrix = np.array(embeddings)

        # Save to disk
        with open(db_path, 'wb') as f:
            pickle.dump({'chunks': self.vector_db, 'matrix': self.embeddings_matrix}, f)

        print(f"📌 Total chunks indexed: {len(self.vector_db)}")

    # -----------------------------------------------------------
    # 4. Retrieval (NumPy Optimized)
    # -----------------------------------------------------------
    def retrieve(self, query: str) -> List[str]:
        # Embed query
        q_resp = ollama.embed(model=Config.EMBED_MODEL, input=query)
        q_emb = np.array(q_resp['embeddings'][0])

        # Cosine Similarity: (A . B) / (||A|| * ||B||)
        # 1. Dot product of Query vs All Chunks
        dot_products = np.dot(self.embeddings_matrix, q_emb)

        # 2. Norms (magnitude)
        norm_q = np.linalg.norm(q_emb)
        norm_matrix = np.linalg.norm(self.embeddings_matrix, axis=1)

        # 3. Calculate scores
        scores = dot_products / (norm_matrix * norm_q)

        # 4. Get Top K indices
        top_k_indices = np.argsort(scores)[-Config.TOP_K:][::-1]

        return [self.vector_db[i]['text'] for i in top_k_indices]

    # -----------------------------------------------------------
    # 5. Generation
    # -----------------------------------------------------------
    def ask(self, query: str):
        relevant_chunks = self.retrieve(query)

        # Create a visually distinct context block
        formatted_context = "\n---\n".join(relevant_chunks)

        system_prompt = (
            "You are an expert analyst based on the provided document.\n"
            "Strictly follow these rules:\n"
            "1. Answer ONLY using the context below.\n"
            "2. If the answer is not in the context, say 'I cannot find that information in the document'.\n\n"
            f"Context:\n{formatted_context}"
        )

        print("\n🤖 Assistant:\n")
        stream = ollama.chat(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            stream=True
        )

        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print("\n" + "-" * 50 + "\n")


# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------
def main():
    rag = LocalRAG()

    # 1. Get Text
    raw_text = rag.extract_text(Config.PDF_PATH)
    if not raw_text:
        print("❌ No text extracted. Exiting.")
        return

    # 2. Chunk
    chunks = rag.chunk_text(raw_text)

    # 3. Build DB
    rag.build_or_load_index(chunks, Config.PDF_PATH.stem)

    # 4. Loop
    print("\n✅ System Ready! Type 'exit' to quit.\n")
    while True:
        query = input("❓ Question: ")
        if query.lower() in ['exit', 'quit']:
            break
        rag.ask(query)


if __name__ == "__main__":
    main()
