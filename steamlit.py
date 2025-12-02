import streamlit as st
import numpy as np
import ollama
from pdf2image import convert_from_bytes
import pytesseract


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Paths - You might still need to set these based on your local machine setup
    # If running on a cloud server, these paths usually need to be standard Linux paths
    TESSERACT_PATH = st.text_input("Tesseract Path", value=r"C:\Ahsan\Work\python open source files\tesseract.exe")
    POPPLER_PATH = st.text_input("Poppler Path",
                                 value=r"C:\Ahsan\Work\python open source files\Release-25.11.0-0\poppler-25.11.0\Library\bin")

    # Models
    EMBED_MODEL = st.text_input("Embedding Model", value="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf")
    LLM_MODEL = st.text_input("LLM Model", value="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")

    # Parameters
    CHUNK_SIZE = st.number_input("Chunk Size", value=500)
    CHUNK_OVERLAP = st.number_input("Chunk Overlap", value=50)
    TOP_K = st.slider("Retrieval Count (Top K)", 1, 10, 3)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Apply Tesseract Config
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# --- CACHED FUNCTIONS (Performance) ---

@st.cache_data(show_spinner=False)
def process_pdf(file_bytes) -> str:
    """
    Extracts text from PDF bytes using OCR.
    Cached so re-uploading the same file is instant.
    """
    try:
        # Convert bytes to images
        images = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)

        full_text = []
        progress_bar = st.progress(0, text="Performing OCR...")

        for i, image in enumerate(images):
            # OCR logic
            text = pytesseract.image_to_string(image)
            full_text.append(text)

            # Update progress
            progress = (i + 1) / len(images)
            progress_bar.progress(progress, text=f"Processing page {i + 1} of {len(images)}...")

        progress_bar.empty()
        return "\n".join(full_text)

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""


@st.cache_data(show_spinner=False)
def create_vector_db(text: str, chunk_size: int, overlap: int, model_name: str):
    """
    Chunks text and creates embeddings.
    Returns: vector_db (list of dicts), embeddings_matrix (numpy array)
    """
    # 1. Clean Text
    text = " ".join(text.split())

    # 2. Chunk Text
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)

    # 3. Embed
    vector_db = []
    embeddings = []

    progress_bar = st.progress(0, text="Generating Embeddings...")

    for i, chunk in enumerate(chunks):
        try:
            response = ollama.embed(model=model_name, input=chunk)
            emb = response['embeddings'][0] if 'embeddings' in response else []

            vector_db.append({"text": chunk, "embedding": emb})
            embeddings.append(emb)
        except Exception as e:
            st.warning(f"Failed to embed chunk {i}: {e}")

        # Update progress
        if i % 10 == 0:
            progress_bar.progress((i + 1) / len(chunks))

    progress_bar.empty()

    return vector_db, np.array(embeddings)


# --- MAIN APP LOGIC ---

st.title("🧠 Local RAG Document Assistant")
st.markdown("Upload a PDF, let the system read it using OCR, and ask questions.")

# 1. File Upload
uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

if uploaded_file is not None:
    # A. Extract Text
    with st.spinner("Reading PDF... (This uses OCR and might take a moment)"):
        file_bytes = uploaded_file.read()
        raw_text = process_pdf(file_bytes)

    if raw_text:
        st.success("✅ Text extracted successfully!")

        # B. Build Vector DB
        with st.spinner("Building Knowledge Base..."):
            # We use st.session_state to store the DB so it persists between chats
            if 'vector_db' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
                vector_db, embeddings_matrix = create_vector_db(
                    raw_text, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL
                )
                st.session_state.vector_db = vector_db
                st.session_state.embeddings_matrix = embeddings_matrix
                st.session_state.current_file = uploaded_file.name
                st.info(f"Indexed {len(vector_db)} chunks.")

        # C. Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question about the document..."):
            # 1. Display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Retrieval Logic
            with st.spinner("Thinking..."):
                q_resp = ollama.embed(model=EMBED_MODEL, input=prompt)
                q_emb = np.array(q_resp['embeddings'][0])

                # Vector math
                matrix = st.session_state.embeddings_matrix
                dot_products = np.dot(matrix, q_emb)
                norm_q = np.linalg.norm(q_emb)
                norm_matrix = np.linalg.norm(matrix, axis=1)
                scores = dot_products / (norm_matrix * norm_q)

                # Get Top K
                top_k_indices = np.argsort(scores)[-TOP_K:][::-1]
                relevant_chunks = [st.session_state.vector_db[i]['text'] for i in top_k_indices]

                # Construct Prompt
                context_block = "\n---\n".join(relevant_chunks)
                system_prompt = (
                    "You are a helpful assistant. Use ONLY the specific context below to answer the user's question.\n"
                    "If the answer is not in the context, say 'I cannot find that information in the document'.\n\n"
                    f"Context:\n{context_block}"
                )

                # 3. Generate Response
                stream = ollama.chat(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True
                )

                # Stream Output
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in stream:
                        content = chunk["message"]["content"]
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Debug: Show context used (optional, in expander)
                with st.expander("View Context Sources"):
                    for idx, chunk in enumerate(relevant_chunks):
                        st.markdown(f"**Chunk {idx + 1}:** {chunk}")
