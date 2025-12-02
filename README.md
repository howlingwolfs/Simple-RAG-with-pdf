# 📘 Simple-RAG-with-PDF
A minimal implementation of Retrieval-Augmented Generation (RAG) using Ollama and PDF files.
### 🔍 What is RAG?
RAG (Retrieval-Augmented Generation) is an AI technique that enhances Large Language Model (LLM) responses by retrieving relevant information from an external knowledge base before generating an answer.
This makes the output:
- More accurate
- Up-to-date
- Factually grounded
Instead of relying only on the model’s internal training data, RAG ensures responses are backed by real context.

### ⚙️ How it Works (Simplified)
- Extract information from a source (e.g., PDF file).
- Split the text into chunks (paragraphs, pages, or sections).
- Generate embeddings — numerical representations of text that can be understood by machine learning models.
- User asks a question → the system retrieves the most relevant chunk.
- Pass the chunk to the LLM with instructions to answer only using the provided information.

### 🚀 Why Use This Project?
- Learn the basics of RAG with a simple, PDF-based workflow.
- Understand how embeddings and chunking improve retrieval.
- Experiment with Ollama + LLMs for grounded question answering.

### 📦 Installation
Clone the repository and install dependencies:

git clone https://github.com/your-username/Simple-RAG-with-pdf.git
cd Simple-RAG-with-pdf
pip install -r requirements.txt

#### Make sure you have Ollama installed and running from https://ollama.com/

### ▶️ Usage
Run the script with your PDF file:
steamlit run steamlit.py


### 📂 Project Structure
Simple-RAG-with-pdf/
│── main.py             # Main script
│── requirements.txt    # Dependencies
│── README.md           # Documentation
│── data/|              # PDF files
|── data/cashe/         # Cashe of pdf files

### 🧪 Example
You have a PDF about UAE property market(Given in data folder).

You can ask:

> In first month of 2025 how may residential transactions?

The system will:
- Retrieve the most relevant chunk from the PDF
- Pass it to the LLM
- Generate a grounded answer based only on the PDF content










