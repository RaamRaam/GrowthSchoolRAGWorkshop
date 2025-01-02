import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
from ollama import chat, ChatResponse

# Helper functions
def read_pdf(file):
    reader = PdfReader(file)
    content = ""
    for page in reader.pages:
        content += page.extract_text() + "\n"
    return content

def split_text_into_chunks(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text_chunks(chunks, embedding_model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Streamlit App
st.title("PDF Query and Embedding App")

# Initialize session state
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.chunk_data = None
    st.session_state.embedding_model_name = "all-mpnet-base-v2"

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    st.write("Embedding for the uploaded file is in progress...")
    pdf_content = read_pdf(uploaded_file)
    chunks = split_text_into_chunks(pdf_content, chunk_size=64)
    embeddings = embed_text_chunks(chunks, st.session_state.embedding_model_name)
    st.session_state.chunk_data = {"chunks": chunks, "embeddings": embeddings}
    st.session_state.faiss_index = build_faiss_index(embeddings)
    st.success("File processed and embeddings created!")

# Query input
query = st.text_input("Enter your query:")
if query and st.session_state.faiss_index:
    query_embedding = SentenceTransformer(st.session_state.embedding_model_name).encode([query], convert_to_numpy=True)
    distances, indices = st.session_state.faiss_index.search(query_embedding, k=3)
    response_chunks = '\n'.join([st.session_state.chunk_data["chunks"][i] for i in indices[0]])

    system_message = {
    'role': 'system',
    'content': (
        "You are a precise and factual assistant. "
        "Always provide accurate and concise answers to user queries based solely on the provided context. "
        "Avoid adding extra information or being overly creative."
        )
    }
    
    # Use LLM to generate a response
    response: ChatResponse = chat(model='llama3.2', 
                                  messages=[
                                        system_message,
                                        {
                                            'role': 'user',
                                            'content': f"Answer {query} from (do not hallucinate) in 100 words: {response_chunks}"
                                        },
                                    ],
                                 )
    st.subheader("Response")
    st.write(response['message']['content'])

# Reset button
# if st.button("Reset"):
#     st.session_state.faiss_index = None
#     st.session_state.chunk_data = None
#     st.experimental_rerun()