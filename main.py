import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import LLM
from transformers import pipeline
import requests

# === CONFIG ===
st.set_page_config(page_title="RAG PDF Q&A", layout="centered")
st.title("RAG PDF Q&A (Cloud-Only)")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GEN_MODEL = "google/flan-t5-small"

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("Set HF_TOKEN in Streamlit Secrets.")
    st.stop()

# === CUSTOM LLM USING HF INFERENCE API ===
class HFCloudLLM(LLM):
    def __init__(self, model_name, token):
        self.model_name = model_name
        self.token = token
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {token}"}

    def complete(self, prompt, **kwargs):
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 150, "do_sample": False}
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            return f"Error: {response.text}"
        return response.json()[0]["generated_text"]

    def stream_complete(self, prompt, **kwargs):
        yield self.complete(prompt, **kwargs)

# === UPLOAD PDF ===
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if "query_engine" not in st.session_state:
            with st.spinner("Building index..."):
                # Embedding (runs on Streamlit CPU)
                embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

                # LLM (runs on HF cloud)
                llm = HFCloudLLM(model_name=GEN_MODEL, token=HF_TOKEN)

                Settings.embed_model = embed_model
                Settings.llm = llm
                Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=100)

                reader = SimpleDirectoryReader(input_files=[pdf_path])
                docs = reader.load_data()
                index = VectorStoreIndex.from_documents(docs)
                st.session_state.query_engine = index.as_query_engine(similarity_top_k=4)
            st.success("Ready!")

        question = st.text_input("Ask:")
        if question:
            with st.spinner("Thinking..."):
                resp = st.session_state.query_engine.query(question)
            st.write("**Answer:**", resp.response)
            st.write("**Sources:**")
            for i, node in enumerate(resp.source_nodes):
                st.write(f"[{i+1}] Score: {node.score:.3f}\n{node.node.text[:300]}...")
