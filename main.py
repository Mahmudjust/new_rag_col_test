import streamlit as st
import os
import tempfile
import requests
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import LLM
from typing import Any, Generator

# === CONFIG ===
st.set_page_config(page_title="RAG PDF Q&A", layout="centered")
st.title("RAG PDF Q&A (Cloud)")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GEN_MODEL = "google/flan-t5-small"

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("Set HF_TOKEN in Streamlit Secrets.")
    st.stop()

# === CUSTOM LLM (NO IMPORT ERRORS) ===
class HFCloudLLM(LLM):
    def __init__(self, model_name: str, token: str):
        self.model_name = model_name
        self.token = token
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {token}"}

    @property
    def metadata(self) -> Any:
        return type('obj', (), {'model_name': self.model_name})

    def complete(self, prompt: str, **kwargs) -> str:
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150}}
        resp = requests.post(self.api_url, headers=self.headers, json=payload)
        if resp.status_code != 200:
            return f"Error: {resp.text}"
        return resp.json()[0]["generated_text"]

    def stream_complete(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        yield self.complete(prompt, **kwargs)

    # Stubs
    def chat(self, *args, **kwargs): raise NotImplementedError
    def achat(self, *args, **kwargs): raise NotImplementedError
    def stream_chat(self, *args, **kwargs): raise NotImplementedError
    def astream_chat(self, *args, **kwargs): raise NotImplementedError
    def acomplete(self, *args, **kwargs): raise NotImplementedError
    def astream_complete(self, *args, **kwargs): raise NotImplementedError

# === UPLOAD ===
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if "engine" not in st.session_state:
            with st.spinner("Building..."):
                embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
                llm = HFCloudLLM(GEN_MODEL, HF_TOKEN)

                Settings.embed_model = embed_model
                Settings.llm = llm
                Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=100)

                docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
                index = VectorStoreIndex.from_documents(docs)
                st.session_state.engine = index.as_query_engine(similarity_top_k=4)
            st.success("Ready!")

        q = st.text_input("Ask:")
        if q:
            with st.spinner("Thinking..."):
                r = st.session_state.engine.query(q)
            st.write("**Answer:**", r.response)
            st.write("**Sources:**")
            for i, n in enumerate(r.source_nodes):
                st.write(f"[{i+1}] {n.node.text[:300]}...")
