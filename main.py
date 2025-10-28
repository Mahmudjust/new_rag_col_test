import streamlit as st
import os
import tempfile
import requests
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import LLM
from llama_index.core.llms.llm import CompletionResponse
from typing import Any, Generator

# === CONFIG ===
st.set_page_config(page_title="RAG PDF Q&A", layout="centered")
st.title("RAG PDF Q&A (100% Cloud)")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GEN_MODEL = "google/flan-t5-small"

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("Set HF_TOKEN in Streamlit Secrets.")
    st.stop()

# === FULLY IMPLEMENTED HF CLOUD LLM ===
class HFCloudLLM(LLM):
    def __init__(self, model_name: str, token: str):
        super().__init__()
        self.model_name = model_name
        self.token = token
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {token}"}

    @property
    def metadata(self) -> Any:
        return {"model_name": self.model_name}

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 150, "do_sample": False}
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            text = f"HF API Error: {response.text}"
        else:
            text = response.json()[0]["generated_text"]
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        yield self.complete(prompt, **kwargs)

    # Required stubs (not used in RAG)
    def chat(self, *args, **kwargs): raise NotImplementedError
    def achat(self, *args, **kwargs): raise NotImplementedError
    def stream_chat(self, *args, **kwargs): raise NotImplementedError
    def astream_chat(self, *args, **kwargs): raise NotImplementedError
    def acomplete(self, *args, **kwargs): raise NotImplementedError
    def astream_complete(self, *args, **kwargs): raise NotImplementedError

# === UPLOAD PDF ===
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if "query_engine" not in st.session_state:
            with st.spinner("Building index..."):
                embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
                llm = HFCloudLLM(model_name=GEN_MODEL, token=HF_TOKEN)

                Settings.embed_model = embed_model
                Settings.llm = llm
                Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=100)

                reader = SimpleDirectoryReader(input_files=[pdf_path])
                docs = reader.load_data()
                index = VectorStoreIndex.from_documents(docs)
                st.session_state.query_engine = index.as_query_engine(similarity_top_k=4)
            st.success("Ready! Ask anything.")

        question = st.text_input("Ask:")
        if question:
            with st.spinner("Thinking..."):
                resp = st.session_state.query_engine.query(question)
            st.write("**Answer:**", resp.response)
            st.write("**Sources:**")
            for i, node in enumerate(resp.source_nodes):
                st.write(f"[{i+1}] Score: {node.score:.3f}\n{node.node.text[:300]}...")
