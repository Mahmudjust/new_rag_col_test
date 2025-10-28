import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter

# === CONFIG ===
st.set_page_config(page_title="RAG PDF Q&A", layout="centered")
st.title("RAG PDF Q&A (Cloud-Powered)")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GEN_MODEL = "google/flan-t5-small"

# === HUGGING FACE TOKEN ===
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Set `HF_TOKEN` in Streamlit Secrets (read token from HF).")
    st.stop()

# === UPLOAD PDF ===
uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if "query_engine" not in st.session_state:
            with st.spinner("Building index..."):
                # 1. Embedding (CPU, tiny)
                embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

                # 2. LLM via Hugging Face Inference Endpoint (CLOUD)
                llm = HuggingFaceLLM(
                    model_name=GEN_MODEL,
                    token=HF_TOKEN,
                    max_new_tokens=150,
                    generate_kwargs={"do_sample": False},
                    # This uses HF's free inference server
                    inference_server_url=f"https://api-inference.huggingface.co/models/{GEN_MODEL}"
                )

                # 3. Settings
                Settings.embed_model = embed_model
                Settings.llm = llm
                Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=100)

                # 4. Index
                reader = SimpleDirectoryReader(input_files=[pdf_path])
                documents = reader.load_data()
                index = VectorStoreIndex.from_documents(documents, show_progress=True)

                st.session_state.query_engine = index.as_query_engine(
                    similarity_top_k=4,
                    response_mode="compact"
                )
            st.success("Ready! Ask your question.")

        # === Q&A ===
        question = st.text_input("Ask a question:")
        if question:
            with st.spinner("Answering..."):
                response = st.session_state.query_engine.query(question)

            st.subheader("Answer")
            st.write(response.response)

            st.subheader("Sources")
            for i, node in enumerate(response.source_nodes):
                score = node.score
                text = node.node.text[:400] + ("..." if len(node.node.text) > 400 else "")
                st.write(f"**[{i+1}]** Score: {score:.4f}\n\n{text}\n")
