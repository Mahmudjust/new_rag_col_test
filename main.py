import streamlit as st
import PyPDF2
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
import torch
import os
import tempfile

st.set_page_config(page_title="RAG PDF Q&A", layout="centered")
st.title("RAG PDF Q&A with **LlamaIndex**")

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
GEN_MODEL = "google/flan-t5-small"

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if "index" not in st.session_state:
            with st.spinner("Loading models & building index..."):
                embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

                llm = HuggingFaceLLM(
                    model_name=GEN_MODEL,
                    tokenizer_name=GEN_MODEL,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    model_kwargs={"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32},
                )

                Settings.embed_model = embed_model
                Settings.llm = llm
                Settings.chunk_size = 500
                Settings.chunk_overlap = 100

                reader = SimpleDirectoryReader(input_files=[pdf_path])
                documents = reader.load_data()
                index = VectorStoreIndex.from_documents(documents, show_progress=True)

                st.session_state.index = index
                st.session_state.query_engine = index.as_query_engine(
                    similarity_top_k=4,
                    response_mode="compact"
                )
            st.success("Index ready!")

        question = st.text_input("Enter your question:")
        if question:
            with st.spinner("Thinking..."):
                response = st.session_state.query_engine.query(question)

            st.subheader("Answer")
            st.write(response.response)

            st.subheader("Sources")
            for i, node_with_score in enumerate(response.source_nodes):
                node = node_with_score.node
                score = node_with_score.score
                preview = node.text[:400] + ("..." if len(node.text) > 400 else "")
                st.write(f"**[{i+1}]** Score: {score:.4f}\n\n{preview}\n")
