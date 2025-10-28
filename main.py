import streamlit as st
import PyPDF2
import numpy as np
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer
import faiss

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# LangChain text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Streamlit UI ---
st.title("RAG PDF Test App")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    # --- PDF Extraction ---
    def extract_text_from_pdf(file) -> str:
        reader = PyPDF2.PdfReader(file)
        text = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(text)

    raw_text = extract_text_from_pdf(uploaded_file)
    st.write(f"Extracted {len(raw_text)} characters from PDF")

    # --- Text Splitting ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_text(raw_text)
    st.write(f"Number of text chunks: {len(docs)}")

    # --- Embeddings ---
    EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    # --- FAISS Index ---
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    st.success("FAISS index created!")

    # --- LLM Setup ---
    GEN_MODEL = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if st.runtime.scriptrunner.is_running_with_streamlit else -1
    )

    # --- Retrieval Functions ---
    def retrieve(query, k=4):
        q_emb = embedder.encode([query])
        q_emb = np.array(q_emb).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((idx, float(score), docs[idx]))
        return results

    def make_prompt(context_chunks, question):
        context = "\n\n".join(context_chunks)
        return f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely:"

    def answer_question(question, k=4, max_new_tokens=150):
        results = retrieve(question, k)
        if not results:
            return "No relevant information found.", []
        ctxs = [r[2] for r in results]
        prompt = make_prompt(ctxs, question)
        out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
        return out, results

    # --- Question UI ---
    question = st.text_input("Enter your question about the PDF:")
    if question:
        answer, sources = answer_question(question)
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Top-k source chunks:")
        for i, (idx, score, text) in enumerate(sources):
            st.markdown(f"**Chunk {i}** (Score={score:.4f}):")
            st.write(text[:600] + "...")
