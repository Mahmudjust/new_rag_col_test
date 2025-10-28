import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("RAG PDF Q&A")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Extract text
    reader = PyPDF2.PdfReader(uploaded_file)
    raw_text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_text(raw_text)
    
    # Embed documents
    embed_model = "paraphrase-multilingual-mpnet-base-v2"
    embedder = SentenceTransformer(embed_model)
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    # Load generation model (CPU/GPU safe)
    GEN_MODEL = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer,
                         device=0 if hasattr(torch, "cuda") and torch.cuda.is_available() else -1)
    
    # Q&A input
    question = st.text_input("Enter your question:")
    if question:
        # Retrieve top chunks
        q_emb = embedder.encode([question]).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb, 4)
        ctxs = [docs[idx] for idx in indices[0] if idx != -1]
        
        prompt = f"Context:\n{chr(10).join(ctxs)}\n\nQuestion: {question}\nAnswer concisely:"
        answer = generator(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]
        
        st.subheader("Answer")
        st.write(answer)
        
        st.subheader("Sources")
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx != -1:
                st.write(f"[{i}] Score={score:.4f}\n{docs[idx][:400]}...")

