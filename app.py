import streamlit as st
import torch
from transformers import pipeline
from modules.build_kb import build_kb
from semantic_search.data import build_corpus
from semantic_search.local import LocalKnowledgeBase

st.title("👷🏻‍♀️👨🏻‍🌾 ATECO 2025")
st.markdown("Cerca il codice ATECO della tua attività tramite linguaggio naturale.")

if "kb" not in st.session_state:
    with st.spinner("Caricamento della Knowledge Base..."):
        st.session_state.kb = build_kb(
            path="classification/ateco_2025/ateco_2025_level_4.csv",
            model_id="BAAI/bge-m3"
        )
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    with st.spinner("Caricamento del modello..."):
        st.session_state.llm = pipeline("text-generation", model="google/gemma-3-1b-it", device="cuda", torch_dtype=torch.bfloat16)

def stream_response(prompt: str):
    for chunk in st.session_state.llm.stream(prompt):
        yield chunk

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Descrivi la tua attività. Ad esempio: \"Produzione di vini\" o \"Attività di scenografi\"."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    results = st.session_state.kb.search(prompt, top_k=5)
    mrkwn = []
    for result in results:
        codes = [r.metadata["code"] for r in result]
        names = [r.text for r in result]
        mrkwn = [f"**{c}**: {n}" for c, n in zip(codes, names)]
    
    candidates = '\n\n'.join(mrkwn)
    
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": f"Sei un assistente per la classificazione di imprese in codici ATECO 2025 partendo dalle descrizioni delle loro attività. Ricevi 5 codici candidati, scegli il più opportuno, anche facendo domande per scegliere tra più codici o se credi che il codice giusto non sia tra i candidati. Rispondi in maniera concisa."},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Descrizione: {prompt}\n\nCandidati: {candidates}"},]
            },
        ],
    ]

    output = st.session_state.llm(messages, max_new_tokens=100)[0][0]["generated_text"][2]["content"]

    with st.chat_message("assistant"):
        st.markdown(output)
        with st.expander("Mostra tutti i candidati", expanded=True):
            st.markdown(candidates)
    st.session_state.messages.append({"role": "assistant", "content": output})
