import streamlit as st
from modules.rag import Llama, build_kb
from modules import params

st.title("👷🏻‍♀️👨🏻‍🌾 ATECO 2025")
st.logo("resources/assistant_logo.png", size="large")
st.set_page_config(
    page_title="ATECO 2025 Classificatore",
    page_icon="📌",
    layout="centered",
)
st.markdown("Cerca il codice ATECO della tua attività tramite linguaggio naturale.")

if "kb" not in st.session_state:
    with st.spinner("Creazione della Knowledge Base..."):
        st.session_state.kb = build_kb(
            path="classification/ateco_2025/ateco_2025_level_4.csv",
            model_id="BAAI/bge-m3"
        )
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "llm" not in st.session_state:
    with st.spinner("Caricamento del modello..."):
        st.session_state.llm = Llama(model_id="meta-llama/Llama-3.2-3B-Instruct")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"] if "avatar" in message else None):
        st.markdown(message["content"])

if prompt := st.chat_input("Descrivi la tua attività. Ad esempio: \"Produzione di vini\" o \"Attività di scenografi\"."):
    st.chat_message("user").markdown(prompt)
   
    prompt_list = [m["content"] for m in st.session_state.messages if m["role"] == "user"] if st.session_state.messages else []
    prompt_list.append(prompt)
    
    all_prompts = ". ".join(prompt_list)
    results = st.session_state.kb.search(all_prompts, top_k=5)
    mrkwn = []
    for result in results:
        codes = [r.metadata["code"] for r in result]
        names = [r.text for r in result]
        mrkwn = [f"**{c}**: {n}" for c, n in zip(codes, names)]
    
    candidates = '\n\n'.join(mrkwn)
    parsed_prompt = params.LLM["instruction_template"].format(description=prompt, candidates=candidates)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "user", "content": parsed_prompt})

    with st.chat_message("assistant", avatar="resources/chatbot_ateco_logo.png"):
        full_resp = st.write_stream(st.session_state.llm.stream_with_history(
            system=params.LLM["system_prompt"], messages=st.session_state.history, max_new_tokens=512)
        )
        st.info("#### Candidati\n" + candidates)
    st.session_state.messages.append({"role": "assistant", "content": full_resp, "avatar": "resources/chatbot_ateco_logo.png"})
    st.session_state.history.append({"role": "assistant", "content": full_resp})

with st.sidebar:
    if st.button("Cancella cronologia"):
        st.session_state.messages = []
        st.session_state.history = []
        st.success("Cronologia cancellata.")
    #st.markdown(st.session_state.llm.generate_description(
    #    system=params.LLM["system_prompt_parsing"], 
    #    history=st.session_state.history,
    #    max_new_tokens=512
    #))