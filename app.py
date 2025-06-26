import streamlit as st
from dotenv import load_dotenv
from modules import params
from modules.knowledge_base import get_base, parse_retrieved, parse_description
from modules.plots import plot_scores

load_dotenv()

## --- SESSION STATES --- ##
if "base" not in st.session_state:
    with st.spinner("Creazione della Knowledge Base..."):
        st.session_state.base = get_base(
            path="data/ateco_2025_leaf.csv",
            model_id="BAAI/bge-m3"
        )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "plots" not in st.session_state:
    st.session_state.plots = []

## --- APP --- ##

if prompt := st.text_input("Attività svolta.", placeholder=params.DESCRIPTIONS["chat_placeholder"]):
    results = st.session_state.base.search(prompt, top_k=5)
    result_df = parse_retrieved(results)

    with st.spinner():
        results = st.session_state.base.search(prompt, top_k=5)
        result_df = parse_retrieved(results)
        filtered_result = result_df[result_df["activity"]==st.session_state.activity]
        fig = plot_scores(filtered_result, 50)
        st.markdown(params.DESCRIPTIONS["init_message"])

        col1, col2 = st.columns([1, 1])
        with col1:
            st.selectbox("Seleziona la tua attività principale", options=result_df["activity"].unique(), key="activity")

        with st.expander("Dettagli", expanded=False):
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")



#for message in st.session_state.messages:
#    with st.chat_message(message["role"], avatar=message["avatar"] if "avatar" in message else None):
#        st.markdown(message["content"])
#
#if prompt := st.chat_input(placeholder=params.DESCRIPTIONS["chat_placeholder"]):
#    st.chat_message("user").markdown(prompt)
#    st.session_state.messages.append({"role": "user", "content": prompt})
#
#    results = st.session_state.base.search(prompt, top_k=5)
#    result_df = parse_retrieved(results)
#
#    with st.chat_message("assistant", avatar="resources/chatbot_ateco_logo.png"):
#        fig = plot_scores(result_df, 50)
#        st.markdown("Ho individuato is seguenti codici ATECO.")
#
#        with st.expander("Codici ATECO individuati.", expanded=False):
#            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
#            st.divider()
#
#            for idx, row in result_df.iterrows():
#                code, title, activity, desc = row["code"], row["title"], row["activity"], row["description"]
#                st.markdown(f"##### 🏷️ **:blue[{code} - {title}]**\n* Attività principale: _:blue[{activity}]_\n* Include:\n{parse_description(desc)}")
#
#        if len(result_df["activity"].unique()) > 1:
#            st.markdown("Per aiutarti a scegliere il codice ATECO corretto, ho bisogno di conoscere l'attività principale svolta dall'azienda.")
#            st.selectbox("Seleziona l'attività principale", options=result_df["activity"].unique(), key="activity")
#