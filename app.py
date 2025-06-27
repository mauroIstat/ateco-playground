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
if "activity" not in st.session_state:
    st.session_state.activity = None

## --- APP --- ##
if prompt := st.text_input("Attività svolta.", placeholder=params.DESCRIPTIONS["chat_placeholder"]):
    results = st.session_state.base.search(prompt, top_k=5)
    result_df = parse_retrieved(results)

    with st.spinner():
        results = st.session_state.base.search(prompt, top_k=5)
        result_df = parse_retrieved(results)
        activities = result_df["activity"].unique()
        filtered_result = result_df[result_df["activity"]==st.session_state.activity] if st.session_state.activity else result_df
        fig = plot_scores(filtered_result, 50)
        st.markdown(params.DESCRIPTIONS["init_message"])

        if len(activities) > 1:
            st.markdown(params.DESCRIPTIONS["select_activity_message"])
            col1, col2 = st.columns([1, 1])
            with col1:
                if len(activities) > 1:
                    st.selectbox("Seleziona la tua attività principale", options=activities, key="activity")

        with st.expander("Dettagli", expanded=False):
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
