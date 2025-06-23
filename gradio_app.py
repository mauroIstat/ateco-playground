import gradio as gr
from modules.rag import build_kb

base = build_kb(
    path="data/ateco_2025_leaves.csv",
    model_id="paraphrase-multilingual-MiniLM-L12-v2"
)

def echo_with_label(message, history):
    results = base.search(message, top_k=10)
    codes = [r.metadata["code"] for r in results[0]]
    names = [r.metadata["title"] for r in results[0]]
    scores = [r.score for r in results[0]]
    return message, {f"{c}: {n}": float(s) for c, n, s in zip(codes, names, scores)}

with gr.Blocks() as app:
    with gr.Row(equal_height=True):
        # Left column: the label
        with gr.Column(scale=4):
            results_label = gr.Label(label="RAG Results", show_heading=False, num_top_classes=5)

        # Right column: chat interface
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(type="messages", render=False)
            textbox = gr.Textbox(
                placeholder="Type your query…", show_label=False
            )
            gr.ChatInterface(
                fn=echo_with_label,
                type="messages",
                chatbot=chatbot,
                textbox=textbox,
                additional_outputs=[results_label],
            )

app.launch()