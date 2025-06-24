import plotly.graph_objects as go

def plot_scores(data, height=100):
    codes = data["code"].tolist()[::-1]
    texts = data["title"].tolist()[::-1]
    scores = data["score"].tolist()[::-1]

    bar_height = 0.3
    y_positions = list(range(len(scores)))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=scores,
        y=y_positions,
        orientation='h',
        marker_color='mediumslateblue',
        width=bar_height,
        text=[f"{s:.3f}" for s in scores],
        textposition='outside',
        showlegend=False
    ))

    for y, code, title in zip(y_positions, codes, texts):
        fig.add_annotation(
            x=0, y=y + bar_height / 2 + 0.15,
            text=f"<b>{code}</b>: {title}",
            showarrow=False,
            font=dict(size=13, color='white'),
            xanchor='left',
            align='left',
            xshift=0
        )

    fig.update_layout(
        height=height * len(data),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            range=[-0.1, 1.1],
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.2)',
            gridwidth=0.5,
            tickvals=[i/10 for i in range(11)],
            ticktext=["" for _ in range(11)],
            showticklabels=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        template='plotly_dark'
    )

    return fig
