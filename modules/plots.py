import plotly.graph_objects as go


def plot_scores(codes, texts, scores):
    codes = codes[::-1]
    texts = texts[::-1]
    scores = scores[::-1]

    # Parameters
    bar_height = 0.3  # thinner bars
    y_positions = list(range(len(scores)))

    # Create figure
    fig = go.Figure()

    # Add bars
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

    # Add code + title as annotations above each bar
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

    # Style layout
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        template='plotly_dark'
    )

    return fig
