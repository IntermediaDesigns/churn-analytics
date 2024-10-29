import plotly.graph_objects as go


def create_guage_chart(probability):
    if probability < 0.3:
        color = "green"
    elif probability < 0.6:
        color = "yellow"
    else:
        color = "red"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "Churn Probability",
                "font": {"size": 24, "color": "white"},
            },
            number={"font": {"size": 40, "color": "white"}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": color},
                "bgcolor": "black",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "red"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="black",
        font={"color": "white"},
        width=400,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="black",
    )

    return fig


def create_model_probability_chart(probabilities):
    models = list(probabilities.keys())
    probs = list(probabilities.values())

    fig = go.Figure(
        data=[
            go.Bar(
                y=models,
                x=probs,
                orientation="h",
                text=[f"{p:.2%}" for p in probs],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Churn Probability by Model",
        xaxis_title="Probability",
        yaxis_title="Model",
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig
