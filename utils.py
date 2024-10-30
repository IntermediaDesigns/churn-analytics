import plotly.graph_objects as go


def update_chart_styling(fig):
    """Apply consistent styling to Plotly charts"""
    fig.update_layout(
        paper_bgcolor="#1e2433",
        plot_bgcolor="#1e2433",
        font=dict(family="Segoe UI, sans-serif", color="#ffffff"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=300,
        # Center all titles
        title={
            "y": 0.95,  # Adjust title vertical position
            "x": 0.5,  # Center title horizontally
            "xanchor": "center",
            "yanchor": "top",
        },
    )
    return fig


def create_guage_chart(probability):
    # Ensure probability is a valid number and convert to percentage
    try:
        probability_pct = float(probability) * 100
    except (TypeError, ValueError):
        probability_pct = 0  # Default value if conversion fails

    # Define color based on probability threshold
    if probability_pct < 30:
        color = "#4CAF50"  # Modern green
    elif probability_pct < 60:
        color = "#FFC107"  # Modern yellow
    else:
        color = "#F44336"  # Modern red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_pct,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "Churn Risk",
                "font": {"size": 20, "color": "#ffffff"},
            },
            number={
                "font": {"size": 40, "color": "#ffffff"},
                "suffix": "%",
                "valueformat": ".1f",  # Format to 1 decimal place
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#ffffff",
                    "tickmode": "linear",
                    "tickformat": ".0f",
                    "dtick": 20,  # Show ticks every 20%
                },
                "bar": {"color": color, "thickness": 0.75},
                "bgcolor": "#262d3d",
                "borderwidth": 2,
                "bordercolor": "#3d4554",
                "steps": [
                    {"range": [0, 30], "color": "#4CAF50", "thickness": 1},
                    {"range": [30, 60], "color": "#FFC107", "thickness": 1},
                    {"range": [60, 100], "color": "#F44336", "thickness": 1},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 4},
                    "thickness": 0.75,
                    "value": probability_pct,
                },
            },
        )
    )

    # Update layout with error handling
    fig.update_layout(
        paper_bgcolor="#1e2433",
        plot_bgcolor="#1e2433",
        font=dict(family="Segoe UI, sans-serif", color="#ffffff"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=300,
        title={
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
    )

    return fig


def create_model_probability_chart(probabilities):
    models = list(probabilities.keys())
    probs = list(probabilities.values())

    # Color bars based on probability values
    colors = [
        "#4CAF50" if p < 0.3 else "#FFC107" if p < 0.6 else "#F44336" for p in probs
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                y=models,
                x=probs,
                orientation="h",
                text=[f"{p:.1%}" for p in probs],
                textposition="auto",
                marker_color=colors,  # Use color array
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Model Predictions",
            "font": {"size": 20, "color": "#ffffff"},
        },
        xaxis_title="Probability",
        yaxis_title="Model",
        xaxis=dict(
            tickformat=".0%",
            range=[0, 1],
            gridcolor="#3d4554",
            tickcolor="white",
            tickfont={"color": "white"},
            title_font={"color": "white"},
        ),
        yaxis=dict(
            gridcolor="#3d4554",
            tickcolor="white",
            tickfont={"color": "white"},
            title_font={"color": "white"},
        ),
    )

    return update_chart_styling(fig)
