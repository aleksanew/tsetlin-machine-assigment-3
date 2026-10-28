import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

def stationary_distributions(s, p_l_y, p_y, p_lbar_ybar):
    """Return pi_1..pi_8 from the given equations, normalised by alpha"""
    p_lbar_y = 1.0 - p_l_y
    p_ybar = 1.0 - p_y
    q = p_l_y + p_lbar_ybar * p_ybar

    unnormalised = np.array([
        p_y**4 * p_lbar_y**7,                                  # pi_1
        p_y**3 * p_lbar_y**6 * s    * q,                       # pi_2
        p_y**2 * p_lbar_y**5 * s**2 * q**2,                    # pi_3
        p_y    * p_lbar_y**4 * s**3 * q**3,                    # pi_4
                 p_lbar_y**3 * s**4 * q**4,                    # pi_5
        p_l_y    * p_lbar_y**2 * s**5 * q**4,                  # pi_6
        p_l_y**2 * p_lbar_y    * s**6 * q**4,                  # pi_7
        p_l_y**3               * s**7 * q**4,                  # pi_8
    ])

    total = unnormalised.sum()
    if total == 0:
        return np.zeros(8)
    alpha = 1.0 / total
    return alpha * unnormalised

def make_figure(pi):
    states = list(range(1, 9))
    fig = go.Figure(go.Bar(
        x=states, y=pi,
        marker_color="lightgrey", marker_line_color="dimgrey", marker_line_width=1,
        hovertemplate="State %{x}<br>π = %{y:.4f}<extra></extra>",
    ))
    # Divider between Forgotten (1-4) and Memorized (5-8)
    fig.add_vline(x=4.5, line_dash="dash", line_color="black")
    fig.add_annotation(x=2.5, y=1.0, yref="y", text="Forgotten", showarrow=False, yanchor="bottom")
    fig.add_annotation(x=6.5, y=1.0, yref="y", text="Memorized", showarrow=False, yanchor="bottom")
    fig.update_layout(
        xaxis=dict(title="State", tickmode="linear", dtick=1, range=[0.5, 8.5]),
        yaxis=dict(title="Stationary probability π", range=[0, 1.08]),
        plot_bgcolor="white", margin=dict(l=60, r=20, t=30, b=50), height=450,
    )
    fig.update_xaxes(showline=True, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linecolor="black", mirror=True, gridcolor="#eee")
    return fig

def slider(id_, label, lo, hi, step, value, marks):
    return html.Div([
        html.Label(label, id=f"{id_}-label", style={"fontWeight": "bold"}),
        dcc.Slider(id=id_, min=lo, max=hi, step=step, value=value, marks=marks, tooltip={"placement": "bottom", "always_visible": False}),
    ], style={"marginBottom": "18px"})

prob_marks = {i / 10: f"{i / 10:.1f}" for i in range(0, 11, 2)}

app = Dash(__name__)
app.title = "Literal Automaton – Stationary Distribution"
app.layout = html.Div([
    html.H2("Literal Automaton: Stationary Distribution (8 states)"),
    html.Div([
        html.Div([
            slider("s", "s", 1.0, 25.0, 0.1, 3.9, {i: str(i) for i in [1, 5, 10, 15, 20, 25]}),
            slider("p_l_y", "P(L | Y)", 0.0, 1.0, 0.01, 0.6, prob_marks),
            slider("p_y", "P(Y)", 0.0, 1.0, 0.01, 0.5, prob_marks),
            slider("p_lbar_ybar", "P(L̄ | Ȳ)", 0.0, 1.0, 0.01, 0.6, prob_marks),
            html.Div(id="derived", style={"marginTop": "10px", "fontSize": "14px"}),
        ], style={"width": "35%", "paddingRight": "30px"}),
        html.Div([dcc.Graph(id="bar-chart", config={"displayModeBar": False})],
                 style={"width": "65%"}),
    ], style={"display": "flex"}),
], style={"fontFamily": "sans-serif", "maxWidth": "1100px", "margin": "auto", "padding": "20px"})

@app.callback(
    Output("bar-chart", "figure"),
    Output("derived", "children"),
    Output("s-label", "children"),
    Output("p_l_y-label", "children"),
    Output("p_y-label", "children"),
    Output("p_lbar_ybar-label", "children"),
    Input("s", "value"),
    Input("p_l_y", "value"),
    Input("p_y", "value"),
    Input("p_lbar_ybar", "value"),
)
def update(s, p_l_y, p_y, p_lbar_ybar):
    pi = stationary_distributions(s, p_l_y, p_y, p_lbar_ybar)
    derived = [
        html.Div(f"P(L̄ | Y) = 1 − P(L | Y) = {1 - p_l_y:.2f}"),
        html.Div(f"P(Ȳ) = 1 − P(Y) = {1 - p_y:.2f}"),
        html.Div(f"P(memorized) = π5+…+π8 = {pi[4:].sum():.4f}"),
    ]
    return (make_figure(pi), derived,
            f"s = {s:.1f}", f"P(L | Y) = {p_l_y:.2f}",
            f"P(Y) = {p_y:.2f}", f"P(L̄ | Ȳ) = {p_lbar_ybar:.2f}")
 
 
if __name__ == "__main__":
    app.run(debug=True)