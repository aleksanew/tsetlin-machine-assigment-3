import numpy as np
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go

N_STATES = 8
TRACE_LEN = 400

def stationary_distribution(s, p_l_y, p_y, p_lbar_ybar):
    p_lbar_y = 1.0 - p_l_y                    # P(L̄|Y)
    p_ybar = 1.0 - p_y                        # P(Ȳ)
    q = p_l_y * p_y + p_lbar_ybar * p_ybar    # P(L|Y)P(Y) + P(L̄|Ȳ)P(Ȳ)
 
    unnormalised = np.array([
        p_y**4 * p_lbar_y**7,                         # pi_1
        p_y**3 * p_lbar_y**6 * s    * q,              # pi_2
        p_y**2 * p_lbar_y**5 * s**2 * q**2,           # pi_3
        p_y    * p_lbar_y**4 * s**3 * q**3,           # pi_4
                 p_lbar_y**3 * s**4 * q**4,           # pi_5
        p_l_y    * p_lbar_y**2 * s**5 * q**4,         # pi_6
        p_l_y**2 * p_lbar_y    * s**6 * q**4,         # pi_7
        p_l_y**3               * s**7 * q**4,         # pi_8
    ])
    total = unnormalised.sum()
    if total == 0:
        return np.zeros(N_STATES)
    alpha = 1.0 / total
    return alpha * unnormalised

def ratios(s, p_l_y, p_y, p_lbar_ybar):
    """Ratio pi_{k+1}/pi_k in the forgotten and memorized halves."""
    q = p_l_y * p_y + p_lbar_ybar * (1 - p_y)
    down = p_y * (1 - p_l_y)                    # P(Y, L̄)
    r_f = np.inf if down == 0 else s * q / down
    r_m = np.inf if p_l_y == 1 else s * p_l_y / (1 - p_l_y)
    return r_f, r_m

def simulate(state, n_steps, s, p_l_y, p_y, p_lbar_ybar, rng):
    visited = np.empty(n_steps, dtype=int)
    for t in range(n_steps):
        y = rng.random() < p_y
        l = (rng.random() < p_l_y) if y else (rng.random() >= p_lbar_ybar)
        forgotten = state <= 4
        if y and l:                                   # memorize
            state = min(state + 1, N_STATES)
        elif y and not l:                             # forget with prob 1/s
            if rng.random() < 1.0 / s:
                state = max(state - 1, 1)
        elif (not y) and (not l) and forgotten:       # memorize (negative example)
            state = state + 1
        visited[t] = state
    return state, visited

def style_axes(fig):
    fig.update_xaxes(showline=True, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linecolor="black", mirror=True, gridcolor="#eee")
    fig.update_layout(plot_bgcolor="white", margin=dict(l=60, r=20, t=40, b=50))
    return fig
 
 
def distribution_figure(pi, counts):
    states = list(range(1, 9))
    fig = go.Figure(go.Bar(
        x=states, y=pi, name="Theory (equations)",
        marker_color="lightgrey", marker_line_color="dimgrey", marker_line_width=1,
        hovertemplate="State %{x}<br>π = %{y:.4f}<extra></extra>",
    ))
    total = sum(counts)
    if total > 0:
        fig.add_trace(go.Scatter(
            x=states, y=np.array(counts) / total, mode="markers",
            name=f"Simulation ({total:,} steps)",
            marker=dict(color="crimson", size=11, symbol="diamond"),
            hovertemplate="State %{x}<br>fraction = %{y:.4f}<extra></extra>",
        ))
    fig.add_vline(x=4.5, line_dash="dash", line_color="black")
    fig.add_annotation(x=2.5, y=1.0, text="Forgotten", showarrow=False, yanchor="bottom")
    fig.add_annotation(x=6.5, y=1.0, text="Memorized", showarrow=False, yanchor="bottom")
    fig.update_layout(
        title="Stationary distribution",
        xaxis=dict(title="State", tickmode="linear", dtick=1, range=[0.5, 8.5]),
        yaxis=dict(title="Probability", range=[0, 1.08]),
        height=420, legend=dict(x=0.01, y=0.85),
    )
    return style_axes(fig)
 

def trace_figure(trace, step):
    start = step - len(trace) + 1
    fig = go.Figure()
    fig.add_hrect(y0=4.5, y1=8.5, fillcolor="#d8ecd8", opacity=0.5, line_width=0)
    fig.add_hrect(y0=0.5, y1=4.5, fillcolor="#f2dede", opacity=0.5, line_width=0)
    if trace:
        fig.add_trace(go.Scatter(x=list(range(start, step + 1)), y=trace,
                                 mode="lines", line_shape="hv", line_color="black",
                                 hovertemplate="step %{x}<br>state %{y}<extra></extra>"))
    fig.update_layout(
        title="Learning process: state of the automaton over time",
        xaxis_title="Step (one training example per step)",
        yaxis=dict(title="State", range=[0.5, 8.5], dtick=1),
        height=320, showlegend=False,
    )
    return style_axes(fig)
 
 
def s_effect_figure(s, p_l_y, p_y, p_lbar_ybar):
    s_grid = np.linspace(1, 25, 241)
    p_mem = [stationary_distribution(v, p_l_y, p_y, p_lbar_ybar)[4:].sum() for v in s_grid]
    fig = go.Figure(go.Scatter(x=s_grid, y=p_mem, mode="lines", line_color="black",
                               hovertemplate="s = %{x:.1f}<br>P(memorized) = %{y:.3f}<extra></extra>"))
    fig.add_vline(x=s, line_dash="dot", line_color="crimson")
    if p_l_y < 1:
        s_star = (1 - p_l_y) / p_l_y if p_l_y > 0 else np.inf  # where r_M = 1
        if 1 <= s_star <= 25:
            fig.add_vline(x=s_star, line_dash="dash", line_color="grey",
                          annotation_text="r_M = 1", annotation_position="top left")
    fig.update_layout(
        title="Effect of s: probability that the literal is memorized",
        xaxis_title="s", yaxis=dict(title="P(state ≥ 5)", range=[0, 1.05]),
        height=320, showlegend=False,
    )
    return style_axes(fig)
 
 

def slider(id_, lo, hi, step, value, marks):
    return html.Div([
        html.Label(id=f"{id_}-label", style={"fontWeight": "bold"}),
        dcc.Slider(id=id_, min=lo, max=hi, step=step, value=value, marks=marks),
    ], style={"marginBottom": "16px"})

prob_marks = {i / 10: f"{i / 10:.1f}" for i in range(0, 11, 2)}
btn = {"marginRight": "8px", "padding": "6px 14px", "cursor": "pointer"}
app = Dash(__name__)
app.title = "Literal Automaton"
app.layout = html.Div([
    html.H2("Literal Automaton: stationary distribution and learning"),
    html.Div([
        html.Div([
            slider("s", 1.0, 25.0, 0.1, 3.9, {i: str(i) for i in [1, 5, 10, 15, 20, 25]}),
            slider("p_l_y", 0.0, 1.0, 0.01, 0.6, prob_marks),
            slider("p_y", 0.0, 1.0, 0.01, 0.5, prob_marks),
            slider("p_lbar_ybar", 0.0, 1.0, 0.01, 0.6, prob_marks),
            html.Div(id="derived", style={"fontSize": "14px", "lineHeight": "1.6"}),
            html.Hr(),
            html.B("Learning simulation"),
            html.Div([
                html.Button("▶ Start", id="start", style=btn),
                html.Button("⏸ Pause", id="pause", style=btn),
                html.Button("↺ Reset", id="reset", style=btn),
            ], style={"margin": "8px 0"}),
            html.Label("Start state after reset:"),
            dcc.Slider(id="start-state", min=1, max=8, step=1, value=1,
                       marks={i: str(i) for i in range(1, 9)}),
            html.Label("Steps per update (speed):"),
            dcc.Slider(id="speed", min=1, max=200, step=1, value=5,
                       marks={1: "1", 50: "50", 100: "100", 200: "200"}),
        ], style={"width": "34%", "paddingRight": "25px"}),
        html.Div([dcc.Graph(id="dist", config={"displayModeBar": False})],
                 style={"width": "66%"}),
    ], style={"display": "flex"}),
    html.Div([
        html.Div([dcc.Graph(id="trace", config={"displayModeBar": False})], style={"width": "60%"}),
        html.Div([dcc.Graph(id="s-effect", config={"displayModeBar": False})], style={"width": "40%"}),
    ], style={"display": "flex"}),
    html.Hr(),
    dcc.Interval(id="tick", interval=150, disabled=True),
    dcc.Store(id="sim", data={"state": 1, "counts": [0] * 8, "trace": [1], "step": 0}),
], style={"fontFamily": "sans-serif", "maxWidth": "1200px", "margin": "auto", "padding": "20px"})
 
 
PARAMS = [Input("s", "value"), Input("p_l_y", "value"),
          Input("p_y", "value"), Input("p_lbar_ybar", "value")]
 
 
@app.callback(
    Output("s-label", "children"), Output("p_l_y-label", "children"),
    Output("p_y-label", "children"), Output("p_lbar_ybar-label", "children"),
    Output("derived", "children"), Output("s-effect", "figure"),
    *PARAMS,
)
def update_params(s, p_l_y, p_y, p_lbar_ybar):
    pi = stationary_distribution(s, p_l_y, p_y, p_lbar_ybar)
    r_f, r_m = ratios(s, p_l_y, p_y, p_lbar_ybar)
    derived = [
        html.Div(f"P(L̄ | Y) = 1 − P(L | Y) = {1 - p_l_y:.2f}"),
        html.Div(f"P(Ȳ) = 1 − P(Y) = {1 - p_y:.2f}"),
        html.Div(f"Ratio in forgotten half r_F = {r_f:.3g}"),
        html.Div(f"Ratio in memorized half r_M = {r_m:.3g}"),
        html.Div(f"Memorize threshold: P(L | Y) > 1/(s+1) = {1 / (s + 1):.3f}"),
        html.Div(html.B(f"P(memorized) = π5+…+π8 = {pi[4:].sum():.4f}")),
    ]
    return (f"s = {s:.1f}", f"P(L | Y) = {p_l_y:.2f}", f"P(Y) = {p_y:.2f}",
            f"P(L̄ | Ȳ) = {p_lbar_ybar:.2f}", derived,
            s_effect_figure(s, p_l_y, p_y, p_lbar_ybar))
 
 
@app.callback(
    Output("tick", "disabled"),
    Input("start", "n_clicks"), Input("pause", "n_clicks"),
    prevent_initial_call=True,
)
def start_pause(_start, _pause):
    return ctx.triggered_id == "pause"
 
 
rng = np.random.default_rng()
 
 
@app.callback(
    Output("sim", "data"),
    Input("tick", "n_intervals"), Input("reset", "n_clicks"), *PARAMS,
    State("sim", "data"), State("speed", "value"), State("start-state", "value"),
)
def step_simulation(_n, _reset, s, p_l_y, p_y, p_lbar_ybar, sim, speed, start_state):
    trig = ctx.triggered_id
    if trig == "reset":
        return {"state": start_state, "counts": [0] * 8, "trace": [start_state], "step": 0}
    if trig in ("s", "p_l_y", "p_y", "p_lbar_ybar"):
        # New parameters -> new target distribution: restart the histogram,
        # but keep the automaton where it is so you can watch it move.
        return {**sim, "counts": [0] * 8}
    if trig != "tick":
        return no_update
    state, visited = simulate(sim["state"], speed, s, p_l_y, p_y, p_lbar_ybar, rng)
    counts = np.array(sim["counts"]) + np.bincount(visited - 1, minlength=8)
    trace = (sim["trace"] + visited.tolist())[-TRACE_LEN:]
    return {"state": int(state), "counts": counts.tolist(), "trace": trace,
            "step": sim["step"] + speed}
 
 
@app.callback(
    Output("dist", "figure"), Output("trace", "figure"),
    Input("sim", "data"), *PARAMS,
)
def draw(sim, s, p_l_y, p_y, p_lbar_ybar):
    pi = stationary_distribution(s, p_l_y, p_y, p_lbar_ybar)
    return distribution_figure(pi, sim["counts"]), trace_figure(sim["trace"], sim["step"])
 
 
if __name__ == "__main__":
    app.run(debug=True)