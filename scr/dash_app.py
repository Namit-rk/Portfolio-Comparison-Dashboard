import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, ALL

from scr.config import STOCKS, START_DATE, END_DATE
from scr.portfolio_making import load_returns, PortfoliosForTickers
from scr.efficient_frontier import compute_frontier_data, plot_efficient_frontier


# ---------------------------------------------------------------------------
# App Configeration
# ---------------------------------------------------------------------------

# Setting default template
pio.templates.default = "plotly_dark"
pio.templates["plotly_dark"].layout.update( paper_bgcolor="#000000",
                                            plot_bgcolor="#000000",
                                            font=dict(color="#EAEAEA"),
                                            hovermode="x unified" )

# Setting default theme of app
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
server = app.server

# Portfolio universe Cache for different stock universe configeration
UNIVERSE_CACHE: dict[tuple, PortfoliosForTickers] = {}

def get_universe(tickers: list[str], 
                 start_date:str = START_DATE, 
                 end_date:str = END_DATE) -> PortfoliosForTickers:
    """
    Adds different stock configeration to universal cache
    """
    key = (tuple(sorted(tickers)),start_date, end_date)
    if key not in UNIVERSE_CACHE:
        UNIVERSE_CACHE[key] = PortfoliosForTickers(tickers)
    return UNIVERSE_CACHE[key]


# ---------------------------------------------------------------------------
# Helper Function
# ---------------------------------------------------------------------------

def section_header(title:str) -> html.Div:
    """
    Title format
    """
    return html.Div([
                html.H3(
                        title, 
                        style={"marginBottom": "20px",
                                "fontWeight": "600",
                                "marginTop": "25px",}
                        )
                ])


def load_filtered_returns(tickers:list[str], 
                          start:str, 
                          end:str) -> pd.DataFrame:
    """
    Filters the returns dataframe based on start and end date
    """
    if not tickers:
        return None
    return load_returns()[tickers].loc[start:end]


def graph_row(*graphs: tuple) -> dbc.Row:
    """
    In a row makes any number of graph cards to plot figures
    """
    def card(title, gid):
        return dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                                    html.H5(title),
                                    dcc.Graph(id=gid, config={"displayModeBar": False}),
                                    ]),
                            style={"backgroundColor": "#000000", "border": "1px solid #1F51FF"},
                            ),
                    width=12 // len(graphs),  # auto-splits the 12-column grid evenly
                    )
    
    return dbc.Row([card(*g) for g in graphs], style={"marginBottom": "25px"})


def selector_graph_card(selector_id:str,
                        graph_id:str):
    """
    Shows the graph based on selection
    """
    card = dbc.Card(
                dbc.CardBody([
                            html.Label("Select Strategy"),
                            dbc.RadioItems(
                                        id=selector_id,
                                        options=[],
                                        value=None,
                                        inline=True,
                                        ),
                            dcc.Graph(id=graph_id, config={"displayModeBar": False}),
                            ]),
                style={"backgroundColor": "#000000", "border": "1px solid #1F51FF", "marginBottom": "25px"},
                )
    return card


def calculate_test_metrics(name:str,
                           portfolio_test_returns:pd.DataFrame) -> dict[str,float]:
        """
        Calculates the portfolio performance from out of sample portfolio returns
        """
        cumulative = (1 + portfolio_test_returns).cumprod()
        ann_return = portfolio_test_returns.mean() * 252
        ann_vol = portfolio_test_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        return {"Strategy": name,"Annual Return": ann_return,"Volatility": ann_vol,
                "Sharpe": sharpe,"Max Drawdown": max_dd}


def make_test_split(returns:pd.DataFrame, split:float):
    split_idx = int(len(returns) * split)
    split_date = returns.index[split_idx]
    train_returns = returns.loc[:split_date]
    test_returns  = returns.loc[split_date:]
    return train_returns, test_returns, split_date


def backtesting(tickers, start_date, end_date, split=0.7):
    returns = load_filtered_returns(tickers,start_date, end_date)
    _, test_returns, split_date = make_test_split(returns, split)

    universe = get_universe(tickers, start_date, split_date)
    portfolios = universe.all_portfolios(just_names=False)

    results = []
    for name,port in portfolios:
        portfolio_test_returns = test_returns[tickers] @ port.weights
        metrics = calculate_test_metrics(name, portfolio_test_returns)
        results.append(metrics)

    df = pd.DataFrame(results).set_index("Strategy")
    return df, split_date


def plot_test_performance(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Annual Return"], name="Annual Return"))
    fig.add_trace(go.Bar(x=df.index, y=df["Sharpe"], name="Sharpe Ratio"))

    fig.update_layout(barmode="group", xaxis_title="Portfolio Strategy",
                      yaxis_title="Metric Value") # margin=dict(t=30),)
    return fig


def plot_test_risk(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volatility"], name="Volatility"))
    fig.add_trace(go.Bar(x=df.index, y=df["Max Drawdown"], name="Max Drawdown"))
    fig.update_layout(barmode="group",xaxis_title="Portfolio Strategy",
                      yaxis_title="Risk",margin=dict(t=40),)
    return fig


def portfolio_monthly_returns(port):
    daily_returns = port.data.returns @ port.weights
    monthly = (1 + daily_returns).resample("ME").prod() - 1  # Understnad the formula here, diff because start point chenages
    return monthly


def line_fig(returns, transform=None) -> go.Figure:
    """Build a multi-trace line chart, with an optional transform on the returns df."""
    if returns is None:
        return go.Figure()
    data = transform(returns) if transform else returns
    return go.Figure([go.Scatter(x=data.index, y=data[col], name=col) for col in data.columns])



# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------

# The control bar aesthetic shown at the top of the website
control_bar = dbc.Card(
                dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                    html.Label("Assets"),
                                    dcc.Dropdown(
                                                id="ticker-dropdown",
                                                options=[{"label": t, "value": t} for t in STOCKS],
                                                value=STOCKS[:3],
                                                multi=True,
                                                )
                                    ], width=6),

                            dbc.Col([
                                    html.Label("Date Range"),
                                    dcc.DatePickerRange(id="date-range", start_date=START_DATE, end_date=END_DATE),
                                    ], width=6),

                                ], className="mb-3"),

                        dbc.Row(id="weight-inputs-row", align="end", className="g-2 mb-2"),

                        dbc.Row([
                                dbc.Col([
                                        dbc.Button("Add Portfolio", id="add-portfolio", color="primary"),
                                        html.Div(id="weight-error", style={"color": "red"})
                                        ], width=2),
                                ]),
                            ]),
                    style={"backgroundColor": "#000000", "marginBottom": "25px", "borderRadius": "12px", "border": "1px solid #1F51FF"},
                    )




app.layout = dbc.Container(
                            [   

                            html.H1("Quant Portfolio Research Dashboard", style={"textAlign": "center"}),

                            dcc.Markdown(""" 
                                        This dashboard compares multiple portfolio construction techniques 
                                        Equal Weight, Maximum Sharpe, Minimum Variance, Risk Parity, and Momentum 
                                        using historical equity data.The goal is to evaluate how optimization methods 
                                        affect risk, returns, and investor experience.
                                        """,
                                        style={"color": "#ffffff", "padding": "10px 0px"}),
                            
                            control_bar,

                            section_header("Growth of 1 rupee (Cummulative Returns)"),

                            dcc.Markdown(""" 
                                        This plot shows the growth of a 1 rupee investment for each strategy.
                                        Differences here represent actual investor outcomes, not just statistical measures.
                                        """,
                                        style={"color": "#ffffff", "padding": "0.5px 0px"}),

                            graph_row(("Returns", "returns-graph")),
                            
                            section_header("Underwater Plot (Drawdown)"),

                            dcc.Markdown("""
                                        The underwater plot shows percentage decline from previous peak wealth.
                                        It captures investor pain and recovery time, which is not visible in average returns.
                                        """,
                                        style={"color": "#ffffff", "padding": "0.5px 0px"}),

                            graph_row(("Strategy Drawdowns","drawdown-graph")),

                            section_header("Strategy Metrics Comparison"),

                            dbc.Button("Align All Metrics", id="align-btn", color="secondary"),
                            dcc.Store(id="align-state", data=False),

                            graph_row(("Metrics Comparison", "metrics-bar")),

                            section_header("Information About Stocks"),

                            graph_row(("Stocke Returns Graph","stock-returns-graph")),

                            graph_row(("Correlation Matrix", "corr-graph"), ("Rolling Volatility","vol-graph")),

                            section_header("Theoretical Prespective"),

                            dcc.Markdown("""
                                         The efficient frontier represents the set of portfolios with maximum expected return for each level of risk under mean-variance theory.
                                        Observed portfolios are compared against theoretical optimality along with weight allocation for each stock.
                                        """,
                                        style={"color": "#ffffff", "padding": "0.5px 0px"}),

                            graph_row(("Efficient Frontier","frontier-graph"), ("Weight Allocation","weight-all-graph")),

                            section_header("Monthly Strategy Returns"),

                            selector_graph_card(selector_id="strategy-selector", graph_id="monthly-sharpe-graph"),

                            section_header("Monte-Carlo simulated returns"),

                            selector_graph_card(selector_id="monte-carlo-selector", graph_id="monte-carlo-graph"),

                            section_header("Out of Sample Backtesting"),

                            dbc.Col([
                                    html.Label("Date Range"),
                                    dcc.DatePickerRange(id="backtest-date-range", start_date=START_DATE, end_date=END_DATE)
                                    ], width=6),

                            graph_row(("Out-of-Sample Performance", "backtest-performance")),

                            graph_row(("Out-of-Sample Risk", "backtest-risk"))
                            
                            ],fluid=True,
                    )



# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

# Weight button panels
@app.callback(
    Output("weight-inputs-row", "children"),
    Input("ticker-dropdown", "value"),
)
def render_weight_inputs(tickers):
    '''
    The weight options for the stocks to make our own portfolio
    '''
    if not tickers:
        return []
    # optimal_weights = PortfoliosForTickers(tickers).get("Max Sharpe Portfolio").weights
    inputs = []
    for i,ticker in enumerate(tickers):
        weight_button = dbc.Col(
                            dbc.Input(
                                    id={"type": "weight", "index": ticker},
                                    type="number", min=0, max=1, step=0.01,
                                    value = 1/len(tickers),   
                                    # value=float(optimal_weights[i]),
                                    placeholder=ticker,
                                    ),
                                width=2
                                )
        
        inputs.append(weight_button)
    return inputs


# Saves portfolio
@app.callback(
    Output("weight-error", "children"),
    Input("add-portfolio", "n_clicks"),
    State("ticker-dropdown", "value"),
    State({"type": "weight", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def save_portfolio(_, tickers, raw_weights):
    '''
    Saves the user entered portfolio to the stock universe, 
    if the weights are invalid returns error
    '''
    weights = np.array(raw_weights)
    if not np.isclose(weights.sum(), 1, atol=1e-2):
        return "Weights must sum to 1"

    universe = get_universe(tickers)
    name = f"User Portfolio {sum(1 for k in universe.portfolios if k.startswith('User')) + 1}"  # robust
    universe.add_portfolio(weights, name)
    return ""

# Toggle button
@app.callback(
    Output("align-state", "data"),
    Output("align-btn", "children"),
    Input("align-btn", "n_clicks"),
    State("align-state", "data"),
    prevent_initial_call=True
)
def toggle_align(_, is_aligned:str) -> go.Figure:
    '''
    Changes the writeup on the toglle button and its state
    '''
    new_state = not is_aligned
    return new_state, "De-Align Metrics" if new_state else "Align All Metrics"


# Selecting Strategy option panel
@app.callback(
    Output("strategy-selector", "options"),
    Output("strategy-selector", "value"),
    Output("monte-carlo-selector", "options"),
    Output("monte-carlo-selector", "value"),
    Input("ticker-dropdown", "value"),
    Input("add-portfolio", "n_clicks"),
)
def update_strategy_options(tickers:str,_) -> go.Figure:
    '''
    Updates the options for both the monte carlo graph selector
    and the strategy monyly returns heatmap, when we add a new portfolio
    to oour portfolio universe
    '''
    universe = get_universe(tickers)
    names = universe.all_portfolios(just_names=True)
    options = [{"label": name, "value": name} for name in names]
    return options, names[0], options, names[0]  # default to first strategy


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

# Eficient fronteir graph
@app.callback(
    Output("frontier-graph", "figure"),
    Input("ticker-dropdown", "value"),
    Input("add-portfolio", "n_clicks")
)
def render_frontier(tickers:str, _):
    "Plots the eficient fronteir graph for the portfolios"
    if not tickers or len(tickers) < 2:
        return go.Figure()
    universe = get_universe(tickers)
    frontier_data = compute_frontier_data(universe.market.mu, universe.market.cov)
    portfolios = universe.all_portfolios(just_names=False)
    return plot_efficient_frontier(portfolios, frontier_data)


# Cummulative stock returns graph
@app.callback(
    Output("stock-returns-graph", "figure"),
    Input("ticker-dropdown", "value"),
    Input("add-portfolio", "n_clicks"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def render_returns(tickers,
                   _, 
                   start=START_DATE, 
                   end=END_DATE) -> go.Figure:
    """
    Plots the cummulative returns of the stocks
    """
    returns = load_filtered_returns(tickers, start, end)
    if returns is None:
        return go.Figure()

    fig = line_fig(returns,transform=lambda r: (1 + r).cumprod())

    fig.update_layout(yaxis_title="Stock Return",xaxis_title="Date",
                    hovermode="x unified", margin=dict(t=40))
    return fig


# Portfolio cummulative returns graphs
@app.callback(
    Output("returns-graph", "figure"),
    Input("ticker-dropdown", "value"),
    Input("add-portfolio", "n_clicks"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date")  
)
def cumm_render_returns(tickers:str,
                        _,
                        start:str =START_DATE,
                        end:str =END_DATE) -> go.Figure:
    """
    Plots the cummulative returns of the portfolio strategies
    """
    returns = load_filtered_returns(tickers, start, end)
    if returns is None:
        return go.Figure()
    
    universe = get_universe(tickers)
    portfolios = universe.all_portfolios(just_names=False)

    fig = go.Figure()
    for _,p in portfolios:
        port_returns = returns @ p.weights
        cumulative = (1 + port_returns).cumprod()
        fig.add_trace(
                go.Scatter(
                           x=cumulative.index, 
                           y=cumulative, 
                           name=p.name
                           )
                        )

    fig.update_layout(yaxis_title="Cumulative Return", xaxis_title="Date",
                    hovermode="x unified", margin=dict(t=40))
    return fig


# Volatility graph of stocks
@app.callback(
    Output("vol-graph", "figure"),
    Input("ticker-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def render_volatility(tickers:str, 
                      start:str =START_DATE, 
                      end:str =END_DATE) -> go.Figure:
    '''
    Plots the volatility of each stock
    '''
    figure = line_fig(
                    load_filtered_returns(tickers, start, end),
                    transform=lambda r: r.rolling(20).std() * np.sqrt(252),
                    )
    return figure


# Correlation graph
@app.callback(
    Output("corr-graph", "figure"),
    Input("ticker-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date")
)
def render_correlation(tickers:str, 
                       start:str= START_DATE, 
                       end:str= END_DATE):
    '''
    Plots the correlation graph
    '''
    returns = load_filtered_returns(tickers, start, end)
    if returns is None:
        return go.Figure()
    corr = returns.corr()
    figure = go.Figure(
                    go.Heatmap(
                            z=corr.values, 
                            x=corr.columns, 
                            y=corr.index, 
                            colorscale="RdBu_r", zmin=-1, zmax=1
                            )
                        )
    return figure


# Portfolio metrics graph (could toggle)
@app.callback(
    Output("metrics-bar", "figure"),
    Input("add-portfolio", "n_clicks"),
    Input("ticker-dropdown", "value"),
    Input("align-state", "data")
)
def plot_strategy_metrics(_,tickers:str
                          ,is_allign:str) -> go.Figure:
    '''
    Plots bar graphs for each metric (mean returns, volatility, sharpe, max dd
    var, cvar) for all portfolios
    '''
    fig = go.Figure()
    universe = get_universe(tickers)
    df = universe.compare_metric("all")
    
    df = df.T if is_allign else df
    for metric in df.columns:
        fig.add_trace(
                go.Bar(
                        x=df.index, 
                        y=df[metric], 
                        name=metric
                      )
                    )

    fig.update_layout(barmode="group",yaxis_title="Metric Value",margin=dict(t=40))
    return fig


# Stacked weight bar graph
@app.callback(
    Output("weight-all-graph", "figure"),
    Input("add-portfolio", "n_clicks"),
    Input("ticker-dropdown", "value")
)
def plot_portfolio_weights(_,tickers):
    '''
    Plots a stacked bar chart of the weight allocation for all portfolios
    '''
    universe = get_universe(tickers)
    df = pd.DataFrame(index=tickers)
    portfolios = universe.all_portfolios(just_names=False)

    for name,port in portfolios:
        name = name.replace("Portfolio","")
        df[name] = port.weights

    df = df.T
    fig = go.Figure()
    for portfolio_name in df.columns:
        fig.add_trace(
                go.Bar(
                        x=df.index, 
                        y=df[portfolio_name], 
                        name=portfolio_name
                      )
                    )

    fig.update_layout(barmode="stack", xaxis_title="Strategy", yaxis_title="Portfolio Weight",
                    yaxis=dict(tickformat=".0%"), margin=dict(t=40))
    return fig


# Drawdown (Underwater) plot
@app.callback(
    Output("drawdown-graph", "figure"),
    Input("ticker-dropdown", "value"),
    Input("add-portfolio", "n_clicks"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date")
)
def render_drawdown(tickers:str,_, 
                    start:str=START_DATE, 
                    end:str=END_DATE) -> go.Figure:
    '''
    Plots the drawdon of all the portfolios for a particular
    portfolio universe.
    '''
    returns = load_filtered_returns(tickers, start, end)
    if returns is None:
        return go.Figure()

    universe = get_universe(tickers)
    portfolios = universe.all_portfolios(just_names=False)

    fig = go.Figure()
    for name,p in portfolios:
        port_returns = returns @ p.weights
        cumulative = (1 + port_returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        fig.add_trace(
                go.Scatter(
                            x=drawdown.index, 
                            y=drawdown, name=name, 
                            fill="tozeroy", opacity=0.4
                            )
                        )

    fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown",
        yaxis=dict(tickformat=".0%"), hovermode="x unified",
        margin=dict(t=40))
    
    return fig


# Monthly returns heatmap
@app.callback(
    Output("monthly-sharpe-graph", "figure"),
    Input("strategy-selector", "value"),
    Input("ticker-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date")
)
def render_monthly_sharpe(selected_strategy:str, 
                          tickers:str, 
                          start_date:str= START_DATE, 
                          end_date:str= END_DATE) -> go.Figure:
    '''
    Plots the monthly returns heatmap, for a given stock universe
    '''
    universe = get_universe(tickers,start_date,end_date)
    selected_strategy = universe.get(selected_strategy)
    monthly = portfolio_monthly_returns(selected_strategy)

    df = monthly.to_frame("ret")
    df["Year"] = df.index.year
    df["Month"] = df.index.month

    pivot = df.pivot(index="Year", columns="Month", values="ret")

    fig = go.Figure(
                go.Heatmap(
                        z=pivot.values,
                        x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                        y=pivot.index,
                        colorscale="RdYlGn",
                        zmid=0
                        )
                    )

    fig.update_layout(xaxis_title="Month",yaxis_title="Year",
                      margin=dict(t=40))

    return fig


# Monte Carlo returns graph
@app.callback(
    Output("monte-carlo-graph", "figure"),
    Input("monte-carlo-selector", "value"),
    Input("ticker-dropdown", "value")
)
def render_monte_carlo(selected_strategy:str, 
                       tickers:str) -> go.Figure:
    '''
    Plots returns histogram obtained from monte carlo simulation
    For selected ticker and startegy
    '''
    if not selected_strategy or not tickers:
        return go.Figure()

    universe = get_universe(tickers)
    p = universe.get(selected_strategy)
    results = p.simulate_paths()  # returns array of final values

    fig = go.Figure(
            go.Histogram(
                        x=results,
                        nbinsx=100,
                        marker_color="#1F51FF",
                        opacity=0.75,
                        name=selected_strategy,
                        )
                    )

    var, cvar = p.var_cvar

    fig.add_vline(x=var, line=dict(color="orange", dash="dash"), annotation_text="VaR")
    fig.add_vline(x=cvar, line=dict(color="red", dash="dash"), annotation_text="CVaR")
    fig.update_layout(xaxis_title="Final Portfolio Value", yaxis_title="Frequency",
                      margin=dict(t=40))
    return fig


# Backtesting Graphs
@app.callback(
    Output("backtest-performance", "figure"),
    Output("backtest-risk", "figure"),
    Input("ticker-dropdown", "value"),
    Input("add-portfolio", "n_clicks"),
    Input("backtest-date-range", "start_date"),
    Input("backtest-date-range", "end_date") 
)
def update_backtest(tickers:str,
                    _, 
                    start:str =START_DATE, 
                    end:str =END_DATE):
    '''
    splits the backtesting time frame to train/test split and
    plots the return centred metrics graph and risk centered metrics graph
    '''
    if not tickers or len(tickers) < 2:
        return go.Figure(), go.Figure()
    df, split_date = backtesting(tickers, start, end)
    perf_fig = plot_test_performance(df)
    risk_fig = plot_test_risk(df)
    perf_fig.add_annotation(
                            text=f"Training/Test split: {split_date.date()}",
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=1.15,
                            showarrow=False
                            )
    return perf_fig, risk_fig



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
    # app.run()