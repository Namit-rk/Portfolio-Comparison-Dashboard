import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

RISK_FREE_RATE = 0.05

def portfolio_performance(weights:list[float], 
                          mu:pd.Series, 
                          cov:pd.DataFrame,
                          rf:float =0) -> tuple[float,float,float]:
    """
    Returns (return, volatility, sharpe) based on weights
    """
    ret = np.dot(weights, mu)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe

def random_portfolio_metrics(mu, cov, rf = RISK_FREE_RATE, n_portfolios=5000):
	n_assets = len(mu)
	weights_list = np.random.dirichlet(np.ones(n_assets), size=n_portfolios)

	rets = []
	vols = []
	sharpes = []

	for w in weights_list:
		r, v, _ = portfolio_performance(w,mu, cov, rf)
		rets.append(r)
		vols.append(v)
		sharpes.append((r - rf) / v)

	rets = np.array(rets)
	vols = np.array(vols)
	sharpes = np.array(sharpes)

	return rets, vols, sharpes

def frontier_volatalities(mu,cov):
	n_assets = len(mu)
	target_returns = np.linspace(mu.min(), mu.max(), 60)
	frontier_vols = []

	bounds = tuple((0,1) for _ in range(n_assets))
	init = np.ones(n_assets)/n_assets

	for target in target_returns:

		constraints = (
			{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
			{'type': 'eq', 'fun': lambda w: w @ mu - target}
		)

		result = minimize(
			lambda w: w @ cov @ w,
			init,
			method='SLSQP',
			bounds=bounds,
			constraints=constraints
		)

		frontier_vols.append(np.sqrt(result.fun))

	frontier_vols = np.array(frontier_vols)
	return frontier_vols,target_returns

def compute_frontier_data(mu, cov):
    rets, vols, sharpes = random_portfolio_metrics(mu, cov)
    frontier_vols, target_returns = frontier_volatalities(mu, cov)

    return {
        "cloud_vols": vols,
        "cloud_rets": rets,
        "cloud_sharpes": sharpes,
        "frontier_vols": frontier_vols,
        "frontier_rets": target_returns
    }


def plot_efficient_frontier(portfolios,frontier_data):
	fig = go.Figure()

	vols = frontier_data["cloud_vols"]
	rets = frontier_data["cloud_rets"]
	sharpes = frontier_data["cloud_sharpes"]
	frontier_vols = frontier_data["frontier_vols"]
	target_returns = frontier_data["frontier_rets"]

	# Random portfolios cloud
	fig.add_trace(go.Scatter(
		x=vols,
		y=rets,
		mode='markers',
		marker=dict(
			size=2,
			color=sharpes,
			colorscale='Viridis',
			showscale=True,
			colorbar=dict(title="Sharpe Ratio"),
			opacity=0.5
		),
		name="Random Portfolios",
		showlegend=False,
		hovertemplate=None,
        hoverinfo='none'
	))

	# Efficient frontier curve
	fig.add_trace(go.Scatter(
		x=frontier_vols,
		y=target_returns,
		mode='lines',
		line=dict(color='orange', width=3),
		name="Efficient Frontier",
		hovertemplate=
			"Volatility: %{x:.2%}<br>" +
			"Return: %{y:.2%}<extra></extra>"
	))
		
	for name,p in portfolios:
		ret = p.expected_return
		vol = p.volatility
		sharpe = p.sharpe

		if p.name == "Optimal Portfolio":
			marker = dict(size=18,color='red',symbol='star')
		else:
			marker = dict(size=16,symbol='triangle-down')

		fig.add_trace(go.Scatter(
			x=[vol],
			y=[ret],
			customdata=[[sharpe]],
			mode="markers",
			marker=marker,
			name=p.name,
			hovertemplate=
				"<b>%{fullData.name}</b><br>" +
				"Sharpe: %{customdata[0]:.2f}<br>" +
				"Volatility: %{x:.2%}<br>" +
				"Return: %{y:.2%}<extra></extra>"
		))
		
	fig.update_layout(
		# title="Markowitz Efficient Frontier",
		xaxis_title="Annualized Volatility (Risk)",
		yaxis_title="Annualized Expected Return",
		showlegend=False,
		margin=dict(t=40)
	)
	return fig
	# fig.show(renderer="browser")
	
