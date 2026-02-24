import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scr.logger import setup_logger
from scr.config import NUM_TRADING_DAYS,RISK_FREE_RATE,START_DATE,END_DATE
from scr.optimization_methods import momentum,max_sharpe,min_variance,equal_weight,risk_parity

logger = setup_logger()

RETURNS_PATH = "data/processed/returns.parquet"

def load_returns() -> pd.DataFrame:
	"""
	Reads the returns.parquet file, converts it into a 
	dataframe and returns it
	"""
	try:
		returns = pd.read_parquet(RETURNS_PATH)
		logger.info("Returns loaded for portfolio engine")
		if returns.shape[0] <= returns.shape[1]:
			raise ValueError(
				f"Not enough observations (T={returns.shape[0]}, N={returns.shape[1]})"
			)
		return returns
	except FileNotFoundError:
		logger.exception("Returns file missing")
		raise
	except Exception as e:
		logger.exception(f"The file did not load | Reason : {e}")
		raise

def validate_cov(cov: pd.DataFrame):
	"""
	Checks wether or not the covariance matrix of the stocks
	is valid or not.
	"""
	eigvals = np.linalg.eigvalsh(cov)
	if np.any(eigvals < -1e-8):
		raise ValueError("Covariance matrix is not positive semi-definite")

def validate_weights(weights: np.ndarray):
	"""
	Validates the weights 
		- Should sum to 1
		- Should be an array
		- No elements should be Nan
		- It should be 1 Dimensional
	"""
	if not isinstance(weights, np.ndarray):
		raise TypeError("weights must be numpy array")
	if weights.ndim != 1:
		raise ValueError("weights must be 1D")
	if np.any(np.isnan(weights)):
		raise ValueError("weights contain NaN")
	if not np.isclose(weights.sum(), 1.0, atol=1e-2):
		raise ValueError("weights must sum to 1")


# Market Data
class MarketData:
	"""
	Represents market data (returns, mean returns, covariance) 
	for a particular set of tickers.
	Attributes:
		tickers (list[str]) -> The tickers for who the market data was made 
		returns (float) -> The dataset containing market returns
		mu (pd.Series) -> The mean returns of each stocks
		cov (pd.dataframe) -> The correlation matix between stocks
	"""
	def __init__(self, tickers: list[str], start_date=START_DATE, end_date=END_DATE):
		self.tickers = tickers
		self.returns = self._load_returns().loc[start_date:end_date]
		self.mu, self.cov = self._estimate_statistics()
		validate_cov(self.cov)

	def _load_returns(self) -> pd.DataFrame:
		"""
		Loads the returns from our returns.parquet file
		and filters based on tickers.
		"""
		returns = load_returns()
		returns = returns[self.tickers]
		return returns

	def _estimate_statistics(self):
		"""
		Estimates the annualized statistics of market returns 
		data of each stock
		"""
		mu = self.returns.mean() * NUM_TRADING_DAYS
		cov = self.returns.cov() * NUM_TRADING_DAYS
		return mu, cov



# PORTFOLIO OBJECT 
class Portfolio:
	"""
	A financial object representing a tradable portfolio with its own weights and metrics
	for a given market data.
	Attributes:
		- data (MarketData) -> An instance of the Market data class we made our portfolio for
		- weights (np.ndarray) -> what percentage of capital is allocated to each stock.
		- name (str) -> The name of the portfolio.
		- expected_returns (float) -> The expected returns of the portfolio.
		- volatility (float) -> The risk associated with the portfolio.
		- sharpe (float) -> The ratio of expected_return to volatility.
		- max_drawdown (float) -> The difference between the max returns and the lowest return.
		- var (float) -> The probaility of the returns being less than this value is only 5% (in general).
		- cvar (float) -> The average value of returns we could expect during time periods of very low returns.
	"""
	def __init__(self, market_data: MarketData, weights: np.ndarray, name="Portfolio"):
		validate_weights(weights)
		if len(weights) != len(market_data.tickers):
			raise ValueError("weights do not match number of assets")
		self.data = market_data
		self.weights = np.array(weights)
		self.name = name

	@property
	def expected_return(self):
		return float(np.dot(self.weights, self.data.mu))
	@property
	def volatility(self):
		return float(np.sqrt(self.weights.T @ self.data.cov @ self.weights))
	@property
	def sharpe(self, rf=RISK_FREE_RATE):
		return (self.expected_return - rf) / self.volatility
	@property
	def max_drawdown(self):
		port_returns = self.data.returns @ self.weights.T
		cumulative = (1 + port_returns).cumprod()
		peak = cumulative.cummax()
		drawdown = (cumulative - peak) / peak
		max_dd = drawdown.min()
		return max_dd
	@property
	def var_cvar(self, confidence=0.95):
		sims = self.simulate_paths()
		percentile = 100 * (1 - confidence)
		var = np.percentile(sims, percentile)
		cvar = sims[sims <= var].mean()
		return float(var), float(cvar)
	
	def simulate_paths(self, days:int=252, sims:int=3000):
		"""
		Uses monte carlo simulation to obtain different trajectories the portfolio
		may follow to calculate theoretical var and cvar.
		Parameters:
			- days : The number of days of returns for each simulation
			- sims : The number of simulated paths you want to create
		"""
		port_mu = np.dot(self.weights, self.data.mu)
		port_vol = np.sqrt(self.weights.T @ self.data.cov @ self.weights)
		results = []
		for _ in range(sims):
			simulated_returns = np.random.normal(
				port_mu / NUM_TRADING_DAYS,
				port_vol / np.sqrt(NUM_TRADING_DAYS),
				days,
			)
			path = (1 + simulated_returns).cumprod()
			results.append(path[-1])
		results = np.array(results)
		if not np.all(np.isfinite(results)):
			raise RuntimeError("Monte Carlo produced invalid values")
		return results
	

	
class PortfoliosForTickers:
	"""
	Represents a trading universe for a specific set of tickers.
	Stores multiple portfolios built on the same market data.
	 	- Equal Weight Portfolio
		- Max Sharpe Portfolio
		- Minimum Variance Portfolio
		- Risk Parity Portfolio
		- Momentum Portfolio

	Attributes:
		- tickers (list[str]) -> Tickers for the portfolio universe
		- market (MarketData) -> An instance of the Market data class we made our portfolios for
		- portfolios (dict[str, Portfolio]) -> Dictionary containing the portfolios and theyre associated classes
		 
	"""
	def __init__(self, tickers: list[str],start_date=START_DATE,end_date=END_DATE):
		self.tickers = tuple(tickers)
		self.timeline = (start_date,end_date)
		self.market = MarketData(tickers, start_date, end_date)
		self.portfolios: dict[str, Portfolio] = {}
		# --- Strategy portfolios ---
		self.add_portfolio(equal_weight(self.market), "Equal Weight Portfolio")
		self.add_portfolio(max_sharpe(self.market, RISK_FREE_RATE), "Max Sharpe Portfolio")
		self.add_portfolio(min_variance(self.market), "Minimum Variance Portfolio")
		self.add_portfolio(risk_parity(self.market), "Risk Parity Portfolio")
		self.add_portfolio(momentum(self.market), "Momentum Portfolio")

	def add_portfolio(self, weights: np.ndarray, name: str):
		port = Portfolio(self.market, weights, name)
		self.portfolios[name] = port

	def get(self, name: str) -> Portfolio:
		return self.portfolios[name]

	def all_portfolios(self,just_names=True):
		if just_names:
			return list(self.portfolios.keys()) 
		else:
			return list(self.portfolios.items())
		
	def compare_metric(self, metric: str):
		rows = []
		for name,port in self.portfolios.items():
			if metric == 'all':
				var, cvar = port.var_cvar
				rows.append({
					"Strategy": name,
					"Expected Return": port.expected_return,
					"Volatility": port.volatility,
					"Sharpe Ratio": port.sharpe,
					"Max Drawdown": port.max_drawdown,
					"VaR (95%)": var,
					"CVaR (95%)": cvar})
			else:
				rows.append({"Strategy":name,f"{metric} values":getattr(port, metric)})

		df = pd.DataFrame(rows).set_index("Strategy")
		return df
