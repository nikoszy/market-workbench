import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, kurtosis
from scipy.optimize import minimize
from datetime import datetime
from .data_fetcher import get_stock_data, get_risk_free_rate


def import_portfolio(tickers, start='2021-01-01', end=None):
    '''Gets historical closing price data for a list of tickers, 
    calculates log returns, and returns a DataFrame of log returns for a specific portfolio.
    '''
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start, end=end)['Close']
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)
    returns = np.log(data / data.shift(1)).dropna()
    return returns

def plot_portfolio(returns):
    '''Plots historical close price index and log returns for a portfolio of stocks.
    '''
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    data = np.exp(returns.cumsum()) # Convert log returns back to price index
    data.plot(ax=axes[0])
    axes[0].set_title('Historical Close Prices')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price Index')
    axes[0].legend(returns.columns)
    
    returns.plot(ax=axes[1])
    axes[1].set_title('Log Returns')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Log Return')
    axes[1].legend(returns.columns)
    
    plt.tight_layout()
    return plt.gcf()


def plot_correlation(returns):
    '''Plots the correlation matrix of log returns as a heatmap.
    '''
    corr = returns.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title('Correlation Matrix of Log Returns')
    plt.grid(False)
    return plt.gcf()

def plot_covariance(returns):
    '''Plots the covariance matrix of log returns as a heatmap.
    '''
    cov = returns.cov()
    plt.figure(figsize=(10, 8))
    plt.imshow(cov, cmap='viridis')
    plt.colorbar(label='Covariance')
    plt.xticks(range(len(cov)), cov.columns, rotation=45)
    plt.yticks(range(len(cov)), cov.columns)
    plt.title('Covariance Matrix of Log Returns')
    plt.grid(False)
    return plt.gcf()

# Screen for risk before optimization 

def screen_stocks(log_returns, risk_free_rate):
    '''Screens stocks based on annualized return, volatility, Sharpe ratio, and max drawdown.
    Flags stocks with negative Sharpe, severe drawdown (>40%), or high volatility (>1.5x mean).
    '''
    annualized_returns = log_returns.mean() * 252
    annualized_volatility = log_returns.std() * np.sqrt(252)
    sharpe_ratios = (annualized_returns - risk_free_rate) / annualized_volatility
    
    max_drawdown = {}
    for col in log_returns.columns:
        cumulative = (1 + log_returns[col]).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown[col] = drawdown.min()

    screened = pd.DataFrame({
        'Annualized Return': annualized_returns,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratios,
        'Max Drawdown': pd.Series(max_drawdown)
    })

    screened['Flag'] = 'Pass'
    screened.loc[screened['Sharpe Ratio'] < 0, 'Flag'] = 'Negative Sharpe'
    median_dd = screened['Max Drawdown'].median()
    screened.loc[screened['Max Drawdown'] < median_dd * 1.5, 'Flag'] = 'Severe drawdown'
    screened.loc[screened['Annualized Volatility'] > annualized_volatility.mean() * 1.5, 'Flag'] = 'High volatility'
    
    return screened

#Build the mean-variance optimizer using the screened stocks and return the optimized portfolio log returns

def optimize_portfolio(log_returns, risk_free_rate, profile = 'Balanced'):
    '''
    Optimizes a portfolio using mean-variance optimization based on the Sharpe ratio,
    with 3 portfolio profiles with their own default optimized weight allocations.

    Args:
        log_returns (pd.DataFrame): DataFrame of log returns for the selected stocks.
        risk_free_rate (float): Daily risk-free rate (in decimal form).
        profile (str): Risk profile for optimization ('Conservative', 'Balanced', 'Aggressive').
    Returns:
        tuple: (optimized portfolio log returns, Sharpe ratio, optimized weights)
        '''
    profiles = {
            'Conservative': (0.1, 0.25),
            'Balanced': (0.05, 0.3),
            'Aggressive': (0, 0.4)
        }
    min_weight, max_weight = profiles[profile]

    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252
    num_assets = len(log_returns.columns)

    def negative_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    initial_guess = np.array(num_assets * [1. / num_assets])
    result = minimize(negative_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_weights = result.x
    optimized_portfolio_return = log_returns.dot(optimized_weights)    
    sharpe = -result.fun
    return optimized_portfolio_return, sharpe, optimized_weights

#Show optimized returns vs equally weighted returns 
def plot_optimized_portfolio(log_returns, optimized_returns):
    '''Plots cumulative returns of the optimized portfolio against an equally weighted portfolio.'''
    equal_weights = np.ones(len(log_returns.columns)) / len(log_returns.columns)
    equal_weighted_returns = log_returns.dot(equal_weights)
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.exp(optimized_returns.cumsum()), label='Optimized Portfolio')
    plt.plot(np.exp(equal_weighted_returns.cumsum()), label='Equally Weighted Portfolio', linestyle='--')
    plt.title('Cumulative Returns: Optimized vs Equally Weighted')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    return plt.gcf()

def backtest_portfolio(tickers, risk_free_rate, profile='Balanced'):
    '''Backtests the optimized portfolio by training on historical data from 2019-2023 and evaluating on a holdout set from 2024-2025.
    Args:
        tickers (list): List of stock ticker symbols to include in the portfolio.
        risk_free_rate (float): Daily risk-free rate (in decimal form) to use for Sharpe ratio calculation.
        profile (str): Risk profile for optimization ('Conservative', 'Balanced', 'Aggressive').
    Returns:
        dict: A dictionary containing in-sample Sharpe ratio, out-of-sample Sharpe ratio, overfit flag, optimized weights, and selected tickers.
    '''
    train_returns = import_portfolio(tickers, '2019-01-01', '2023-12-31')
    test_returns = import_portfolio(tickers, '2024-01-01', '2025-12-31')
    
    screened = screen_stocks(train_returns, risk_free_rate)
    selected = screened[screened['Flag'] == 'Pass'].index.tolist()
    
    _, sharpe_train, weights = optimize_portfolio(train_returns[selected], risk_free_rate, profile)
    
    # Evaluate SAME weights on test data
    test_mean = test_returns[selected].mean() * 252
    test_cov = test_returns[selected].cov() * 252
    test_return = np.dot(weights, test_mean)
    test_vol = np.sqrt(np.dot(weights, np.dot(test_cov, weights)))
    sharpe_test = (test_return - risk_free_rate) / test_vol
    
    return {
        'Sharpe In-Sample': sharpe_train,
        'Sharpe Out-of-Sample': sharpe_test,
        'Overfit Flag': sharpe_train > 2.0,
        'Optimized Weights': dict(zip(selected, weights)),
        'Selected Tickers': selected
    }

#Value at Risk:
def parametric_var(weights, daily_mean, daily_cov, confidence_level=0.95):
    '''Calculates the parametric Value at Risk (VaR) for a portfolio based on the mean and 
    covariance of daily log returns, and a specified confidence level.
    '''
    portfolio_mean = np.dot(weights, daily_mean)
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(daily_cov, weights)))
    z_score = norm.ppf(1 - confidence_level)
    return -(portfolio_mean + z_score * portfolio_vol)

def historical_var(weights, returns, confidence_level=0.95):
    '''Calculates the historical Value at Risk (VaR) for a portfolio based on the distribution of
    historical portfolio returns and a specified confidence level.
    '''
    portfolio_returns = returns.dot(weights)
    return -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

def calculate_var(log_returns, weights, confidence_level=0.95):
    '''Calculates both parametric and historical Value at Risk (VaR) for a portfolio, as well as the excess kurtosis of the portfolio returns, based on the log returns of the assets and their weights in the portfolio.
    Args:
        log_returns (pd.DataFrame): DataFrame of log returns for each asset in the portfolio.
        weights (np.array): Array of portfolio weights for each asset.
        confidence_level (float): Confidence level for VaR calculation (default is 0.95).
    Returns:
        dict: A dictionary containing the calculated parametric VaR, historical VaR, excess kurtosis of the portfolio returns, and the confidence level used for the calculations.
    '''
    daily_mean = log_returns.mean()
    daily_cov = log_returns.cov()
    
    param_var = parametric_var(weights, daily_mean, daily_cov, confidence_level)
    hist_var = historical_var(weights, log_returns.dropna(), confidence_level)
    
    excess_kurtosis = kurtosis(log_returns.dropna().dot(weights), fisher=True)
    
    return {
        'Parametric VaR': param_var,
        'Historical VaR': hist_var,
        'Excess Kurtosis': excess_kurtosis,
        'Confidence Level': confidence_level
    }

def plot_var_comparison(log_returns, weights):
    '''Plots the distribution of portfolio returns along with the parametric and historical Value at Risk (VaR) thresholds, and overlays a normal distribution curve for comparison.
    '''
    var_results = calculate_var(log_returns, weights)
    portfolio_returns = log_returns.dot(weights).dropna()
    plt.figure(figsize=(10, 6))
    plt.hist(portfolio_returns, bins=50, density = True, alpha=0.6, color='g', label='Portfolio Returns')
    x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
    pdf = norm.pdf(x, portfolio_returns.mean(), portfolio_returns.std())
    plt.plot(x, pdf, color='blue', label='Normal PDF')
    plt.axvline(-var_results['Parametric VaR'], color='red', linestyle='--', label='Parametric VaR')
    plt.axvline(-var_results['Historical VaR'], color='orange', linestyle='--', label='Historical VaR')
    plt.title('Portfolio Returns Distribution with VaR')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    return plt.gcf()

#bootstrap the optimization to get confidence intervals on the optimized weights and returns

def boot(returns, risk_free_rate, profile='Balanced', num_portfolios=1000):
    '''Performs bootstrapping of the portfolio optimization process to generate a distribution of optimized weights,
    returns, and volatilities, allowing us to estimate a confidence interval for the optimize portfolio. 
    Args:
        returns (pd.DataFrame): DataFrame of log returns for the selected stocks.
        risk_free_rate (float): Daily risk-free rate (in decimal form).
        profile (str): Risk profile for optimization ('Conservative', 'Balanced', 'Aggressive').
        num_portfolios (int): Number of bootstrap samples to generate (default is 1000).
    Returns:
        tuple: (bootstrap_weights, bootstrap_returns, bootstrap_vols)
        - bootstrap_weights: A 2D numpy array of shape (num_portfolios, num_assets) containing the optimized weights for each bootstrap sample.
        - bootstrap_returns: A 1D numpy array of shape (num_portfolios,) containing the expected returns of the optimized portfolio for each bootstrap sample.
        - bootstrap_vols: A 1D numpy array of shape (num_portfolios,) containing the volatilities of the optimized portfolio for each bootstrap sample.
    '''
    profiles = {
        'Conservative': (0.10, 0.25),
        'Balanced': (0.05, 0.30),
        'Aggressive': (0.00, 0.40)
    }
    min_weight, max_weight = profiles[profile]
    num_assets = returns.shape[1]
    initial_weights = np.array([1/num_assets] * num_assets)
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    def negative_sharpe(weights, mean_ret, cov):
        '''Minimize negative Sharpe ratio for the portfolio optimization process, given the mean returns and covariance matrix of the assets.
        Args:
            weights (np.array): Array of portfolio weights for each asset.
            mean_ret (pd.Series): Series of mean annualized returns for each asset.
            cov (pd.DataFrame): Covariance matrix of annualized returns for the assets.
        Returns:
            float: The negative Sharpe ratio of the portfolio for the given weights, mean returns, and covariance matrix.
        '''
        ret = np.dot(weights, mean_ret)
        vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        return -(ret - risk_free_rate) / vol
    
    bootstrap_weights = []
    bootstrap_returns = []
    bootstrap_vols = []
    for _ in range(num_portfolios):
        boot_returns = returns.sample(frac=1, replace=True)
        boot_mean = boot_returns.mean() * 252
        boot_cov = boot_returns.cov() * 252
        result = minimize(negative_sharpe, initial_weights,
                          args=(boot_mean, boot_cov),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x
        bootstrap_weights.append(weights)
        bootstrap_returns.append(np.dot(weights, boot_mean))
        bootstrap_vols.append(np.sqrt(np.dot(weights, np.dot(boot_cov, weights))))
    return np.array(bootstrap_weights), np.array(bootstrap_returns), np.array(bootstrap_vols)
#Plot bootstrapped efficient frontier w sharpe ratio
def plot_bootstrap(bootstrap_returns, bootstrap_vols, risk_free_rate):
    '''Plots the bootstrapped efficient frontier by creating a scatter plot of the bootstrapped portfolio returns against their corresponding volatilities, with points colored according to their Sharpe ratios, allowing for visual analysis of the distribution of optimized portfolios under the specified risk profile.
    '''
    sharpe_ratios = (bootstrap_returns - risk_free_rate) / bootstrap_vols
    plt.figure(figsize=(10, 6))
    plt.scatter(bootstrap_vols, bootstrap_returns, c=sharpe_ratios, cmap='viridis', marker='o', alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.title('Bootstrapped Efficient Frontier with Sharpe Ratios')
    plt.grid()
    return plt.gcf()

def get_portfolio_summary(log_returns, risk_free_rate, profile = 'Balanced'):
    '''Summarizes the portfolio optimization process by returning optimized weights, 
    Sharpe ratio, annualized return, annualized volatility, and VaR metrics for the 
    selected stocks in the portfolio. 
    '''
    screening = screen_stocks(log_returns, risk_free_rate)
    selected = screening[screening['Flag'] == 'Pass'].index.tolist()
    filtered = log_returns[selected]

    optimized_returns, sharpe, weights = optimize_portfolio(filtered, risk_free_rate, profile)
    var_results = calculate_var(filtered, weights)

    mean_return = filtered.mean() * 252
    cov_matrix = filtered.cov() * 252
    port_return = np.dot(weights, mean_return)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

    return {
        'screening': screening,
        'qualified_tickers': selected,
        'weights': dict(zip(selected, weights)),
        'sharpe_ratio': sharpe,
        'annualized_return': port_return,
        'annualized_volatility': port_vol,
        'var': var_results,
        'profile': profile
    }