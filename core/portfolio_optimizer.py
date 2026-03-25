import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, kurtosis
from scipy.optimize import minimize

def import_portfolio(tickers, start='2021-01-01', end='2026-01-01'):
    data = yf.download(tickers, start=start, end=end)['Close']
    returns = np.log(data / data.shift(1)).dropna()
    return returns

#Plot historical close prices for each ticker and log returns for each ticker 
def plot_portfolio(returns):
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

#Compute correlation matrix of log returns and plot as heatmap

def plot_correlation(returns):
    corr = returns.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title('Correlation Matrix of Log Returns')
    plt.grid(False)
    return plt.gcf()

#Compute covariance matrix of log returns and plot as heatmap

def plot_covariance(returns):
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
    screened.loc[screened['Max Drawdown'] < -0.40, 'Flag'] = 'Severe drawdown'
    screened.loc[screened['Annualized Volatility'] > annualized_volatility.mean() * 1.5, 'Flag'] = 'High volatility'
    return screened

#Build the mean-variance optimizer using the screened stocks and return the optimized portfolio log returns

def optimize_portfolio(log_returns, risk_free_rate, profile = 'Balanced'):
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
        optimized_portfolio_return = np.dot(log_returns, optimized_weights)
        sharpe = -result.fun
        return optimized_portfolio_return, sharpe, optimized_weights

#Show optimized returns vs equally weighted returns 
def plot_optimized_portfolio(log_returns, optimized_returns):
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

#Train/test split:
#Optimize on 2019–2023 data
#Evaluate on 2024–2025 holdout
#Report in-sample vs out-of-sample Sharpe ratio
#RED FLAG CHECK: if backtest Sharpe > 2.0, it's overfit. Document this.

def backtest_portfolio(tickers, risk_free_rate, profile='Balanced'):
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
    
    print(f'In-sample Sharpe: {sharpe_train:.2f}')
    print(f'Out-of-sample Sharpe: {sharpe_test:.2f}')
    if sharpe_train > 2.0:
        print('RED FLAG: potential overfitting')
    
    return sharpe_train, sharpe_test, weights

#Value at Risk:
def parametric_var(weights, daily_mean, daily_cov, confidence_level=0.95):
    portfolio_mean = np.dot(weights, daily_mean)
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(daily_cov, weights)))
    z_score = norm.ppf(1 - confidence_level)
    return -(portfolio_mean + z_score * portfolio_vol)

def historical_var(weights, returns, confidence_level=0.95):
    portfolio_returns = returns.dot(weights)
    return -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

def calculate_var(log_returns, weights, confidence_level=0.95):
    daily_mean = log_returns.mean()
    daily_cov = log_returns.cov()
    
    param_var = parametric_var(weights, daily_mean, daily_cov, confidence_level)
    hist_var = historical_var(weights, log_returns.dropna(), confidence_level)
    
    excess_kurtosis = kurtosis(log_returns.dropna().dot(weights), fisher=True)
    
    print(f"Parametric VaR at {confidence_level*100:.0f}% confidence: {param_var:.6f}")
    print(f"Historical VaR at {confidence_level*100:.0f}% confidence: {hist_var:.6f}")
    print(f"Excess Kurtosis of Portfolio Returns: {excess_kurtosis:.4f}")
    
    plt.figure(figsize=(10, 6))
    portfolio_returns = log_returns.dropna().dot(weights)
    plt.hist(portfolio_returns, bins=50, density=True, alpha=0.6, color='g', label='Portfolio Returns')
    
    x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
    pdf = norm.pdf(x, portfolio_returns.mean(), portfolio_returns.std())
    plt.plot(x, pdf, 'r--', label='Normal Distribution')
    
    plt.title('Distribution of Portfolio Returns')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.show()

#bootstrap the optimization to get confidence intervals on the optimized weights and returns

def boot(returns, risk_free_rate, profile='Balanced', num_portfolios=1000):
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
    sharpe_ratios = (bootstrap_returns - risk_free_rate) / bootstrap_vols
    plt.figure(figsize=(10, 6))
    plt.scatter(bootstrap_vols, bootstrap_returns, c=sharpe_ratios, cmap='viridis', marker='o', alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.title('Bootstrapped Efficient Frontier with Sharpe Ratios')
    plt.grid()
    plt.show()
    