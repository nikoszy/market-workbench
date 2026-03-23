from .data_fetcher import get_merged_data
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import numpy as np

def run_adf(df):
    stock_adf = ts.adfuller(df['Excess_Returns'].dropna())
    spy_adf = ts.adfuller(df['Excess_Returns_SPY'].dropna())
    return stock_adf, spy_adf

def run_beta_regression(df):
    X = sm.add_constant(df['Excess_Returns_SPY'].dropna())
    y = df['Excess_Returns'].dropna()
    X,y = X.align(y, join='inner', axis=0)
    model = sm.OLS(y, X).fit()
    # Use newey-west standard errors to account for heteroskedasticity and autocorrelation
    model_nw = model.get_robustcov_results(cov_type='HAC', maxlags=5)
    return model, model_nw

def run_rolling_beta(df, window = 60):
    rolling = RollingOLS(df['Excess_Returns'], sm.add_constant(df['Excess_Returns_SPY']), window=window)
    results = rolling.fit(cov_type='HAC', maxlags=5)
    results = results.params.rename(columns={'Excess_Returns_SPY': 'Beta'})
    return results['Beta'].dropna()

def plot_residual_diagnostics(df):
    model, model_nw = run_beta_regression(df)
    fig = sm.graphics.plot_regress_exog(model, 'Excess_Returns_SPY', fig=plt.figure(figsize=(12, 8)))
    return fig

# Plot rolling beta vs quarterly returns 
def plot_quarterly_scatter(df, results):
    quarterly = df['Log_Returns'].resample('Q').sum()
    quarterly_beta = results.resample('Q').last()
    X_quarterly = sm.add_constant(quarterly_beta)
    y_quarterly = quarterly.reindex(X_quarterly.index).dropna()
    X_quarterly = X_quarterly.reindex(y_quarterly.index)
    model_quarterly = sm.OLS(y_quarterly, X_quarterly).fit()
    plt.figure(figsize=(10, 6))
    plt.scatter(quarterly_beta, quarterly, label='Quarterly Returns vs Beta')
    plt.plot(quarterly_beta, model_quarterly.predict(X_quarterly), color='red', label='Fit Line')
    plt.xlabel('Rolling Beta (Quarterly)')
    plt.ylabel('Quarterly Log Returns')
    plt.title('Quarterly Log Returns vs Rolling Beta')
    plt.legend()
    plt.grid()
    return plt.gcf()
def plot_realized_volatility(df):
    ARV = df['Log_Returns'].rolling(window=60).std() * np.sqrt(252)
    plt.figure(figsize=(10, 6))
    plt.plot(ARV, label='Realized Volatility (60-day)')
    plt.xlabel('Date')
    plt.ylabel('Annualized Realized Volatility')
    plt.title('Realized Volatility Over Time')
    plt.legend()
    plt.grid()
    return plt.gcf()

