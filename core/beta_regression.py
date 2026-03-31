'''Beta regression with Newey-West SE and residual diagnostics.

Computes rolling and static beta for any given stock against SPY, 
with full diagnostics: ADF tests, residual plots, Durbin-Watson, 
and Breusch-pagan.
'''

from .data_fetcher import get_merged_data
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.diagnostic import het_breuschpagan   
from scipy.stats import kurtosis as calc_kurtosis
import matplotlib.pyplot as plt
import numpy as np

def run_adf(df):
    '''Runs Augmented Dicky-Fuller tests on excess returns to
    determine whether the series is stationary.
    
    Returns:
        dict with ADF statistic, p-value, critical values, 
        and pass/fail for both stock and SPY excess returns.
        '''
    stock_adf = ts.adfuller(df['Excess_Returns'].dropna())
    spy_adf = ts.adfuller(df['Excess_Returns_SPY'].dropna())
    return {
        'Stock': {'ADF Statistic': stock_adf[0], 'p_value': stock_adf[1], 'Critical Values': stock_adf[4], 'Stationary': stock_adf[1] < 0.05},
        'SPY': {'ADF Statistic': spy_adf[0], 'p_value': spy_adf[1], 'Critical Values': spy_adf[4], 'Stationary': spy_adf[1] < 0.05}
    }

def run_beta_regression(df):
    '''Run OLS beta regression with and without Newey-West SE.
    Regresses stock excess returns on SPY excess returns

    Returns:
        tuple of (model, model_nw) - base OLS and Newey-West corrected results.
    '''
    X = sm.add_constant(df['Excess_Returns_SPY'].dropna())
    y = df['Excess_Returns'].dropna()
    X,y = X.align(y, join='inner', axis=0)
    model = sm.OLS(y, X).fit()
    # Use newey-west standard errors to account for heteroskedasticity and autocorrelation
    model_nw = model.get_robustcov_results(cov_type='HAC', maxlags=5)
    return model, model_nw

def get_regression_summary(df):
    '''Extract the key regression metrics and statistics to display.
    
     Returns:
        dict with beta, alpha, R-squared, standard errors, Durbin-Watson,
        kurtosis, Breusch-Pagan test results, and number of observations.
    '''
    model, model_nw = run_beta_regression(df)
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, model.model.exog)
    summary = {
        'Beta': model.params['Excess_Returns_SPY'],
        'Alpha': model.params['const'],
        'r_squared': model.rsquared,
        'beta_se_ols': model.bse['Excess_Returns_SPY'],
        'beta_se_nw': model_nw.bse[1],
        'durbin_watson': sm.stats.stattools.durbin_watson(model.resid),
        'kurtosis': calc_kurtosis(model.resid, fisher=True),
        'breusch_pagan_stat': bp_stat,
        'breusch_pagan_pvalue': bp_pvalue,
        'heteroskedasticity': bp_pvalue < 0.05,
        'nobs': int(model.nobs),
    }
    return summary

def run_rolling_beta(df, window = 60):
    '''Run rolling beta regression with Newey-West SE, over a time window of 60 days.
    '''
    rolling = RollingOLS(df['Excess_Returns'], sm.add_constant(df['Excess_Returns_SPY']), window=window)
    results = rolling.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    results = results.params.rename(columns={'Excess_Returns_SPY': 'Beta'})
    return results['Beta'].dropna()

def plot_residual_diagnostics(df):
    '''Plot residual diagnostics for static beta regression.'''
    model, model_nw = run_beta_regression(df)
    fig = sm.graphics.plot_regress_exog(model, 'Excess_Returns_SPY', fig=plt.figure(figsize=(12, 8)))
    return fig

def plot_quarterly_scatter(df, results):
    '''Plot quarterly log returns vs quarterly rolling beta estimates with a regression line.'''
    quarterly = df['Log_Returns'].resample('QE').sum()
    quarterly_beta = results.resample('QE').last()
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
    '''Plot annualized realized volatility over time.'''
    ARV = df['Log_Returns'].rolling(window=60).std() * np.sqrt(252)
    plt.figure(figsize=(10, 6))
    plt.plot(ARV, label='Realized Volatility (60-day)')
    plt.xlabel('Date')
    plt.ylabel('Annualized Realized Volatility')
    plt.title('Realized Volatility Over Time')
    plt.legend()
    plt.grid()
    return plt.gcf()

