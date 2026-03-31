'''Unified analysis interface for all 4 modules: beta regression, macro classification, DCF valuation, and portfolio optimization.
Ties them together to help users easily run analyses and view results in one place.
'''

import pandas as pd
import numpy as np
from .data_fetcher import get_merged_data, get_risk_free_rate
from .beta_regression import run_adf, run_beta_regression, get_regression_summary, run_rolling_beta
from .dcf_montecarlo import get_dcf_summary
from .portfolio_optimizer import import_portfolio, screen_stocks, optimize_portfolio, calculate_var, get_portfolio_summary
from .macro_classifier import get_classifier_summary, predict_current_regime

def analyze_ticker(ticker):
    '''Run a comprehensive single stock analysis: beta regression + DCF valuation.
    
    Args:
        ticker (str): Stock ticker symbol to analyze.
    Returns:
        dict: A dictionary containing regression summary, DCF summary, and rolling beta.
    '''
    df = get_merged_data(ticker)
    adf = run_adf(df)
    regression = get_regression_summary(df)
    rolling_beta = run_rolling_beta(df)

    try: 
        dcf_summary = get_dcf_summary(ticker)
    except Exception:
        dcf_summary = None

    assessment = assess_ticker(regression, dcf_summary)

    return {
        'ticker': ticker,
        'beta': {
            'adf': adf,
            'regression': regression,
            'rolling_beta_now': float(rolling_beta.iloc[-1]),
            'rolling_beta_mean': float(rolling_beta.mean())
        },
        'dcf': dcf_summary,
        'assessment': assessment
    }

def assess_ticker(regression_summary, dcf_summary):
    '''Provide an assessment of the stock based on beta regression and dcf simulation.
    '''
    signals = []
    beta = regression_summary['Beta']
    if beta > 1.2:
        signals.append("High Beta (more volatile than market)")
    elif beta < 0.8:
        signals.append("Low Beta (less volatile than market)")
    else:
        signals.append("Market Beta (similar volatility to market)")

    if dcf_summary is not None:
        upside = dcf_summary['upside_pct']
        if upside > 20:
            signals.append("DCF indicates significant upside potential")
        elif upside < -20:
            signals.append("DCF indicates significant downside risk")
        else:
            signals.append("DCF indicates limited upside/downside")

        pct = dcf_summary['mc_pct_undervalued']
        signals.append(f"DCF Monte Carlo shows {pct:.1f}% of scenarios undervalued")
    else:
        signals.append("DCF not available for this ticker.")
    return " | ".join(signals)

def analyze_portfolio(tickers, profile=None):
    '''Run portfolio analysis using risk profile derived from predicted macro regime.
    Args:
        tickers (list of str): List of stock ticker symbols to include in the portfolio.
        profile (str, optional): Risk profile ('conservative', 'balanced', 'aggressive'). If None, it will be determined by macro regime.
    Returns:
        dict: A dictionary containing portfolio summary, optimized weights, and risk metrics.
    '''
    rfr = get_risk_free_rate()

    #Step 1 Check macro regime
    macro = get_classifier_summary()
    regime = macro['current_prediction']
    #Step 2 set risk profile unless overriden
    if profile is None:  
        if regime['regime'] == 'Contraction':
            profile = 'Conservative'
        elif regime['confidence'] > 0.7:
            profile = 'Aggressive'
        else:
            profile = 'Balanced'
    #Step 3 import and screen stocks
    log_returns = import_portfolio(tickers)
    screened = screen_stocks(log_returns, rfr)
    qualified_tickers = screened[screened['Flag'] == 'Pass'].index.tolist()
    #Step 4 optimize portfolio
    filtered = log_returns[qualified_tickers]
    optimized_returns, sharpe, weights = optimize_portfolio(filtered, rfr, profile)
    var_results = calculate_var(filtered, weights)
    #Step 5 Analyze each qualified ticker individually
    ticker_analyzed = {}
    for ticker in qualified_tickers:
        try:
            ticker_analyzed[ticker] = analyze_ticker(ticker)
        except Exception as e:
            ticker_analyzed[ticker] = {'error': str(e)}
    #Step 6 summarize results
    rec = build_rec(regime, profile, screened, qualified_tickers, weights, sharpe, var_results, ticker_analyzed) 
    return {
        'macro': {
            'regime': regime['regime'],
            'confidence': regime['confidence'],
            'as_of': regime['as_of'],
            'eval_results': macro['evaluation']
        },
        'profile': profile,
        'screening': screened,
        'qualified_tickers': qualified_tickers,
        'optimized_weights': dict(zip(qualified_tickers, weights)),
        'sharpe_ratio': sharpe,
        'var': var_results,
        'ticker_analysis': ticker_analyzed,
        'recommendation': rec
    }   

def build_rec(regime, profile, screened, qualified_tickers, weights, sharpe, var_results, ticker_analyzed):
    '''Build a recommendation and summary based on the various factors of analysis including 
      macro regime, risk profile, screening results, optimized portfolio performance, and individual ticker analyses.
    '''
    rec = []
    rec.append(f"The model is {regime['confidence']:.1f}% confident in the.{regime['regime']} with confidence ")
    rec.append(f"Based on this, a {profile} risk profile is recommended.")

    dropped = screened[screened['Flag'] != 'Pass'].index.tolist()
    if dropped:
        rec.append(f"The following tickers were flagged due to poor risk-return profiles: {', '.join(dropped)}.")
    else:
        rec.append("All tickers passed the screening process.")
    
    rec.append(f"The optimized portfolio has a Sharpe ratio of {sharpe:.2f} and the following Value at Risk metrics: {var_results}.")
    rec.append(f"Daily VaR at 95% confidence: {var_results['Parametric VaR']:.4f}")
    
    if var_results["Excess Kurtosis"] > 3: 
        rec.append("However, the portfolio returns exhibit excess kurtosis, indicating higher tail risk than a normal distribution.")
    rec.append("Reccommended allocation:")
    for ticker, weight in zip(qualified_tickers, weights):
        dcf_info = ""
        if ticker in ticker_analyzed and 'dcf' in ticker_analyzed[ticker]:
            dcf = ticker_analyzed[ticker]['dcf']
            dcf_info = f" (DCF upside: {dcf['upside_pct']:.1f}%)"
        rec.append(f" - {ticker}: {weight:.1%}{dcf_info}")
    return " ".join(rec)
