'''Discounted Cash Flow (DCF) model, values a company based on it's projected 
free cash flows discounted back to present value using a calculated WACC.
Runs 10,000 Monte Carlo simulations with bounded distributions.
Uses analyst consensus growth estimates when available, and it falls back to historical growth rates if not.
'''

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from .data_fetcher import get_risk_free_rate


def get_financials(ticker_symbol):
    '''Fetches financial statements and key metrics for a given ticker using yfinance.
    
    Returns:
        dict with revenue, free cash flow, operating margin, balance sheet, financials, and info.
    '''
    t = yf.Ticker(ticker_symbol)
    financials = t.financials
    balance_sheet = t.balance_sheet
    cashflow = t.cashflow

    revenue = financials.loc['Total Revenue']
    fcf = cashflow.loc['Free Cash Flow']
    operating_income = financials.loc['Operating Income']
    operating_margin = operating_income / revenue

    return {
        'revenue': revenue,
        'fcf': fcf,
        'operating_margin': operating_margin,
        'balance_sheet': balance_sheet,
        'financials': financials,
        'info': t.info
    }

def get_growth_estimates(ticker_symbol):
    '''Get growth estimates from Yahoo Finance analyst consensus. 
    Historical growth rates as fallback if estimates are not available.
    
    Returns:
        dict with historical growth, analyst growth/None, and growth source.
    '''

    t=yf.Ticker(ticker_symbol)
    historical_revenue = t.financials.loc['Total Revenue'].pct_change(-1, fill_method=None).dropna().mean()
    
    try:
        estimates = t.growth_estimates
        if estimates is not None and not estimates.empty:
            analyst_growth = estimates.loc['growth'].iloc[0]
            return {
                'historical_growth': historical_revenue,
                'analyst_growth': analyst_growth,
                'growth_source': 'Analyst Estimates'
            }
    except Exception as e:
        pass
    return {
        'historical_growth': historical_revenue,
        'analyst_growth': None,
        'growth_source': 'Historical Growth'
    }

def project_growth_rates(current_growth, terminal_growth, years):
    '''Growth rate decays from current to terminal linearly over projection period.
    
    Prevents unrealistic compounding of high growth rates over long periods.

    '''
    return [current_growth + (terminal_growth - current_growth) * (i / years) for i in range(1, years + 1)]

def discount_fcfs(revenue_base, growth_rates, fcf_margin, wacc, terminal_growth):
    '''Calculate discounted free cash flows for each projection year and terminal value.
    
    Returns:
        dict with discounted_fcfs list and discounted_terminal value.
    '''
    years = len(growth_rates)
    discounted_fcfs = []

    current_revenue = revenue_base 
    for i, j in enumerate(growth_rates):
        current_revenue *= (1 + j)
        projected_fcf = current_revenue * fcf_margin
        discounted_fcf = projected_fcf / (1 + wacc) ** (i + 1)
        discounted_fcfs.append(discounted_fcf)
    final_year_fcf = current_revenue * fcf_margin
    terminal_value = (final_year_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    discounted_terminal = terminal_value / (1 + wacc) ** years
    return {
        'discounted_fcfs': discounted_fcfs,
        'terminal_value': terminal_value,
        'discounted_terminal': discounted_terminal,
        'intrinsic_value': sum(discounted_fcfs) + discounted_terminal
    }

def calculate_wacc(financials_dict):
    '''Calculated Weighted Average Cost of Capital (WACC)
    
    Uses the Capital Asset Pricing Model (CAPM) for cost of equity,
    and interest expense/total debt for cost of debt.
    Falls back to risk-free rate + 1% if interest expense data is unavailable.
    '''
    rfr = get_risk_free_rate()
    balance_sheet = financials_dict['balance_sheet']
    financials = financials_dict['financials']
    beta = financials_dict['info']['beta']

    equity_risk_premium = 0.05
    cost_of_equity = rfr + beta * equity_risk_premium

    total_debt = balance_sheet.loc['Total Debt'].iloc[0]
    cash_and_equivalents = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
    total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
    net_debt = total_debt - cash_and_equivalents
    debt_weight = net_debt / (net_debt + total_equity)
    equity_weight = total_equity / (net_debt + total_equity)

    interest_expense = financials.loc['Interest Expense'].dropna()
    if len(interest_expense) > 0:
        cost_of_debt = abs(interest_expense.iloc[0]) / total_debt
    else:
        for i in range(len(financials.columns)):
            val = financials.loc['Interest Expense'].iloc[i]
            if not pd.isna(val):
                cost_of_debt = abs(val) / total_debt
                break
        else:
            cost_of_debt = rfr + 0.01

    wacc = debt_weight * cost_of_debt * (1 - 0.21) + equity_weight * cost_of_equity
    return wacc


def run_dcf(ticker_symbol, projection_years=5, terminal_growth=0.025):
    '''Runs deterministic DCF valuation for a given ticker,
    using growth decay to prevent unrealistic compounding growth.

    Returns:
        dict with implied price, WACC, revenue growth, 
        FCF margin, current price, and shares outstanding.
    '''

    fin = get_financials(ticker_symbol)
    wacc = calculate_wacc(fin)
    growth = get_growth_estimates(ticker_symbol)

    revenue = fin['revenue']
    fcf = fin['fcf']
    shares_outstanding = fin['info']['sharesOutstanding']
    current_price = fin['info']['currentPrice']

    base_growth = growth['analyst_growth'] if growth['analyst_growth'] is not None else growth['historical_growth']
    fcf_margin_mean = (fcf / revenue).dropna().mean()

    growth_rates = project_growth_rates(base_growth, terminal_growth, projection_years)
    result = discount_fcfs(revenue.iloc[0], growth_rates, fcf_margin_mean, wacc, terminal_growth)
    implied_price = result['intrinsic_value'] / shares_outstanding

    return {
        'implied_price': implied_price,
        'current_price': current_price,
        'wacc': wacc,
        'upside_pct': (implied_price - current_price) / current_price * 100,
        'growth_source': growth['growth_source'],\
        'growth_rates': growth_rates,
        'fcf_margin': fcf_margin_mean,
        'shares_outstanding': shares_outstanding
    }


def run_dcf_monte_carlo(ticker_symbol, num_simulations=10000, projection_years=5, terminal_growth=0.025):
    '''Run Monte Carlo DCF with 10,000 simulations.
    
    Vary revenue growth (truncated normal) and WACC (triangular distribution).
    Growth decays from current to terminal linearly to prevent unrealistic compounding.\
    
    Returns:
        dict with array of simulated prices, summary statistics,
        and percentage of simulations above current price.
    '''
    fin = get_financials(ticker_symbol)
    wacc = calculate_wacc(fin)

    revenue = fin['revenue']
    fcf = fin['fcf']
    shares_outstanding = fin['info']['sharesOutstanding']
    current_price = fin['info']['currentPrice']

    # Full series for distribution parameters
    revenue_growth_series = revenue.pct_change(-1, fill_method=None).dropna()
    fcf_margin_mean = (fcf / revenue).dropna().mean()

    rg_mean = revenue_growth_series.mean()
    rg_std = revenue_growth_series.std()

    # Truncated normal for revenue growth (bounded: -10% to +15%)
    revenue_growth_dist = stats.truncnorm(
        (-0.10 - rg_mean) / rg_std,
        (0.15 - rg_mean) / rg_std,
        loc=rg_mean,
        scale=rg_std
    )

    # Triangular distribution for WACC
    wacc_dist = stats.triang(c=0.5, loc=wacc - 0.02, scale=0.04)

    simulated_prices = []

    fcf_margin_series = (fcf / revenue).dropna()
    fcf_margin_mean = fcf_margin_series.mean()
    fcf_std = fcf_margin_series.std()
    
    # Beta distribution for FCF margin (bounded between 0 and 1)
    fm_alpha = (fcf_margin_mean * (1 - fcf_margin_mean)) / fcf_std**2 - 1
    fm_a = fcf_margin_mean * fm_alpha
    fm_b = (1 - fcf_margin_mean) * fm_alpha
    fcf_margin_dist = stats.beta(fm_a, fm_b)

    for _ in range(num_simulations):
        sim_rg = revenue_growth_dist.rvs()
        sim_wacc = wacc_dist.rvs()
        sim_fcf_margin = fcf_margin_dist.rvs()

        if sim_wacc <= terminal_growth:
            continue

        growth_rates = project_growth_rates(sim_rg, terminal_growth, projection_years)
        result = discount_fcfs(revenue.iloc[0], growth_rates, sim_fcf_margin, sim_wacc, terminal_growth)
        simulated_price = result['intrinsic_value'] / shares_outstanding
        simulated_prices.append(simulated_price)

    prices = np.array(simulated_prices)

    return {
        'prices': prices,
        'mean_price': np.mean(prices),
        'median_price': np.median(prices),
        'ci_5': np.percentile(prices, 5),
        'ci_95': np.percentile(prices, 95),
        'current_price': current_price,
        'pct_undervalued': np.mean(prices > current_price) * 100
    }

def get_dcf_summary(ticker_symbol):
    '''Run full DCF analysis and return summary to be used in analysis page.
    
    Combines deterministic DCF, Monte Carlo results, and
    growth estimates for comprehensive overview of valuation.
    '''
    dcf = run_dcf(ticker_symbol)
    mc = run_dcf_monte_carlo(ticker_symbol)
    return {
        'implied_price': dcf['implied_price'],
        'current_price': dcf['current_price'],
        'upside_pct': dcf['upside_pct'],
        'wacc': dcf['wacc'],
        'growth_rates': dcf['growth_rates'],
        'growth_source': dcf['growth_source'],
        'fcf_margin': dcf['fcf_margin'],
        'mc_mean': mc['mean_price'],
        'mc_median': mc['median_price'],
        'mc_ci_5': mc['ci_5'],
        'mc_ci_95': mc['ci_95'],    
        'mc_pct_undervalued': mc['pct_undervalued']
    }

def plot_monte_carlo_distribution(simulated_prices, current_price):
    '''Plots histogram of Monte Carlo simulated prices with current price overlay.'''
    plt.figure(figsize=(10, 6))
    plt.hist(simulated_prices, bins=50, alpha=0.7, color='blue')
    plt.axvline(current_price, color='red', linestyle='dashed', linewidth=2, label=f'Current Price: ${current_price:.2f}')
    plt.title('Monte Carlo Simulation of Implied Share Price')
    plt.xlabel('Implied Share Price ($)')
    plt.ylabel('Frequency')
    plt.legend()
    return plt.gcf()


def plot_dcf_sensitivity(ticker_symbol, projection_years=5, terminal_growth=0.025):
    '''Plots sensitivty heatmap of implied price vs growth and WACC.

    Args:
        fin: financials dict from get_financials()
        wacc: calculated WACC for the company
    '''
    fin = get_financials(ticker_symbol)
    wacc = calculate_wacc(fin)

    revenue = fin['revenue']
    fcf = fin['fcf']
    shares_outstanding = fin['info']['sharesOutstanding']
    fcf_margin_mean = (fcf / revenue).dropna().mean()

    revenue_growth_range = np.linspace(-0.05, 0.10, 5)
    wacc_range = np.linspace(wacc - 0.02, wacc + 0.02, 5)

    heatmap_data = []
    for rg in revenue_growth_range:
        for w in wacc_range:
            if w <= terminal_growth:
                continue
            growth_rates = project_growth_rates(rg, terminal_growth, projection_years)
            result = discount_fcfs(revenue.iloc[0], growth_rates, fcf_margin_mean, w, terminal_growth)
            implied_price = result['intrinsic_value'] / shares_outstanding
            heatmap_data.append({'Revenue Growth': round(rg, 4), 'WACC': round(w, 4), 'Implied Price': implied_price})
        
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_pivot = heatmap_df.pivot(index='Revenue Growth', columns='WACC', values='Implied Price')

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Sensitivity of Implied Share Price to Revenue Growth and WACC')
    return plt.gcf()


def plot_scenario_comparison(ticker_symbol, projection_years=5, terminal_growth=0.025):
    '''Plot bull vs. base vs. bear scenario comparsion against current price.
    
    Bull: analyst growth (or historical + 5%)
    Base: blended growth with decay
    Bear: negative growth and higher WACC
    '''
    fin = get_financials(ticker_symbol)
    wacc = calculate_wacc(fin)
    growth = get_growth_estimates(ticker_symbol)

    revenue = fin['revenue']
    fcf = fin['fcf']
    shares_outstanding = fin['info']['sharesOutstanding']
    current_price = fin['info']['currentPrice']
    fcf_margin_mean = (fcf / revenue).dropna().mean()

    bull_growth = growth['analyst_growth'] if growth['analyst_growth'] is not None else growth['historical_growth'] + 0.05
    base_growth = growth['analyst_growth'] if growth['analyst_growth'] is not None else growth['historical_growth']
    bear_growth = min(growth['historical_growth'] - 0.03, -0.02)

    scenarios = {
        'Bull': {'growth': bull_growth, 'wacc': wacc - 0.01},
        'Base': {'growth': base_growth, 'wacc': wacc},
        'Bear': {'growth': bear_growth, 'wacc': wacc + 0.01}
    }

    scenario_prices = {}
    for name, params in scenarios.items():
        if params['wacc'] <= terminal_growth:
            scenario_prices[name] = None
            continue
        growth_rates = project_growth_rates(params['growth'], terminal_growth, projection_years)
        result = discount_fcfs(revenue.iloc[0], growth_rates, fcf_margin_mean, params['wacc'], terminal_growth)
        scenario_prices[name] = result['intrinsic_value'] / shares_outstanding

    scenario_df = pd.DataFrame(list(scenario_prices.items()), columns=['Scenario', 'Implied Price'])

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Scenario', y='Implied Price', data=scenario_df, hue='Scenario', legend=False)
    plt.axhline(current_price, color='red', linestyle='dashed', label=f'Current Price: ${current_price:.2f}')
    plt.title('DCF Valuation Under Different Scenarios')
    plt.legend()
    return plt.gcf()