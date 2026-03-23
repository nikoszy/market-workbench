import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

os.chdir(os.path.expanduser('~/market-workbench'))
from core.data_fetcher import get_risk_free_rate


def get_financials(ticker_symbol):
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


def calculate_wacc(financials_dict):
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
    fin = get_financials(ticker_symbol)
    wacc = calculate_wacc(fin)

    revenue = fin['revenue']
    fcf = fin['fcf']
    shares_outstanding = fin['info']['sharesOutstanding']

    revenue_growth = revenue.pct_change(-1, fill_method=None).dropna().mean()
    fcf_margin_mean = (fcf / revenue).dropna().mean()

    discounted_fcfs = []
    for i in range(projection_years):
        projected_revenue = revenue.iloc[0] * (1 + revenue_growth) ** (i + 1)
        projected_fcf = projected_revenue * fcf_margin_mean
        discounted_fcf = projected_fcf / (1 + wacc) ** (i + 1)
        discounted_fcfs.append(discounted_fcf)

    final_year_fcf = revenue.iloc[0] * (1 + revenue_growth) ** projection_years * fcf_margin_mean
    terminal_value = (final_year_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    discounted_terminal = terminal_value / (1 + wacc) ** projection_years

    intrinsic_value = sum(discounted_fcfs) + discounted_terminal
    implied_price = intrinsic_value / shares_outstanding

    return {
        'implied_price': implied_price,
        'wacc': wacc,
        'revenue_growth': revenue_growth,
        'fcf_margin': fcf_margin_mean,
        'current_price': fin['info']['currentPrice'],
        'shares_outstanding': shares_outstanding
    }


def run_dcf_monte_carlo(ticker_symbol, num_simulations=10000, projection_years=5, terminal_growth=0.025):
    fin = get_financials(ticker_symbol)
    wacc = calculate_wacc(fin)

    revenue = fin['revenue']
    fcf = fin['fcf']
    shares_outstanding = fin['info']['sharesOutstanding']

    # Full series for distribution parameters
    revenue_growth_series = revenue.pct_change(-1, fill_method=None).dropna()
    operating_margin_series = fin['operating_margin'].dropna()
    fcf_margin_mean = (fcf / revenue).dropna().mean()

    rg_mean = revenue_growth_series.mean()
    rg_std = revenue_growth_series.std()
    om_mean = operating_margin_series.mean()
    om_std = operating_margin_series.std()
    om_var = operating_margin_series.var()

    # Truncated normal for revenue growth (bounded: -10% to +15%)
    revenue_growth_dist = stats.truncnorm(
        (-0.10 - rg_mean) / rg_std,
        (0.15 - rg_mean) / rg_std,
        loc=rg_mean,
        scale=rg_std
    )

    # Beta distribution for operating margin
    alpha_param = (om_mean * (1 - om_mean)) / om_var - 1
    a = alpha_param * om_mean
    b = alpha_param * (1 - om_mean)
    operating_margin_dist = stats.beta(a, b)

    # Triangular distribution for WACC
    wacc_low = wacc - 0.02
    wacc_high = wacc + 0.02
    wacc_dist = stats.triang(c=0.5, loc=wacc_low, scale=wacc_high - wacc_low)

    simulated_prices = []

    for _ in range(num_simulations):
        sim_rg = revenue_growth_dist.rvs()
        sim_wacc = wacc_dist.rvs()

        if sim_wacc <= terminal_growth:
            continue

        sim_discounted_fcfs = []
        for i in range(projection_years):
            projected_revenue = revenue.iloc[0] * (1 + sim_rg) ** (i + 1)
            projected_fcf = projected_revenue * fcf_margin_mean
            discounted_fcf = projected_fcf / (1 + sim_wacc) ** (i + 1)
            sim_discounted_fcfs.append(discounted_fcf)

        final_fcf = revenue.iloc[0] * (1 + sim_rg) ** projection_years * fcf_margin_mean
        sim_terminal = (final_fcf * (1 + terminal_growth)) / (sim_wacc - terminal_growth)
        discounted_terminal = sim_terminal / (1 + sim_wacc) ** projection_years

        sim_value = sum(sim_discounted_fcfs) + discounted_terminal
        sim_price = sim_value / shares_outstanding
        simulated_prices.append(sim_price)

    return np.array(simulated_prices)


def plot_monte_carlo_distribution(simulated_prices, current_price):
    plt.figure(figsize=(10, 6))
    plt.hist(simulated_prices, bins=50, alpha=0.7, color='blue')
    plt.axvline(current_price, color='red', linestyle='dashed', linewidth=2, label=f'Current Price: ${current_price:.2f}')
    plt.title('Monte Carlo Simulation of Implied Share Price')
    plt.xlabel('Implied Share Price ($)')
    plt.ylabel('Frequency')
    plt.legend()
    return plt.gcf()


def plot_dcf_sensitivity(ticker_symbol, projection_years=5, terminal_growth=0.025):
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
            discounted_fcfs = []
            for i in range(projection_years):
                projected_revenue = revenue.iloc[0] * (1 + rg) ** (i + 1)
                projected_fcf = projected_revenue * fcf_margin_mean
                discounted_fcf = projected_fcf / (1 + w) ** (i + 1)
                discounted_fcfs.append(discounted_fcf)

            final_fcf = revenue.iloc[0] * (1 + rg) ** projection_years * fcf_margin_mean
            terminal_value = (final_fcf * (1 + terminal_growth)) / (w - terminal_growth)
            discounted_terminal = terminal_value / (1 + w) ** projection_years
            intrinsic_value = sum(discounted_fcfs) + discounted_terminal
            implied_price = intrinsic_value / shares_outstanding
            heatmap_data.append({'Revenue Growth': round(rg, 4), 'WACC': round(w, 4), 'Implied Price': implied_price})

    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_pivot = heatmap_df.pivot(index='Revenue Growth', columns='WACC', values='Implied Price')

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Sensitivity of Implied Share Price to Revenue Growth and WACC')
    return plt.gcf()


def plot_scenario_comparison(ticker_symbol, projection_years=5, terminal_growth=0.025):
    fin = get_financials(ticker_symbol)
    wacc = calculate_wacc(fin)

    revenue = fin['revenue']
    fcf = fin['fcf']
    shares_outstanding = fin['info']['sharesOutstanding']
    current_price = fin['info']['currentPrice']
    fcf_margin_mean = (fcf / revenue).dropna().mean()
    revenue_growth_mean = revenue.pct_change(-1, fill_method=None).dropna().mean()

    scenarios = {
        'Bull Case': {'growth': 0.10, 'wacc': wacc - 0.01},
        'Base Case': {'growth': revenue_growth_mean, 'wacc': wacc},
        'Bear Case': {'growth': -0.05, 'wacc': wacc + 0.01}
    }

    scenario_prices = {}
    for name, params in scenarios.items():
        rg = params['growth']
        w = params['wacc']

        if w <= terminal_growth:
            scenario_prices[name] = np.nan
            continue

        discounted_fcfs = []
        for i in range(projection_years):
            projected_revenue = revenue.iloc[0] * (1 + rg) ** (i + 1)
            projected_fcf = projected_revenue * fcf_margin_mean
            discounted_fcf = projected_fcf / (1 + w) ** (i + 1)
            discounted_fcfs.append(discounted_fcf)

        final_fcf = revenue.iloc[0] * (1 + rg) ** projection_years * fcf_margin_mean
        terminal_value = (final_fcf * (1 + terminal_growth)) / (w - terminal_growth)
        discounted_terminal = terminal_value / (1 + w) ** projection_years
        intrinsic_value = sum(discounted_fcfs) + discounted_terminal
        scenario_prices[name] = intrinsic_value / shares_outstanding

    scenario_df = pd.DataFrame(list(scenario_prices.items()), columns=['Scenario', 'Implied Price'])

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Scenario', y='Implied Price', data=scenario_df, hue='Scenario', legend=False)
    plt.axhline(current_price, color='red', linestyle='dashed', label=f'Current Price: ${current_price:.2f}')
    plt.title('DCF Valuation Under Different Scenarios')
    plt.legend()
    return plt.gcf()