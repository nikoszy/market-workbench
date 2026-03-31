''' Market Intelligence Workbench: Research Project Edition '''
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sophomore Year Research Project", layout= "wide")

# --- RESEARCH CONTEXT ---
st.title("About the project...")
st.write("""
In one of my statistics classes I completed a project studying which financial indicators had the strongest association 
with Quarterly Stock Returns and found that Beta had the most significant relationship. This helped me understand that
volatility is an important factor in thr market, and I wanted to learn more.

I built this tool during my Sophomore year spring break to explore uses of statistical modeling, machine learning
and financial theory hand in hand to come up with conclusions. My goal was to build something that I could actually use
to give me actionable insight in my investments, and have a project that I can keep adding to as I develop my 
machine learning and analysis skills more and more.
""")

st.header("What does it do?")
st.write("""
You enter a list of stock tickers, for example a portfolio of 10 different stocks. The system screens your stocks to determine
if they might be risky by screening for high volatility, negative sharpe ratio, or a massive drop in value (>|-40%|). Using a
Random Forest Classifier, classifies predicted Macroeconomic Regime over the next 6 months based on various macro indicators 
such as 10 year treasury yield, 2 year treasury yield, and credit spread, which were found to be the most important non-related
features. Then, using a mean-variance optimizer, runs a portfolio optimization using a risk-profile decided by the macro regime.
Finally, runs a DCF simulation valuing each stock on it's free cash flow in terms of today's value.
""")

st.divider()

# --- INPUTS ---
ticker = st.text_input("Enter all stock tickers, separated by comma.", placeholder="e.g., AAPL, JPM, XOM, JNJ, AMZN, TLT")
profile_override = st.selectbox("Risk Profile", ["Auto (Macro-Informed)", "Conservative", "Balanced", "Aggressive"])
tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()]

if st.button("Run Analysis", type="primary"):
    st.session_state['run_analysis'] = True

if st.session_state.get('run_analysis', False) and len(tickers) >= 2:
    
    # --- STEP 1: MACRO REGIME ---
    st.header("Macro Regime Classifier")
    st.write("> **Methodology:** Random Forest model trained on CPI momentum, yield curve spread, and credit spread. Train/test split, model had 70% accuracy with AUC = 0.73 and 100% Contraction recall. This means when the model says contraction, you better listen.")
    
    try:
        from core.macro_classifier import get_classifier_summary, plot_feature_importance
        macro = get_classifier_summary()
        regime = macro['current_prediction']

        if regime['regime'] == "Expansion":
            st.success(f"**{regime['regime']}** — {regime['confidence']:.0%} confidence · As of {regime['as_of']}")
        else:
            st.error(f"**{regime['regime']}** — {regime['confidence']:.0%} confidence · As of {regime['as_of']}")

        if profile_override == 'Auto (Macro-Informed)':
            profile = 'Conservative' if regime['regime'] == 'Contraction' else ('Aggressive' if regime['confidence'] > 0.7 else 'Balanced')
        else:
            profile = profile_override

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_m = plot_feature_importance(macro['model'], macro['macro_data'][['Yield Curve', 'CPI_6M_Change','Credit_Spread']].dropna())
            st.pyplot(fig_m)
            plt.close()
        with col_m2:
            ev = macro['evaluation']
            st.metric("Accuracy", f"{ev['accuracy']:.0%}")
            st.metric("AUC", f"{ev['auc']:.2f}") # Restored AUC
            st.metric("Contraction Recall", f"{ev['contraction_recall']:.0%}")
    except Exception as e:
        st.error(f"Macro error: {e}")
        profile = "Balanced"

    st.divider()

    # --- STEP 2: STOCK SCREENING ---
    st.header("Stock Screening")
    try:
        from core.portfolio_optimizer import import_portfolio, screen_stocks
        from core.data_fetcher import get_risk_free_rate
        rfr = get_risk_free_rate()
        returns = import_portfolio(tickers)
        screening = screen_stocks(returns, rfr)

        st.dataframe(screening.style.format({
            'Annualized Return': '{:.1%}', 'Annualized Volatility': '{:.1%}',
            'Sharpe Ratio': '{:.2f}', 'Max Drawdown': '{:.1%}'
        }), use_container_width=True)
        
        passed = screening[screening['Flag'] == 'Pass'].index.tolist()
        flagged = screening[screening['Flag'] != 'Pass'].index.tolist()
        qualified = passed + st.multiselect("Include flagged stocks anyway?", flagged) if flagged else passed
    except Exception as e:
        st.error(f"Screening error: {e}")
        qualified = tickers

    st.divider()

    # --- STEP 3: OPTIMIZER ---
    if len(qualified) >= 2:
        st.header("Portfolio Optimizer")
        st.write("> **Methodology:** Mean-variance optimizer with 3 risk-profiles that have different weight constraints, bootstrapped efficient frontier with 1,000 resamples, computed parametric and historical VaR, detecting fat tails.")
        try:
            from core.portfolio_optimizer import (
                optimize_portfolio, calculate_var, plot_optimized_portfolio, 
                plot_var_comparison, boot, plot_bootstrap
            )
            filtered = returns[qualified]
            optimized_returns, sharpe, weights = optimize_portfolio(filtered, rfr, profile)
            var_results = calculate_var(filtered, weights)

            c1, c2, c3 = st.columns(3)
            c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
            c2.metric("Daily VaR (95%)", f"{var_results['Parametric VaR']:.2%}")
            c3.metric("Annual Volatility", f"{np.sqrt(np.dot(weights, np.dot(filtered.cov() * 252, weights))):.1%}")

            st.pyplot(plot_optimized_portfolio(filtered, optimized_returns))
            plt.close()

            t_risk, t_boot = st.tabs(["Risk Analysis", "Bootstrapped Frontier"])
            with t_risk:
                st.pyplot(plot_var_comparison(filtered, weights))
                plt.close()
                if var_results['Excess Kurtosis'] > 3:
                    st.warning("Fat Tails detected! VaR likely understates risk.")
            with t_boot:
                bw, br, bv = boot(filtered, rfr, profile)
                st.pyplot(plot_bootstrap(br, bv, rfr))
                plt.close()
        except Exception as e:
            st.error(f"Optimizer error: {e}")

    st.divider()

    # --- STEP 4: INDIVIDUAL ANALYSIS (RESTORED SAFETY & INTERPRETATIONS) ---
    st.header("Methods/Models: Deep Dive")
    for t in qualified:
        with st.expander(f"Analysis for {t}"):
            try:
                # SAFE HANDLING FOR INDEX FUNDS / ETFS
                try:
                    from core.dcf_montecarlo import get_implied_growth_rate
                    igr = get_implied_growth_rate(t)
                    if igr.get('implied_growth') is not None:
                        st.write(f"**Market-implied growth rate:** {igr['implied_growth']:.1%}")
                        if igr['implied_growth'] > 0.50:
                            st.caption(f"Implied growth above 50% suggests significant margin expansion expectations. Current FCF margin: {igr['fcf_margin']:.1%}")
                        else:
                            st.caption(igr.get('interpretation', ""))
                except Exception:
                    pass # Silence errors for ETFs lacking growth data

                from core.analyzer import analyze_ticker
                analysis = analyze_ticker(t)
                col_b, col_d = st.columns(2)
                
                with col_b:
                    st.subheader("Beta Regression")
                    st.write("OLS using Newey-West standard errors, Breusch-Pagan for heteroskedasticity testing, computing rolling 60-day beta estimation and adf stationary tests.")
                    reg = analysis['beta']['regression']
                    st.metric("Beta", f"{reg['Beta']:.2f}")
                    st.metric("R²", f"{reg['r_squared']:.3f}")
                    with st.expander("Advanced Stats"):
                        st.write(f"Breusch-Pagan p-value: {reg['breusch_pagan_pvalue']:.4f}")
                        st.write(f"Durbin-Watson: {reg['durbin_watson']:.3f}")
                
                with col_d:
                    st.subheader("DCF Monte Carlo")
                    st.write("Ran 10,000 simulations avoiding normal distributions, using truncated normal, beta, and triangular distributions.")
                    if analysis['dcf'] is not None:
                        d = analysis['dcf']
                        st.metric("Implied Price", f"${d['implied_price']:.2f}")
                        st.metric("Upside / Downside", f"{d['upside_pct']:+.1f}%")
                        st.write("**Monte Carlo Simulation Results**")
                        st.write(f"Mean: ${d['mc_mean']:.2f} · Median: ${d['mc_median']:.2f}")
                        st.write(f"90% CI: ${d['mc_ci_5']:.2f} – ${d['mc_ci_95']:.2f}")
                        st.write(f"Simulations supporting current price: **{d['mc_pct_undervalued']:.1f}%**")
                    else:
                        st.info("DCF Valuation not available for this ticker (Common for ETFs/Index Funds).")

                if analysis['dcf'] is not None:
                    c_mc, c_sens = st.columns(2)
                    with c_mc:
                        from core.dcf_montecarlo import run_dcf_monte_carlo, plot_monte_carlo_distribution
                        mc_data = run_dcf_monte_carlo(t)
                        st.pyplot(plot_monte_carlo_distribution(mc_data['prices'], mc_data['current_price']))
                        plt.close()
                    with c_sens:
                        from core.dcf_montecarlo import plot_dcf_sensitivity
                        st.pyplot(plot_dcf_sensitivity(t))
                        plt.close()

            except Exception as e:
                st.error(f"Error analyzing {t}: {e}")

    # --- CONCLUSION ---
    st.divider()
    st.header("What did I learn?")
    st.info("""
    Predicting the stock market is hard. Literally all of these variables only account for such a small amount of variability in returns 
    in the stock market, and just trying to use the free numbers that are available on the internet can only get you so far. I am excited 
    to expand this project to explore more machine learning capabilities and expand the scope of predictors that the model captures to 
    incude political influences, geopolitical events, news headlines, and other factors that economic indicators cannot capture.
    """)
    st.caption("Market Intelligence Workbench · Data from yfinance & FRED · Not financial advice.")

else:
    st.info("Enter at least 2 stock tickers and click Run Analysis to get started.")