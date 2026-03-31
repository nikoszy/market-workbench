''' Market Intelligence Workbench: Research Project Edition '''
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sophomore Year Spring Break Project", layout= "wide")

# --- Research Project Header ---
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

# --- Input Section ---
ticker = st.text_input("Enter all stock tickers, seperated by comma.", placeholder="e.g., AAPL, JPM, XOM, JNJ, AMZN, TLT")
profile_override = st.selectbox("Risk Profile", ["Auto (Macro-Informed)", "Conservative", "Balanced", "Aggressive"])
tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()]

if st.button("Run Analysis", type="primary"):
    st.session_state['run_analysis'] = True

if st.session_state.get('run_analysis', False) and len(tickers) >= 2:
    
    # Step 1: Macro Regime Classifier
    st.header("Macro Regime Classifier")
    st.write("> **Methodology:** Random Forest model trained on CPI momentum, yield curve spread, and credit spread. Train/test split, model had 70% accuracy with AUC = 0.73 and 100% Contraction recall. This means when the model says contraction, you better listen.")
    
    with st.spinner("Training macro regime classifying model..."):
        try:
            from core.macro_classifier import get_classifier_summary, plot_feature_importance
            macro = get_classifier_summary()
            regime = macro['current_prediction']

            if regime['regime'] == "Expansion":
                st.success(f"**{regime['regime']}** — {regime['confidence']:.0%} confidence · As of {regime['as_of']}")
            else:
                st.error(f"**{regime['regime']}** — {regime['confidence']:.0%} confidence · As of {regime['as_of']}")

            if profile_override == 'Auto (Macro-Informed)':
                if regime['regime'] == 'Contraction':
                    profile = 'Conservative'
                elif regime['confidence'] > 0.7:
                    profile = 'Aggressive'
                else:
                    profile = 'Balanced'
                st.info(f"Our macro regime classifying model suggests a **{profile}** profile.")
            else:
                profile = profile_override

            col1, col2 = st.columns(2)
            with col1:
                fig = plot_feature_importance(macro['model'], macro['macro_data'][['Yield Curve', 'CPI_6M_Change','Credit_Spread']].dropna())
                st.pyplot(fig)
                plt.close()
            with col2:
                ev = macro['evaluation']
                st.metric("Accuracy", f"{ev['accuracy']:.0%}")
                st.metric("AUC", f"{ev['auc']:.2f}")
                st.metric("Contraction Recall", f"{ev['contraction_recall']:.0%}")

        except Exception as e:
            st.error(f"Macro classifier error: {e}")
            profile = "Balanced"
        
    st.divider()

    # Step 2: Screen Stocks
    st.header("Stock Screening")
    with st.spinner("Pulling price data and screening stocks..."):
        try:
            from core.portfolio_optimizer import import_portfolio, screen_stocks
            from core.data_fetcher import get_risk_free_rate

            rfr = get_risk_free_rate()
            returns = import_portfolio(tickers)
            screening = screen_stocks(returns, rfr)

            st.dataframe(screening.style.format({
                'Annualized Return': '{:.1%}',
                'Annualized Volatility': '{:.1%}',
                'Sharpe Ratio': '{:.2f}',
                'Max Drawdown': '{:.1%}'
            }), use_container_width=True)
            
            passed = screening[screening['Flag'] == 'Pass'].index.tolist()
            flagged = screening[screening['Flag'] != 'Pass'].index.tolist()

            if flagged:
                st.warning(f"Flagged: {', '.join(flagged)}")
                include_flagged = st.multiselect("Would you like to include flagged stocks anyway?", flagged)
                qualified = passed + include_flagged
            else:
                st.success("All tickers passed screening.")
                qualified = passed
        except Exception as e:
                st.error(f"Screening error: {e}")
                qualified = tickers

    st.divider()

    # Step 3: Portfolio Optimizer
    st.header("Portfolio Optimizer")
    st.write("> **Methodology:** Mean-variance optimizer with 3 risk-profiles that have different weight constraints, bootstrapped efficient frontier with 1,000 resamples, computed parametric and historical VaR, detecting fat tails.")

    if len(qualified) < 2:
        st.error("We need at least 2 qualified stocks to optimize, please try again.")
    else:
        with st.spinner(f"Optimizing with a {profile} profile..."):
            try:
                from core.portfolio_optimizer import ( 
                    optimize_portfolio, calculate_var, plot_optimized_portfolio, 
                    plot_var_comparison, boot, plot_bootstrap
                )

                filtered = returns[qualified]
                optimized_returns, sharpe, weights = optimize_portfolio(filtered, rfr, profile)
                var_results = calculate_var(filtered, weights)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
                c2.metric("Daily VaR (95%)", f"{var_results['Parametric VaR']:.2%}")

                st.subheader("Allocation")
                fig_cum = plot_optimized_portfolio(filtered, optimized_returns)
                st.pyplot(fig_cum)
                plt.close()

                tab_risk, tab_boot = st.tabs(["Risk Analysis", "Bootstrapped Frontier"])
                with tab_risk:
                    fig_var = plot_var_comparison(filtered, weights)
                    st.pyplot(fig_var)
                    plt.close()
                    
                with tab_boot:
                    bw, br, bv = boot(filtered, rfr, profile)
                    fig_boot = plot_bootstrap(br, bv, rfr)
                    st.pyplot(fig_boot)
                    plt.close()

            except Exception as e:
                st.error(f"Optimization error: {e}")

    st.divider()

# --- Conclusion Section ---
    st.header("What did I learn?")
    st.info("""
    Predicting the stock market is hard. Literally all of these variables only account for such a small amount of variability in returns
    in the stock market, and just trying to use the free numbers that are available on the internet can only get you so far. I am excited 
    to expand this project to explore more machine learning capabilities and expand the scope of predictors that the model captures to 
    incude political influences, geopolitical events, news headlines, and other factors that economic indicators cannot capture.
    """)
    
    st.caption("Market Intelligence Workbench · Sophomore Project · Data from yfinance & FRED")

else:
    st.info("Enter at least 2 stock tickers and click Run Analysis to get started.")