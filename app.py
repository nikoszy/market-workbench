import streamlit as st
import pandas as pd
from core.data_fetcher import get_merged_data, get_fred_data
from core.beta_regression import run_beta_regression, run_rolling_beta, plot_residual_diagnostics, plot_quarterly_scatter, plot_realized_volatility
from core.dcf_montecarlo import run_dcf, run_dcf_monte_carlo
from core.portfolio_optimizer import import_portfolio, plot_portfolio, plot_correlation, plot_covariance, screen_stocks, optimize_portfolio, backtest_portfolio
from core.macro_classifier import get_macro_data, engineer_features, prepare_training_data, train_macro_classifier, evaluate_model, get_feature_importance

# Page configuration
st.set_page_config(page_title="Market Workbench", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Analysis", [
    "Beta Regression",
    "DCF Valuation",
    "Portfolio Optimizer",
    "Macro Classifier"
])

# ============================================================================
# Beta Regression Page
# ============================================================================
if page == "Beta Regression":
    st.title("📊 Beta Regression Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Enter stock ticker:", "AAPL")
    with col2:
        benchmark = st.selectbox("Benchmark:", ["SPY", "QQQ", "IWM"])
    
    if st.button("Run Analysis"):
        try:
            with st.spinner("Fetching data..."):
                df = get_merged_data(ticker)
            
            with st.spinner("Running regression..."):
                model, model_nw = run_beta_regression(df)
                rolling_beta = run_rolling_beta(df)
            
            st.subheader("Regression Summary")
            st.text(model_nw.summary())
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Residual Diagnostics")
                st.pyplot(plot_residual_diagnostics(df))
            
            with col2:
                st.subheader("Realized Volatility")
                st.pyplot(plot_realized_volatility(df))
            
            st.subheader("Quarterly Returns Scatter")
            st.pyplot(plot_quarterly_scatter(df, rolling_beta))
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# DCF Valuation Page
# ============================================================================
elif page == "DCF Valuation":
    st.title("💰 DCF Monte Carlo Valuation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Enter stock ticker:", "AAPL")
    with col2:
        projection_years = st.slider("Projection years:", 1, 10, 5)
    with col3:
        num_simulations = st.slider("Simulations:", 1000, 50000, 10000)
    
    if st.button("Run Valuation"):
        try:
            with st.spinner("Running DCF analysis..."):
                dcf_result = run_dcf(ticker, projection_years)
            
            st.subheader("DCF Results")
            st.write(dcf_result)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# Portfolio Optimizer Page
# ============================================================================
elif page == "Portfolio Optimizer":
    st.title("🎯 Portfolio Optimizer")
    
    col1, col2 = st.columns(2)
    with col1:
        tickers_input = st.text_input("Enter tickers (comma-separated):", "AAPL,JPM,XOM,JNJ,AMZN,TLT")
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
    with col2:
        profile = st.selectbox("Risk Profile:", ["Conservative", "Balanced", "Aggressive"])
    
    if st.button("Optimize Portfolio"):
        try:
            with st.spinner("Fetching portfolio data..."):
                returns = import_portfolio(tickers)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Price History")
                st.pyplot(plot_portfolio(returns))
                
                st.subheader("Stock Screening")
                screened = screen_stocks(returns, 0.04)
                st.write(screened)
            
            with col2:
                st.subheader("Correlation Matrix")
                st.pyplot(plot_correlation(returns))
                
                st.subheader("Covariance Matrix")
                st.pyplot(plot_covariance(returns))
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# Macro Classifier Page
# ============================================================================
elif page == "Macro Classifier":
    st.title("🌍 Macro Economic Regime Classifier")
    
    st.markdown("""
    This model predicts whether the market will be in an **Expansion** or **Contraction** 
    regime over the next 3 months based on macroeconomic indicators.
    """)
    
    if st.button("Train & Evaluate Model"):
        try:
            with st.spinner("Loading macro data..."):
                macro = get_macro_data()
                macro = engineer_features(macro)
            
            with st.spinner("Preparing training data..."):
                features, target = prepare_training_data(macro)
            
            with st.spinner("Training model..."):
                model, X_train, X_test, y_train, y_test = train_macro_classifier(features, target)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Evaluation")
                evaluate_model(model, X_test, y_test)
            
            with col2:
                st.subheader("Feature Importance")
                importance = get_feature_importance(model, features)
                st.dataframe(importance)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")