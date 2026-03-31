
[streamlit-app-2026-03-31-10-42-01 (1).webm](https://github.com/user-attachments/assets/6af5d99e-289e-44da-8484-5f232ce9f1f8)

Financial Indicator & Macro-Informed Portfolio Analysis
A Research Project by Nikolai Szymkiewicz

About the Project
In one of my statistics classes I completed a project studying which financial indicators had the strongest association with Quarterly Stock Returns and found that Beta had the most significant relationship. This helped me understand that volatility is an important factor in thr market, and I wanted to learn more.

I built this tool during my Sophomore year spring break to explore uses of statistical modeling, machine learning and financial theory hand in hand to come up with conclusions. My goal was to build something that I could actually use to give me actionable insight in my investments, and have a project that I can keep adding to as I develop my machine learning and analysis skills more and more.

What does it do?
The system follows a multi-stage pipeline to evaluate a user-defined portfolio:

Risk Screening: You enter a list of stock tickers, for example a portfolio of 10 different stocks. The system screens your stocks to determine if they might be risky by screening for high volatility, negative sharpe ratio, or a massive drop in value (>|-40%|).

Macro Regime Classification: Using a Random Forest Classifier, classifies predicted Macroeconomic Regime over the next 6 months based on various macro indicators such as 10 year treasury yield, 2 year treasury yield, and credit spread, which were found to be the most important non-related features.

Portfolio Optimization: Using a mean-variance optimizer, runs a portfolio optimization using a risk-profile decided by the macro regime.

Intrinsic Valuation: Finally, runs a DCF simulation valuing each stock on it's free cash flow in terms of today's value.

Methods & Models
This project utilizes several quantitative finance and statistical methods:

Beta Regression: OLS using Newey-West standard errors, Breusch-Pagan for heteroskedasticity testing, computing rolling 60-day beta estimation and adf stationary tests.

DCF Monte Carlo: Ran 10,000 simulations avoiding normal distributions, using truncated normal, beta, and triangular distributions. Built FCF from components (NOPAT + D&A + - Capex - Change in Working Capital).

Portfolio Optimizer: Mean-variance optimizer with 3 risk-profiles that have different weight constraints, bootstrapped efficient frontier with 1,000 resamples, computed parametric and historical VaR, detecting fat tails.

Macro Regime Classifier: Random Forest model trained on CPI momentum, yield curve spread, and credit spread. Train/test split, model had 70% accuracy with AUC = 0.73 and 100% Contraction recall. This means when the model says contraction, you better listen.

Key Learnings
Predicting the stock market is hard. Literally all of these variables only account for such a small amount of variability in returns in the stock market, and just trying to use the free numbers that are available on the internet can only get you so far.

I am excited to expand this project to explore more machine learning capabilities and expand the scope of predictors that the model captures to incude political influences, geopolitical events, news headlines, and other factors that economic indicators cannot capture.

Installation & Usage
Clone the repository.

Install dependencies: pip install -r requirements.txt

Run the application: streamlit run app.py

Note: Data is sourced from yfinance and FRED. This project is for educational and research purposes and does not constitute financial advice.


