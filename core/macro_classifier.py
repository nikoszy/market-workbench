#Pull all macro data
#Treasury 10yr, cpi, gdp, unemployment, fed funds rate, spy
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from .data_fetcher import get_fred_data


def get_macro_data(start='2010-01-01', end=none):
    """Fetch and align all macro indicators at monthly frequency."""
    credit_spread = get_fred_data('BAMLH0A0HYM2', start, end)
    treasury_10yr = get_fred_data('DGS10', start, end)
    treasury_2yr = get_fred_data('DGS2', start, end)
    cpi = get_fred_data('CPIAUCSL', start, end)
    unemployment = get_fred_data('UNRATE', start, end)
    fed_funds = get_fred_data('FEDFUNDS', start, end)
    spy = yf.download('SPY', start='2005-01-01', end=none')['Close']
    
    # Resample to monthly frequency
    spy_series = spy.squeeze()
    
    macro = pd.DataFrame({
        '10Y': treasury_10yr.resample('ME').last(),
        '2Y': treasury_2yr.resample('ME').last(),
        'CPI': cpi.resample('ME').last(),
        'Unemployment': unemployment.resample('ME').last(),
        'FedFunds': fed_funds.resample('ME').last(),
        'SPY': spy_series.resample('ME').last(),
        'Credit_Spread': credit_spread.resample('ME').last()
    }).dropna()
    
    return macro


def engineer_features(macro):
    """Create macro features and target variable."""
    macro['Yield Curve'] = macro['10Y'] - macro['2Y']
    macro['CPI_6M_Change'] = macro['CPI'].pct_change(6)
    macro['Unemployment_3M_Accel'] = macro['Unemployment'].diff(3)
    
    # Target: forward 3-month SPY return
    macro['SPY_3M_Return'] = macro['SPY'].pct_change(3).shift(-3)
    macro['Market_Regime'] = macro['SPY_3M_Return'].apply(_label_regime)
    
    return macro


def _label_regime(x):
    """Label market regime based on 3-month return."""
    if pd.isna(x):
        return np.nan
    return 'Expansion' if x > 0 else 'Contraction'


def prepare_training_data(macro):
    """Prepare features and target for model training (best model features)."""
    # Best feature set: Yield Curve, CPI_6M_Change, Credit_Spread
    features = macro[['Yield Curve', 'CPI_6M_Change', 'Credit_Spread']].dropna()
    target = macro['Market_Regime'].dropna()
    
    # Align indices
    valid_idx = features.index.intersection(target.index)
    features = features.loc[valid_idx]
    target = target.loc[valid_idx]
    
    # Encode target
    target_encoded = target.map({'Expansion': 1, 'Contraction': 0})
    
    return features, target_encoded


def train_macro_classifier(features, target, test_size=0.2, random_state=42):
    """Train Random Forest classifier for market regime prediction."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    
    model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, return_proba=False):
    """Evaluate model performance with classification metrics."""
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    if return_proba:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nROC-AUC Score: {auc:.3f}")
        return y_pred, y_prob
    
    return y_pred


def get_feature_importance(model, features):
    """Return feature importance scores."""
    importances = model.feature_importances_
    feature_names = features.columns
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df
