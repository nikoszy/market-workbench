#Pull all macro data
#Treasury 10yr, cpi, gdp, unemployment, fed funds rate, spy
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from .data_fetcher import get_fred_data


def get_macro_data(start='2010-01-01', end=None):
    """Fetch and align all macro indicators at monthly frequency.
    
    Args:
        start (str): Start date for fetching data ('YYYY-MM-DD')
        end (str): End date for fetching data ('YYYY-MM-DD')
    Returns:
        pd.DataFrame: A DataFrame containing aligned macro indicators and SPY price.
        """
    credit_spread = get_fred_data('BAMLH0A0HYM2', start, end)
    treasury_10yr = get_fred_data('DGS10', start, end)
    treasury_2yr = get_fred_data('DGS2', start, end)
    cpi = get_fred_data('CPIAUCSL', start, end)
    unemployment = get_fred_data('UNRATE', start, end)
    fed_funds = get_fred_data('FEDFUNDS', start, end)
    spy = yf.download('SPY', start='2005-01-01', end=None)['Close']
    
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
    """Create macro features and target variable.
        - Yield Curve: 10Y - 2Y
        - CPI 6-month percentage change
        - Unemployment 3-month acceleration (difference)
        - Target: forward 3-month SPY return and market regime label
    Args:
        macro (pd.DataFrame): DataFrame containing raw macro indicators and SPY price.
    Returns:
        pd.DataFrame: DataFrame with engineered features and target variable.
    """
    macro['Yield Curve'] = macro['10Y'] - macro['2Y']
    macro['CPI_6M_Change'] = macro['CPI'].pct_change(6)
    macro['Unemployment_3M_Accel'] = macro['Unemployment'].diff(3)
    
    # Target: forward 3-month SPY return
    macro['SPY_3M_Return'] = macro['SPY'].pct_change(3).shift(-3)
    macro['Market_Regime'] = macro['SPY_3M_Return'].apply(_label_regime)
    
    return macro


def _label_regime(x):
    """Label market regime based on 3-month return.
    Args:
        x (float): The forward 3-month return of SPY.
    Returns:
        str: 'Expansion' if return > 0, 'Contraction' if return <= 0, or NaN if input is NaN.
        """
    if pd.isna(x):
        return np.nan
    return 'Expansion' if x > 0 else 'Contraction'


def prepare_training_data(macro):
    """Prepare features and target for model training (best model features).
    
    Args:
        macro (pd.DataFrame): DataFrame containing engineered features and target variable.
    Returns:
        tuple: (features DataFrame, target Series) ready for model training.
    """
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


def train_macro_classifier(features, target):
    '''Train a Random Forest classifier to predict market regime.
    
    Args:
        features (pd.DataFrame): DataFrame containing feature columns.
        target (pd.Series): Series containing encoded target variable.
    Returns:
        tuple: (trained model, X_train, X_test, y_train, y_test)
        '''
    split = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = target.iloc[:split], target.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=200, max_depth =4, 
        class_weight='balanced', random_state=42
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with classification metrics.
    
    Args:
        model: Trained classifier model.
        X_test: Test features.
        y_test: True labels for test set.
        return_proba: If True, also return predicted probabilities.
    Returns:
        dict: A dictionary containing accuracy, precision, recall, AUC, confusion matrix, and optionally probabilities.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': report['accuracy'],
        'contraction_precision': report['0']['precision'],
        'contraction_recall': report['0']['recall'],
        'expansion_precision': report['1']['precision'],
        'expansion_recall': report['1']['recall'],
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_prob
    }

def get_feature_importance(model, features):
    """Return feature importance scores.
    
    Args:
        model: Trained classifier model with feature_importances_ attribute.
        features: DataFrame containing the feature columns used for training.
    Returns:
        pd.DataFrame: A DataFrame with feature names and their corresponding importance scores.
    """
    importances = model.feature_importances_
    feature_names = features.columns
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df

def predict_current_regime(model, latest_macro):
    """Predict current market regime based on latest macro indicators.
    
    Args:
        model: Trained classifier model.
        latest_macro: DataFrame containing the latest macro indicators (must include features used for training).
    Returns:
        dict: A dictionary containing the predicted regime, confidence score, probabilities for each class, and the date of the latest data.
    """
    features = latest_macro[['Yield Curve', 'CPI_6M_Change', 'Credit_Spread']].dropna()
    latest = features.iloc[[-1]]


    prediction = model.predict(latest)[0]
    probability = model.predict_proba(latest)[0]

    regime = 'Expansion' if prediction == 1 else 'Contraction'
    confidence = max(probability)

    return {
        'regime': regime,
        'confidence': confidence,
        'expansion_prob': probability[1],
        'contraction_prob': probability[0],
        'as_of': latest.index[0].strftime('%Y-%m-%d')
    }

def plot_feature_importance(model, features):
    '''Plot feature importance for the trained macro classifier.
    '''
    importance = get_feature_importance(model, features)
    plt.figure(figsize=(8, 5))
    plt.bar(importance['feature'], importance['importance'], color='skyblue')
    plt.title('Feature Importance for Macro Classifier')
    plt.xlabel('Feature')   
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(eval_results, y_test):
    '''Plot the ROC curve for the macro classifier.
    '''
    fpr, tpr, _ = roc_curve(y_test, eval_results['probabilities'])
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {eval_results["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.title('ROC Curve for Macro Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    return plt.gcf()

def get_classifier_summary():
    macro = get_macro_data()
    macro = engineer_features(macro)
    features, target = prepare_training_data(macro)
    model, X_train, X_test, y_train, y_test = train_macro_classifier(features, target)
    eval_results = evaluate_model(model, X_test, y_test)
    current = predict_current_regime(model, macro)
    importance = get_feature_importance(model, features)
    return {
        'model': model,
        'macro_data': macro,
        'evaluation': eval_results,
        'current_prediction': current,
        'feature_importance': importance
    }
