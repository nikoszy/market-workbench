import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
FRED_KEY = os.getenv('FRED_API_KEY')

def get_fred_data(series_id, start = '2005-01-01', end = None):
    '''Fetch data from FRED API for a given series ID and date range.
    Args:
        series_id (str): The FRED series ID.
        start (str): Start date ('YYYY-MM-DD')
        end (str): End date ('YYYY-MM-DD')

    Returns:
        pd.Series: A pandas Series indexed by date with the observed values.
    
    Raises:
        ValueError: If FRED returns no data for the given series. 
    '''
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    url = f'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'api_key': FRED_KEY,
        'file_type': 'json',
        'observation_start': start,
        'observation_end': end
    }
    response = requests.get(url, params=params, verify=False)
    data = response.json()['observations']
    if not data:
        raise ValueError(f"No data returned from FRED for series: {series_id}")
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.set_index('date')['value'].dropna()
def get_stock_data(ticker, start='2021-01-01', end= None):
    '''Fetch historical stock data for a given ticker and date range.
    Args:
        ticker (str): The stock ticker symbol.
        start (str): Start date ('YYYY-MM-DD')
        end (str): End date in ('YYYY-MM-DD')

    Returns:
        pd.DataFrame: A DataFrame with historical stock data indexed by date.
    '''
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        raise ValueError(f"No data returned from yfinance for ticker: {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)  # Drop multi-level column index
    return data 

def get_merged_data(ticker, start='2021-01-01', end=None):
    '''Fetch and merge stock data, treasury yields, and SPY data for regression analysis.
    
    Args:
        ticker (str): The stock ticker symbol.
        start (str): Start date ('YYYY-MM-DD')
        end (str): End date ('YYYY-MM-DD')

    Returns:
        pd.DataFrame: A merged DataFrame containing stock data, treasury yields, SPY data, log returns, and excess returns.
    '''
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    stock = get_stock_data(ticker, start, end)
    treasury = get_fred_data('DGS10', start, end)
    spy = get_stock_data('SPY', start, end)
    
    merged = pd.merge_asof(stock, treasury, left_index=True, right_index=True, direction='backward')
    merged = pd.merge(merged, spy, left_index=True, right_index=True, suffixes=('', '_SPY'))
    merged = merged.rename(columns={'value': '10yr_treasury'})
    # Calculate log returns
    merged['Log_Returns'] = np.log(merged['Close'] / merged['Close'].shift(1))
    merged['Log_Returns_SPY'] = np.log(merged['Close_SPY'] / merged['Close_SPY'].shift(1))
    # Convert annual treasury yield to daily
    merged['10yr_daily'] = (merged['10yr_treasury'] / 100) / 252
    # Calculate excess returns
    merged['Excess_Returns'] = merged['Log_Returns'] - merged['10yr_daily']
    merged['Excess_Returns_SPY'] = merged['Log_Returns_SPY'] - merged['10yr_daily']

    return merged.dropna()

def get_risk_free_rate(start='2021-01-01', end=None) -> float:
    '''Fetch the most recent 10-year treasury yield from FRED and convert it to a daily risk-free rate.
    Args:
        start (str): Start date for fetching treasury data ('YYYY-MM-DD')
        end (str): End date for fetching treasury data ('YYYY-MM-DD')
    Returns:
        float: The most recent 10-year treasury yield converted to a daily risk-free rate (in decimal form).
    '''
    treasury = get_fred_data('DGS10', start, end)
    return float(treasury.iloc[-1] / 100)  # most recent value, converted to decimal