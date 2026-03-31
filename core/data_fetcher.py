import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
FRED_KEY = os.getenv('FRED_API_KEY')

def get_fred_data(series_id, start = '2005-01-01', end = None):
    '''Get data from FRED API for a given series ID and date range.
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
    '''Get historical stock data for a given ticker and date range.
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
    '''Get and merge stock data, treasury yields, and SPY data for easier and smoother analysis.
    '''
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    stock = get_stock_data(ticker, start, end)
    treasury = get_fred_data('DGS10', start, end)
    spy = get_stock_data('SPY', start, end)
    
    merged = pd.merge_asof(stock, treasury, left_index=True, right_index=True, direction='backward')
    merged = pd.merge(merged, spy, left_index=True, right_index=True, suffixes=('', '_SPY'))
    merged = merged.rename(columns={'value': '10yr_treasury'})
    merged['Log_Returns'] = np.log(merged['Close'] / merged['Close'].shift(1))
    merged['Log_Returns_SPY'] = np.log(merged['Close_SPY'] / merged['Close_SPY'].shift(1))
    merged['10yr_daily'] = (merged['10yr_treasury'] / 100) / 252
    merged['Excess_Returns'] = merged['Log_Returns'] - merged['10yr_daily']
    merged['Excess_Returns_SPY'] = merged['Log_Returns_SPY'] - merged['10yr_daily']

    return merged.dropna()

def get_risk_free_rate(start='2021-01-01', end=None) -> float:
    '''Get the most recent 10-year treasury yield from FRED and convert it to a daily risk-free rate.
    '''
    treasury = get_fred_data('DGS10', start, end)
    return float(treasury.iloc[-1] / 100)  # most recent value, converted to decimal