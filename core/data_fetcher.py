import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
FRED_KEY = os.getenv('FRED_API_KEY')

def get_fred_data(series_id, start = '2021-01-01', end = '2026-01-01'):
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
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.set_index('date')['value'].dropna()

def get_merged_data(ticker, start='2021-01-01', end='2026-01-01'):
    stock_data = yf.download(ticker, start=start, end=end)
    treasury_data = get_fred_data('DGS10', start, end)
    spy_data = yf.download('SPY', start=start, end=end)
    
    stock_data.columns = stock_data.columns.droplevel(1)
    spy_data.columns = spy_data.columns.droplevel(1)
    
    merged = pd.merge_asof(stock_data, treasury_data, left_index=True, right_index=True, direction='nearest')
    merged = pd.merge(merged, spy_data, left_index=True, right_index=True, suffixes=('', '_SPY'))
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