import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
FRED_KEY = os.getenv('FRED_API_KEY')
