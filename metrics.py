
import os   

#Finance
import yfinance as yf

#Visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

#Data and Calculations
import numpy as np
import ta
from ta import trend, momentum, volatility, volume
import pandas as pd
import datetime as dt

#MOVING AVERAGES
'''
Interpreting moving averages (MA, SMA, EMA) is a fundamental skill in technical analysis. While there are many specific strategies, the core interpretations are based on a few key concepts:

1. Identifying Trend Direction
This is the most basic and common use of a moving average.

Uptrend: When the current price is above the moving average, it suggests an uptrend. The longer the price stays above the MA, the stronger the uptrend is considered to be.

Downtrend: When the current price is below the moving average, it suggests a downtrend. Similarly, the longer the price stays below the MA, the stronger the downtrend.

Example: A popular strategy uses the 50-day and 200-day moving averages. If a stock's price is above its 50-day MA, and the 50-day MA is above the 200-day MA, it is generally considered to be in a strong uptrend.

2. Identifying Support and Resistance Levels
Moving averages can act as dynamic support and resistance levels.

Support: In an uptrend, a moving average can act as a support level, where the price may fall to before bouncing back up. Traders often look for buying opportunities when the price pulls back to a key MA (e.g., 50-day or 200-day) and then resumes its upward trend.

Resistance: In a downtrend, a moving average can act as a resistance level, where the price may rise to before being pushed back down. Traders may look for selling opportunities when the price rallies to a key MA and then falls again.

Example: A stock in an uptrend that pulls back and touches its 50-day SMA, then rises again, confirms the 50-day SMA as a support level.

3. Crossover Signals
This is a very popular and powerful way to use moving averages. It involves using two different MAs—a shorter-period one (e.g., 50-day) and a longer-period one (e.g., 200-day).

Golden Cross (Bullish Signal): A bullish signal occurs when the shorter-term MA crosses above the longer-term MA. This suggests that momentum is shifting to the upside and a new uptrend may be starting.

Death Cross (Bearish Signal): A bearish signal occurs when the shorter-term MA crosses below the longer-term MA. This suggests that momentum is shifting to the downside and a new downtrend may be starting.

Example: When the 50-day EMA crosses above the 200-day EMA, it is often interpreted as a bullish signal for a sustained rally.

4. Convergence and Divergence
Convergence: When the price and the MA are moving closer together, it indicates that the current trend may be weakening.

Divergence: When the price and the MA are moving farther apart, it indicates that the current trend is strengthening.

5. Interpreting SMA vs. EMA
The choice of SMA or EMA heavily influences the interpretation:

SMA (Lagging): Because the SMA is slower to react, its signals are often more reliable and less prone to false starts. A crossover using SMAs is generally considered a stronger signal than one using EMAs, as it indicates a more sustained shift in the trend.

EMA (Responsive): The EMA's quick response means it can give you an earlier signal of a potential trend change. However, you must be more cautious with EMA signals, as they can be triggered by short-term price noise and lead to false signals.

Practical Application: Many traders use a combination of both. For instance, they might use a 200-day SMA to determine the long-term trend and a 12-day EMA and 26-day EMA to generate crossover signals for a shorter-term trading strategy.

USE SMA FOR LONG-TERM TRENDS AND EMA FOR SHORT-TERM TRENDS
'''

def MA(df, period=20, ind_type=""):

    if ind_type == 'SMA':
        df['SMA'] = df.rolling(window=period).mean()
        return df['SMA']
    elif ind_type == 'EMA':
        df['EMA'] = df.ewm(span=period, adjust=False).mean()   
        return df['EMA']
    

def MACD(df, short_period=12, long_period=26, signal_period=9):
    """
    Calculate the MACD (Moving Average Convergence Divergence) for a given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Close' column.
        short_period (int): Short-term EMA period.
        long_period (int): Long-term EMA period.
        signal_period (int): Signal line EMA period.
        
    Returns:
        dict: Dictionary with 'macd', 'signal', and 'histogram'.
    """
 
    short_ema = df['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_period, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def RSI(df, period=14):
    
    delta = df.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df['RSI']


def STO(df, period=14):
    """
    Calculate the Stochastic Oscillator for a given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        period (int): Lookback period for the oscillator.
        
    Returns:
        pd.Series: Stochastic Oscillator values.
    """
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    
    df['STO'] = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    
    return df['STO']


def BOB(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands for a given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Close' column.
        period (int): Lookback period for the bands.
        std_dev (int): Number of standard deviations for the bands.
        
    Returns:
        dict: Dictionary with 'upper', 'lower', and 'middle' bands.
    """
    middle = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return {
        'upper': upper,
        'lower': lower,
        'middle': middle
    }


def ADX(df, period=14):
    """
    Calculate the Average Directional Index (ADX) for a given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        period (int): Lookback period for the ADX.
        
    Returns:
        pd.Series: ADX values.
    """
    adx = ta.trend.adx(df['High'], df['Low'], df['Close'], window=period)
    di_plus = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=period)
    di_minus = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=period)
    """
    tr = ta.trend.true_range(df['High'], df['Low'], df['Close'])
    dm_plus = ta.trend.positive_directional_movement(df['High'], df['Low'])
    dm_minus = ta.trend.negative_directional_movement(df['High'], df['Low'])
    di_plus = 100 * (dm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum())
    di_minus = 100 * (dm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum())
    
    dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus))
    
    adx = dx.rolling(window=period).mean()
    """
    return {
        'adx': adx,
        '+DI': di_plus,
        '-DI': di_minus
    }


def ATR(df, period=14):
    """
    Calculate the Average True Range (ATR) for a given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        period (int): Lookback period for the ATR.
        
    Returns:
        pd.Series: ATR values.
    """
    atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
    return atr

def OBV(df, period=20):
    """
    Calculate the On-Balance Volume (OBV) for a given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.
        period (int): Lookback period for the OBV.
        
    Returns:
        pd.Series: OBV values.
    """
    obv = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    return obv.rolling(window=period).mean()
    