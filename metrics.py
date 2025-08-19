
import yfinance as yf
import ta
import pandas as pd



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
    print(obv.rolling(window=period).mean())
    return obv.rolling(window=period).mean()

def SP500(period, interval):
    data = yf.download(tickers="^GSPC", period=period, interval=interval, repair=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)  
    return data["Close"].reset_index(drop=True) 
        

def NASDAQ(period, interval):
    data = yf.download(tickers="^IXIC", period=period, interval=interval)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data["Close"].reset_index(drop=True)
    