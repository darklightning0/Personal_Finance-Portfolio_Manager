from contextlib import contextmanager
import datetime as dt
import asyncio
import os
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import python_weather as pw
import time
from transformers import pipeline
from collections import deque
from typing import Callable, Any
import metrics as met
from datetime import datetime, timedelta
import numpy as np
import plotly.colors as p_colors
import sqlite3

try:
    from zoneinfo import ZoneInfo 
except ImportError:
    from pytz import timezone as ZoneInfo 

FINNHUB_API_KEY = st.secrets.get("finnhub_api_key")
SYMBOL_CACHE_FILE = "data/symbol_cache.csv"
COLOR_PALETTE = ['#ffd700', '#ff4081', '#7c4dff', '#00bcd4', '#00e676', '#ffb300', '#e040fb', '#29b6f6', '#ff1744', '#00e5ff']

DATABASE_FILE = "data/trading_portfolio.db"

class DatabaseManager:
    def __init__(self, db_path=DATABASE_FILE):
        self.db_path = db_path
        self.ensure_data_directory()
        self.initialize_database()
    
    def ensure_data_directory(self):
        """Ensure the data directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def initialize_database(self):
        """Create tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Portfolio table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY,
                    value REAL DEFAULT 0,
                    balance REAL DEFAULT 35000,
                    yesterday_value REAL DEFAULT 0,
                    daily_change REAL DEFAULT 0,
                    yesterday_balance REAL DEFAULT 35000,
                    last_update_date TEXT DEFAULT (date('now'))
                )
            ''')
            
            # Holdings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    buyprice REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    current_price REAL
                )
            ''')
            
            # Transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    buyprice REAL,
                    buy_timestamp TEXT,
                    sold BOOLEAN DEFAULT 0,
                    sellprice REAL,
                    sell_timestamp TEXT
                )
            ''')
            
            # Initialize portfolio if empty
            cursor.execute('SELECT COUNT(*) FROM portfolio')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO portfolio (value, balance, yesterday_value, daily_change, yesterday_balance, last_update_date)
                    VALUES (0, 35000, 0, 0, 35000, date('now'))
                ''')
            
            conn.commit()

db_manager = DatabaseManager()

def api_call_rate_monitor(func: Callable) -> Callable:
    """
    A decorator that monitors and reports the rate of function calls.
    It stores timestamps and can calculate the rate per minute or second.
    """
    # The deque stores timestamps of calls within the last 60 seconds.
    timestamps = deque()

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Record the current time
        current_time = time.time()
        timestamps.append(current_time)

        # Remove all timestamps that are older than 60 seconds.
        # This ensures the deque only contains calls from the last minute.
        while timestamps and timestamps[0] <= current_time - 60:
            timestamps.popleft()

        # The number of calls in the deque is now the rate per minute.
        calls_last_minute = len(timestamps)
        
        # Calculate the average rate per second based on the minute's total.
        rate_per_sec = calls_last_minute / 60.0
        
        # Print the updated rates
        print(f"API calls per minute: {calls_last_minute} | Calls per second: {rate_per_sec:.2f}")

        return func(*args, **kwargs)

    return wrapper

def get_weather_emoji(description):
    desc = description.lower()
    if "sun" in desc or "clear" in desc:
        return "☀️"
    if "cloud" in desc or "overcast" in desc:
        return "☁️"
    if "rain" in desc or "shower" in desc or "drizzle" in desc:
        return "🌧️"
    if "storm" in desc or "thunder" in desc:
        return "⛈️"
    if "snow" in desc:
        return "❄️"
    if "fog" in desc or "mist" in desc:
        return "🌫️"
    return "🌡️"

@st.cache_data(ttl=600)
def get_weather_sync(location):
    async def fetch():
        async with pw.Client(unit=pw.METRIC) as client:
            weather = await client.get(location)
            return float(weather.temperature), str(weather.description)
    return asyncio.run(fetch())

@st.cache_data(ttl= 18000)
def get_date(past_days = 0):
    date = dt.date.today() - dt.timedelta(days=past_days)
    return date.isoformat()

def round_digit(value, digits):
    if value == 'N/A' or value is None:
        return 'N/A'
    try:
        return round(float(value), digits)
    except (ValueError, TypeError):
        return 'N/A'

def get_stock_data(url):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.warning(f"API request failed: {e}")
        return None

def get_stock_quote(symbol):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    return get_stock_data(url)
@st.cache_data(ttl=86400)
def get_stock_profile(symbol):
    url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={FINNHUB_API_KEY}"
    return get_stock_data(url)
@st.cache_data(ttl=86400)
def get_stock_lookup(symbol):
    url = f"https://finnhub.io/api/v1/search?q={symbol}&token={FINNHUB_API_KEY}"
    return get_stock_data(url)
@st.cache_data(ttl=3600)
def get_stock_metrics(symbol):
    url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={FINNHUB_API_KEY}"
    data = get_stock_data(url)
    return data.get("metric", {}) if data else {}

@st.cache_data(ttl=5)
def get_last_price(symbol):
    quote = get_stock_quote(symbol)
    if quote and 'c' in quote:
        c = round_digit(quote['c'], 2)
        holdings_df, port_df, _ = load_data()
        for idx, row in holdings_df.iterrows():
            if row["ticker"] == symbol:
                holdings_df.loc[idx, "current_price"] = c
        save_data(holdings_df, port_df, _)
        return c
    return "N/A"


@st.cache_data(ttl=300)
def is_market_open(exchange='US'):
    try:
        url = f"https://finnhub.io/api/v1/stock/market-status?exchange={exchange}&token={FINNHUB_API_KEY}"
        dat = get_stock_data(url)

        if dat.get("isOpen") == False and dat.get("holiday"):
            return f"the market is closed due to {dat.get('holiday')}."
        elif dat.get("isOpen") == False and not dat.get("holiday"):
            return "True"
        else:
            return "True"
    except Exception as e:
        st.warning(f"Could not determine market open status: {e}")
        return "market status unknown"

def fetch_symbols():
    url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list):
            st.error(f"Finnhub symbol API error: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty or not all(col in df.columns for col in ['symbol', 'displaySymbol', 'description', 'type']):
            st.error("Finnhub symbol data missing expected columns.")
            return pd.DataFrame()

        df = df[df['type'] == 'Common Stock'][['symbol', 'displaySymbol', 'description']]
        df.to_csv(SYMBOL_CACHE_FILE, index=False)
        return df

    except Exception as e:
        st.error(f"Failed to fetch symbols: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_symbol_list():
    if os.path.exists(SYMBOL_CACHE_FILE):
        try:
            return pd.read_csv(SYMBOL_CACHE_FILE)
        except Exception as e:
            st.warning(f"Could not read symbol cache: {e}. Refetching...")
    return fetch_symbols()

def search_component(symbols):
    query = st.text_input("Search for a stock...", placeholder="Type a ticker or company name", key="search_query")
    
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = None
    
    if not query:
        st.session_state.selected_symbol = None
        return None
    
    exact_match_ticker = symbols[symbols['symbol'].str.lower() == query.lower()]
    exact_match_name = symbols[symbols['description'].str.lower() == query.lower()]
    
    if not exact_match_ticker.empty:
        st.session_state.selected_symbol = exact_match_ticker.iloc[0]['symbol']
        return st.session_state.selected_symbol
    
    if not exact_match_name.empty:
        st.session_state.selected_symbol = exact_match_name.iloc[0]['symbol']
        return st.session_state.selected_symbol
    
    matches = symbols[symbols['symbol'].str.contains(query, case=False) |
                      symbols['description'].str.contains(query, case=False)].head(5)
    
    if not matches.empty:
        suggestions = [f"{row['symbol']} - {row['description']}" for _, row in matches.iterrows()]
        selected = st.selectbox("Results", suggestions, index=None, placeholder="Select a stock from the results")
        if selected:
            symbol = selected.split(" - ")[0]
            st.session_state.selected_symbol = symbol
            return symbol
    
    st.session_state.selected_symbol = None
    return None


def load_data():
    try:
        with db_manager.get_connection() as conn:
            # Load holdings
            holdings_df = pd.read_sql_query('SELECT * FROM holdings', conn)
            
            # Load portfolio
            portfolio_df = pd.read_sql_query('SELECT * FROM portfolio', conn)
            
            # Load transactions
            transactions_df = pd.read_sql_query('SELECT * FROM transactions', conn)
            
            return holdings_df, portfolio_df, transactions_df
    
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        # Return empty DataFrames as fallback
        return (
            pd.DataFrame(columns=["ticker", "quantity", "buyprice", "timestamp", "current_price"]),
            pd.DataFrame(columns=["value", "balance", "yesterday_value", "daily_change", "yesterday_balance", "last_update_date"]),
            pd.DataFrame(columns=["ticker", "quantity", "buyprice", "buy_timestamp", "sold", "sellprice", "sell_timestamp"])
        )

def save_data(holdings_df, portfolio_df, transactions_df):
    """Save data to SQL database (replaces the original save_data function)"""
    try:
        with db_manager.get_connection() as conn:
            # Clear and insert holdings
            conn.execute('DELETE FROM holdings')
            if not holdings_df.empty:
                holdings_df.to_sql('holdings', conn, if_exists='append', index=False)
            
            # Clear and insert portfolio
            conn.execute('DELETE FROM portfolio')
            if not portfolio_df.empty:
                portfolio_df.to_sql('portfolio', conn, if_exists='append', index=False)
            
            # Clear and insert transactions
            conn.execute('DELETE FROM transactions')
            if not transactions_df.empty:     
                transactions_df.to_sql('transactions', conn, if_exists='append', index=False)
            conn.commit()
    
    except Exception as e:
        st.error(f"Error saving data to database: {e}")

def backup_database():
    """Create a backup of the database"""
    try:
        import shutil
        from datetime import datetime
        
        backup_name = f"data/trading_portfolio_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2(DATABASE_FILE, backup_name)
        return backup_name
    except Exception as e:
        st.error(f"Error creating backup: {e}")
        return None

def get_stock_position(symbol):

    try:
        with db_manager.get_connection() as conn:
            query = '''
                SELECT SUM(quantity) as total_quantity, 
                       SUM(quantity * buyprice) / SUM(quantity) as avg_price
                FROM holdings 
                WHERE ticker = ?
            '''
            result = conn.execute(query, (symbol,)).fetchone()
            
            if result and result[0]:
                return result[0], result[1]
            return 0, 0
    except Exception as e:
        st.error(f"Error getting stock position: {e}")
        return 0, 0

def get_portfolio_value():

    try:
        holdings_df, portfolio_df, _ = load_data()
        
        value = 0
        if not holdings_df.empty:
            for _, row in holdings_df.iterrows():
                current_price = get_last_price(row["ticker"])
                if current_price != "N/A":
                    value += current_price * row["quantity"]
        
        # Update portfolio value in database
        with db_manager.get_connection() as conn:
            conn.execute('UPDATE portfolio SET value = ? WHERE id = 1', (value,))
            conn.commit()
        
        return value
    except Exception as e:
        st.error(f"Error calculating portfolio value: {e}")
        return 0

@st.cache_data(ttl=60)
def get_daily_change(str_type="value"):
    """Get daily change using SQL (replaces get_daily_change)"""
    try:
        with db_manager.get_connection() as conn:
            if str_type == "value":
                result = conn.execute('''
                    SELECT value - yesterday_value as daily_change 
                    FROM portfolio WHERE id = 1
                ''').fetchone()
                
                if result:
                    change = result[0]
                    # Update the daily_change in database
                    conn.execute('UPDATE portfolio SET daily_change = ? WHERE id = 1', (change,))
                    conn.commit()
                    return change
            elif str_type == "balance":
                result = conn.execute('''
                    SELECT balance - yesterday_balance as balance_change 
                    FROM portfolio WHERE id = 1
                ''').fetchone()
                return result[0] if result else 0
        
        return 0
    except Exception as e:
        st.error(f"Error calculating daily change: {e}")
        return 0

@st.cache_data(ttl=300)
def get_alltime_change(str_type="net"):
    """Get all-time change using SQL (replaces get_alltime_change)"""
    try:
        with db_manager.get_connection() as conn:
            # Calculate total buy value
            buy_value_result = conn.execute('''
                SELECT SUM(buyprice * quantity) as total_buy_value 
                FROM holdings
            ''').fetchone()
            
            buy_value = buy_value_result[0] if buy_value_result[0] else 0
            
            # Get current portfolio value
            portfolio_result = conn.execute('SELECT value FROM portfolio WHERE id = 1').fetchone()
            current_value = portfolio_result[0] if portfolio_result else 0
            
            change = current_value - buy_value
            
            if str_type == "net":
                return change
            elif str_type == "percent" and buy_value > 0:
                return (change / buy_value) * 100
            else:
                return 0
    except Exception as e:
        st.error(f"Error calculating all-time change: {e}")
        return 0
    
def get_holdings_summary():
    """Get holdings summary with aggregated data"""
    try:
        with db_manager.get_connection() as conn:
            query = '''
                SELECT 
                    ticker,
                    SUM(quantity) as total_quantity,
                    SUM(quantity * buyprice) / SUM(quantity) as avg_buy_price,
                    MAX(current_price) as current_price
                FROM holdings 
                GROUP BY ticker
                HAVING SUM(quantity) > 0
                ORDER BY ticker
            '''
            
            result = conn.execute(query).fetchall()
            
            if result:
                columns = ['ticker', 'total_quantity', 'avg_buy_price', 'current_price']
                return pd.DataFrame(result, columns=columns)
            
            return pd.DataFrame(columns=columns)
    except Exception as e:
        st.error(f"Error getting holdings summary: {e}")
        return pd.DataFrame()
    
def get_transaction_history(symbol=None, limit=100):
    """Get transaction history with optional symbol filter"""
    try:
        with db_manager.get_connection() as conn:
            if symbol:
                query = '''
                    SELECT * FROM transactions 
                    WHERE ticker = ? 
                    ORDER BY 
                        COALESCE(sell_timestamp, buy_timestamp) DESC 
                    LIMIT ?
                '''
                params = (symbol, limit)
            else:
                query = '''
                    SELECT * FROM transactions 
                    ORDER BY 
                        COALESCE(sell_timestamp, buy_timestamp) DESC 
                    LIMIT ?
                '''
                params = (limit,)
            
            return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.error(f"Error getting transaction history: {e}")
        return pd.DataFrame()
    
def update_daily_portfolio():
    """Update daily portfolio values (replaces relevant parts of initialize_files)"""
    try:
        with db_manager.get_connection() as conn:
            # Check if we need to update daily values
            result = conn.execute('''
                SELECT last_update_date, value, balance 
                FROM portfolio WHERE id = 1
            ''').fetchone()
            
            if result:
                last_update_str, current_value, current_balance = result
                last_update_date = dt.date.fromisoformat(last_update_str)
                today = dt.date.today()
                
                if today > last_update_date:
                    conn.execute('''
                        UPDATE portfolio 
                        SET yesterday_value = ?, 
                            yesterday_balance = ?, 
                            last_update_date = ?
                        WHERE id = 1
                    ''', (current_value, current_balance, today.isoformat()))
                    conn.commit()
    except Exception as e:
        st.error(f"Error updating daily portfolio: {e}")

def buy_stock(symbol, quantity):
    """Updated buy_stock function using SQL"""
    market_open = is_market_open()
    if market_open != "True":
        st.toast(f"Cannot execute this order, because {market_open}", icon='🚨')
        return

    holdings_df, portfolio_df, transactions_df = load_data()
    
    current_price = get_last_price(symbol)
    if current_price == "N/A":
        st.toast("Could not get current price for this stock!", icon='🚨')
        return
        
    cost = current_price * quantity
    balance = portfolio_df["balance"].iloc[0]
    
    if balance >= cost:
        import datetime as dt
        timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add to holdings
        new_holding = pd.DataFrame([{
            'ticker': symbol, 
            'quantity': quantity, 
            'buyprice': current_price,
            'timestamp': timestamp,
            'current_price': current_price
        }])
        holdings_df = pd.concat([holdings_df, new_holding], ignore_index=True)

        # Add to transactions
        new_transaction = pd.DataFrame([{
            'ticker': symbol, 
            'quantity': quantity, 
            'buyprice': current_price,
            'buy_timestamp': timestamp,
            'sold': 0,  # Use 0 for False in SQL
            'sellprice': None,
            'sell_timestamp': None
        }])
        transactions_df = pd.concat([transactions_df, new_transaction], ignore_index=True)
        
        # Update balance
        new_balance = round(balance - cost, 2)
        portfolio_df.loc[0, "balance"] = new_balance
        
        save_data(holdings_df, portfolio_df, transactions_df)
        
        st.session_state.toast_message = f"Bought {quantity} shares of {symbol} for {cost:.2f}! New balance: {new_balance:.2f}"
        st.session_state.toast_type = "success"
    else:
        st.session_state.toast_message = f"Insufficient funds! Need {cost:.2f}, have {balance:.2f}"
        st.session_state.toast_type = "error"


def sell_stock(symbol, quantity):
    """Updated sell_stock function using SQL"""
    market_open = is_market_open()
    if market_open != "True":
        st.session_state.toast_message = f"Cannot execute this order, because {market_open}"
        st.session_state.toast_type = "error"
        return
    
    holdings_df, portfolio_df, transactions_df = load_data()
    
    stock_holdings = holdings_df[holdings_df['ticker'] == symbol].copy()
    
    if stock_holdings.empty:
        st.session_state.toast_message = f"You don't own any shares of {symbol}!"
        st.session_state.toast_type = "error"
        return
    
    available_quantity = stock_holdings['quantity'].sum()
    
    if available_quantity < quantity:
        st.session_state.toast_message = f"You only have {int(available_quantity)} shares of {symbol}!"
        st.session_state.toast_type = "error"
        return
    
    current_price = get_last_price(symbol)
    if current_price == "N/A":
        st.session_state.toast_message = "Could not get current price for this stock!"
        st.session_state.toast_type = "error"
        return
        
    total_revenue = current_price * quantity
    sell_timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Sort holdings by timestamp (FIFO)
    if 'timestamp' in stock_holdings.columns:
        stock_holdings = stock_holdings.sort_values('timestamp')
    else:
        stock_holdings = stock_holdings.sort_index() 
    
    remaining_to_sell = quantity
    indices_to_remove = []
    total_cost_basis = 0
    
    # Process FIFO selling
    for idx, row in stock_holdings.iterrows():
        if remaining_to_sell <= 0:
            break
            
        if row['quantity'] <= remaining_to_sell:
            shares_sold = row['quantity']
            remaining_to_sell -= shares_sold
            indices_to_remove.append(idx)
            total_cost_basis += shares_sold * row['buyprice']
            
            # Update corresponding transaction
            mask = (
                (transactions_df['ticker'] == symbol) &
                (transactions_df['buyprice'] == row['buyprice']) &
                (transactions_df['quantity'] == row['quantity']) &
                (transactions_df['sold'] == 0)
            )
            matching_indices = transactions_df[mask].index
            if len(matching_indices) > 0:
                transactions_df.loc[matching_indices[0], 'sold'] = 1
                transactions_df.loc[matching_indices[0], 'sellprice'] = current_price
                transactions_df.loc[matching_indices[0], 'sell_timestamp'] = sell_timestamp
        else:
            shares_sold = remaining_to_sell
            holdings_df.loc[idx, 'quantity'] -= remaining_to_sell
            total_cost_basis += shares_sold * row['buyprice']
            remaining_to_sell = 0
            
            # Handle partial sale in transactions
            mask = (
                (transactions_df['ticker'] == symbol) &
                (transactions_df['buyprice'] == row['buyprice']) &
                (transactions_df['quantity'] >= shares_sold) &
                (transactions_df['sold'] == 0)
            )
            matching_indices = transactions_df[mask].index
            
            if len(matching_indices) > 0:
                original_idx = matching_indices[0]
                original_qty = transactions_df.loc[original_idx, 'quantity']
                
                # Create sold transaction for partial amount
                sold_transaction = transactions_df.loc[original_idx].copy()
                sold_transaction['quantity'] = shares_sold
                sold_transaction['sold'] = 1
                sold_transaction['sellprice'] = current_price
                sold_transaction['sell_timestamp'] = sell_timestamp
                
                # Reduce original transaction quantity
                transactions_df.loc[original_idx, 'quantity'] -= shares_sold
                
                # Add the sold transaction
                transactions_df = pd.concat([
                    transactions_df, 
                    pd.DataFrame([sold_transaction])
                ], ignore_index=True)
    
    # Remove fully sold holdings
    if indices_to_remove:
        holdings_df = holdings_df.drop(indices_to_remove).reset_index(drop=True)

    # Add sell-only transaction record
    sell_only_transaction = pd.DataFrame([{
        'ticker': symbol,
        'quantity': quantity,
        'buyprice': None, 
        'buy_timestamp': None,  
        'sold': 1,
        'sellprice': current_price,
        'sell_timestamp': sell_timestamp
    }])
    transactions_df = pd.concat([transactions_df, sell_only_transaction], ignore_index=True)

    # Update balance
    current_balance = portfolio_df["balance"].iloc[0]
    new_balance = round(current_balance + total_revenue, 2)
    portfolio_df.loc[0, "balance"] = new_balance

    save_data(holdings_df, portfolio_df, transactions_df)

    profit_loss = total_revenue - total_cost_basis
    pl_emoji = "📈" if profit_loss >= 0 else "📉"
    
    st.session_state.toast_message = f"Sold {quantity} shares of {symbol} for {total_revenue:.2f}! {pl_emoji} P&L: {profit_loss:.2f} | Balance: {new_balance:.2f}"
    st.session_state.toast_type = "success"

def display_toast_messages():
    if 'toast_message' in st.session_state:
        if st.session_state.toast_type == "success":
            st.toast(st.session_state.toast_message, icon="✅")
        else:
            st.toast(st.session_state.toast_message, icon="🚨")
        del st.session_state.toast_message
        del st.session_state.toast_type

def create_trading_interface(symbol=None, holdings_df=None):
    """
    Creates a unified trading interface for buying and selling stocks.
    Handles both single symbol entry (watchlist) and holdings dataframe (portfolio).
    
    Args:
        symbol: Stock symbol to trade (for watchlist) or None (for portfolio)
        holdings_df: DataFrame of holdings (for portfolio page) or None (for watchlist)
    """
    display_toast_messages()
    
    # Determine if we're in portfolio mode (has holdings_df) or watchlist mode (has symbol)
    is_portfolio_mode = holdings_df is not None and not holdings_df.empty
    
    # Handle symbol selection
    if is_portfolio_mode:
        # Portfolio: Select from existing holdings
        holding_symbols = holdings_df['ticker'].unique().tolist()
        selected_symbol = st.selectbox(
            "Select a holding to trade:",
            options=holding_symbols,
            index=0,
            key="selected_holding_symbol"
        )
    else:
        # Watchlist: Use provided symbol
        selected_symbol = symbol
    
    if selected_symbol:
        # Get holdings information
        if is_portfolio_mode:
            # Portfolio mode: use provided holdings_df
            selected_holdings = holdings_df[holdings_df['ticker'] == selected_symbol].copy()
            total_quantity = selected_holdings['quantity'].sum()
            
            # Calculate average buy price
            if total_quantity > 0:
                avg_buy_price = (selected_holdings['quantity'] * selected_holdings['buyprice']).sum() / total_quantity
            else:
                avg_buy_price = 0
        else:
            # Watchlist mode: load data to check for existing holdings
            existing_holdings_df, _, _ = load_data()
            if not existing_holdings_df.empty:
                existing_holdings = existing_holdings_df[existing_holdings_df['ticker'] == selected_symbol].copy()
                total_quantity = existing_holdings['quantity'].sum() if not existing_holdings.empty else 0
                avg_buy_price = 0
                if total_quantity > 0:
                    avg_buy_price = (existing_holdings['quantity'] * existing_holdings['buyprice']).sum() / total_quantity
            else:
                total_quantity = 0
                avg_buy_price = 0
        
        current_price = get_last_price(selected_symbol)
        
        # Display stock info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Symbol", selected_symbol)
        col2.metric("Owned Shares", f"{int(total_quantity)}")
        if total_quantity > 0:
            col3.metric("Avg Buy Price", f"${avg_buy_price:.2f}")
        else:
            col3.metric("Avg Buy Price", "N/A")
        col4.metric("Current Price", f"${current_price:.2f}" if current_price != "N/A" else "N/A")
        
        # Trading interface
        with st.container(border=True):
            if is_portfolio_mode:
                st.markdown("### Trade Selected Holding")
            else:
                st.markdown("### Trade Stock")
            
            trade_col1, trade_col2 = st.columns(2)
            
            with trade_col1:
                st.markdown("#### Buy Shares")
                buy_quantity = st.number_input(
                    "Shares to buy:", 
                    min_value=1, 
                    value=10, 
                    key=f"buy_quantity_{selected_symbol}"
                )
                
                if current_price != "N/A":
                    buy_cost = buy_quantity * current_price
                    st.metric("Total Cost", f"${buy_cost:.2f}")
                    
                    with stylable_container(
                        "green_button",
                        css_styles="""
                        button {
                            background-color: #00C851;
                            color: white;
                            border: none;
                            border-radius: 5px;
                        }
                        button:hover {
                            background-color: #007E33;
                        }
                        """
                    ):
                        market_open = is_market_open() == "True"
                        if st.button(
                            f"Buy {buy_quantity} shares of {selected_symbol}", 
                            key=f"buy_button_{selected_symbol}",
                            disabled=not market_open,
                            use_container_width=True
                        ):
                            buy_stock(selected_symbol, buy_quantity)
                            st.rerun()
            
            with trade_col2:
                st.markdown("#### Sell Shares")
                if total_quantity > 0:
                    max_sell = int(total_quantity)
                    sell_quantity = st.number_input(
                        f"Shares to sell (max {max_sell}):", 
                        min_value=1, 
                        max_value=max_sell, 
                        value=min(10, max_sell), 
                        key=f"sell_quantity_{selected_symbol}"
                    )
                    
                    if current_price != "N/A":
                        sell_revenue = sell_quantity * current_price
                        st.metric("Total Revenue", f"${sell_revenue:.2f}")
                        
                        with stylable_container(
                            "red_button",
                            css_styles="""
                            button {
                                background-color: #ff4444;
                                color: white;
                                border: none;
                                border-radius: 5px;
                            }
                            button:hover {
                                background-color: #cc0000;
                            }
                            """
                        ):
                            market_open = is_market_open() == "True"
                            if st.button(
                                f"Sell {sell_quantity} shares of {selected_symbol}", 
                                key=f"sell_button_{selected_symbol}",
                                disabled=not market_open,
                                use_container_width=True
                            ):
                                # Use appropriate sell function based on mode
                                if is_portfolio_mode:
                                    sell_stock_from_holding(selected_symbol, sell_quantity)
                                else:
                                    sell_stock(selected_symbol, sell_quantity)
                                st.rerun()
                else:
                    st.info("No shares to sell")
        
        # Show individual holdings for the selected symbol (if multiple purchases)
        if is_portfolio_mode and total_quantity > 0:
            selected_holdings = holdings_df[holdings_df['ticker'] == selected_symbol].copy()
            if len(selected_holdings) > 1:
                with st.expander(f"Individual {selected_symbol} Holdings", expanded=False):
                    individual_holdings = []
                    for idx, row in selected_holdings.iterrows():
                        individual_holdings.append({
                            "Quantity": int(row["quantity"]),
                            "Buy Price": f"${row['buyprice']:.2f}",
                            "Purchase Date": row.get("timestamp", "N/A"),
                            "Current Value": f"${row['quantity'] * current_price:.2f}" if current_price != "N/A" else "N/A"
                        })
                    
                    individual_df = pd.DataFrame(individual_holdings)
                    st.dataframe(individual_df, use_container_width=True, hide_index=True)

def create_info_display(symbol, profile, quote, metrics, lookup_data):
    cols = st.columns([1, 3, 1, 1, 1, 1])
    
    try:
        if profile.get('logo'):
            cols[0].image(image=f"{profile.get('logo')}", output_format="PNG", width=100)
    except:
        cols[0].write("📊")

    cols[1].markdown(f"### {profile.get('name', symbol)} ({symbol})")
    
    with st.expander(label="Info", expanded=True):
        cols = st.columns([2, 2, 2, 2])
        cols[0].markdown(f"**Sector:**<br><span style='font-size: 1.5em; color:#ffe300;'>{profile.get('finnhubIndustry', 'N/A')}</span>", unsafe_allow_html=True)
        cols[1].markdown(f"**Stock Symbol:**<br><span style='font-size: 1.5em; color:#ffe300;'>{lookup_data.get('type', 'N/A')}</span>", unsafe_allow_html=True)
        cols[2].markdown(f"**Exchange:**<br><span style='font-size: 1.5em; color:#ffe300;'>{profile.get('exchange', 'N/A')}</span>", unsafe_allow_html=True)
        cols[3].markdown(f"**Market Cap (M):**<br><span style='font-size: 1.5em; color:#ffe300;'>${metrics.get('marketCapitalization', 'N/A')}</span>", unsafe_allow_html=True)
        
        cols = st.columns([2, 2, 2, 2])
        cols[0].metric("**Last Price:**", f"${round_digit(quote.get('c', 'N/A'), 2)}", delta=f"{round_digit(quote.get('d', 'N/A'), 2)} ({round_digit(quote.get('dp', 'N/A'), 2)}%)", delta_color="normal")
        cols[1].metric("**Open Price:**", f"${quote.get('o', 'N/A')}")
        cols[2].metric("**Highest Price:**", f"${quote.get('h', 'N/A')}")
        cols[3].metric("**Lowest Price:**", f"${quote.get('l', 'N/A')}")

        with st.expander("Trade", expanded=True):
            create_trading_interface(symbol=symbol)

def get_metric_sentiment(label, value):
    """
    Determine if a metric is bullish (green), bearish (red), or neutral (yellow)
    Returns: ('bullish'/'bearish'/'neutral', color_hex)
    """
    if value == 'N/A' or value is None:
        return ('neutral', '#FFA500')  # Orange for N/A values
    
    try:
        # Convert percentage strings to float
        if isinstance(value, str) and '%' in value:
            numeric_value = float(value.replace('%', ''))
        else:
            numeric_value = float(value)
    except (ValueError, TypeError):
        return ('neutral', '#FFA500')
    
    # Valuation Metrics (lower is generally better)
    if "P/E Ratio" in label:
        if numeric_value < 15:
            return ('bullish', '#00FF00')
        elif numeric_value > 30:
            return ('bearish', '#FF0000')
        else:
            return ('neutral', '#FFFF00')
    
    elif "PEG Ratio" in label:
        if numeric_value < 1:
            return ('bullish', '#00FF00')
        elif numeric_value > 2:
            return ('bearish', '#FF0000')
        else:
            return ('neutral', '#FFFF00')
    
    elif "P/B Ratio" in label:
        if numeric_value < 3:
            return ('bullish', '#00FF00')
        elif numeric_value > 10:
            return ('bearish', '#FF0000')
        else:
            return ('neutral', '#FFFF00')
    
    elif "P/S Ratio" in label:
        if numeric_value < 2:
            return ('bullish', '#00FF00')
        elif numeric_value > 10:
            return ('bearish', '#FF0000')
        else:
            return ('neutral', '#FFFF00')
    
    elif "P/CF Ratio" in label:
        if numeric_value < 10:
            return ('bullish', '#00FF00')
        elif numeric_value > 20:
            return ('bearish', '#FF0000')
        else:
            return ('neutral', '#FFFF00')
    
    # Profitability Metrics (higher is generally better)
    elif "EPS" in label and "Growth" not in label:
        if numeric_value > 0:
            return ('bullish', '#00FF00')
        else:
            return ('bearish', '#FF0000')
    
    elif any(margin in label for margin in ["Gross Margin", "Operating Margin", "Net Margin"]):
        if "Gross" in label:
            if numeric_value > 40:
                return ('bullish', '#00FF00')
            elif numeric_value < 20:
                return ('bearish', '#FF0000')
            else:
                return ('neutral', '#FFFF00')
        elif "Operating" in label:
            if numeric_value > 15:
                return ('bullish', '#00FF00')
            elif numeric_value < 5:
                return ('bearish', '#FF0000')
            else:
                return ('neutral', '#FFFF00')
        elif "Net" in label:
            if numeric_value > 10:
                return ('bullish', '#00FF00')
            elif numeric_value < 5:
                return ('bearish', '#FF0000')
            else:
                return ('neutral', '#FFFF00')
    
    # Return Ratios (higher is better)
    elif any(ratio in label for ratio in ["ROE", "ROA", "ROI"]):
        if "ROE" in label:
            if numeric_value > 15:
                return ('bullish', '#00FF00')
            elif numeric_value < 10:
                return ('bearish', '#FF0000')
            else:
                return ('neutral', '#FFFF00')
        elif "ROA" in label:
            if numeric_value > 5:
                return ('bullish', '#00FF00')
            elif numeric_value < 2:
                return ('bearish', '#FF0000')
            else:
                return ('neutral', '#FFFF00')
        elif "ROI" in label:
            if numeric_value > 10:
                return ('bullish', '#00FF00')
            elif numeric_value < 5:
                return ('bearish', '#FF0000')
            else:
                return ('neutral', '#FFFF00')
    
    elif "5 Year EPS Growth" in label:
        if numeric_value > 10:
            return ('bullish', '#00FF00')
        elif numeric_value < 0:
            return ('bearish', '#FF0000')
        else:
            return ('neutral', '#FFFF00')
    
    # Risk Metrics
    elif "Beta" in label:
        if 0.8 <= numeric_value <= 1.2:
            return ('neutral', '#FFFF00')
        elif numeric_value > 2:
            return ('bearish', '#FF0000')  # High volatility risk
        else:
            return ('neutral', '#FFFF00')
    
    elif "Debt/Equity" in label:
        if numeric_value == 0:
            return ('bullish', '#00FF00')
        elif numeric_value > 0.5:
            return ('bearish', '#FF0000')
        else:
            return ('neutral', '#FFFF00')
    
    elif "Current Ratio" in label:
        if 1.5 <= numeric_value <= 3:
            return ('bullish', '#00FF00')
        elif numeric_value < 1:
            return ('bearish', '#FF0000')
        else:
            return ('neutral', '#FFFF00')  # Too high might be inefficient
    
    elif "Quick Ratio" in label:
        if numeric_value >= 1:
            return ('bullish', '#00FF00')
        else:
            return ('bearish', '#FF0000')
    
    # Performance Returns
    elif "Return" in label:
        if numeric_value > 0:
            return ('bullish', '#00FF00')
        else:
            return ('bearish', '#FF0000')
    
    # Dividend Metrics
    elif "Dividend Yield" in label:
        if numeric_value > 2:
            return ('bullish', '#00FF00')
        elif numeric_value == 0:
            return ('neutral', '#FFFF00')
        else:
            return ('neutral', '#FFFF00')
    
    elif "Dividend Payout Ratio" in label:
        if 30 <= numeric_value <= 60:
            return ('bullish', '#00FF00')
        elif numeric_value > 80:
            return ('bearish', '#FF0000')
        else:
            return ('neutral', '#FFFF00')
    
    # 52W High/Low are informational, neutral
    elif any(x in label for x in ["52W Low", "52W High"]):
        return ('neutral', '#FFFF00')
    
    # Default case
    return ('neutral', '#FFFF00')


def create_financial_metrics_display(symbol, metrics):
    # Add custom CSS for colored metrics
    st.markdown("""
    <style>
    .metric-bullish {
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 4px solid #00FF00;
        padding: 10px;
        border-radius: 5px;
    }
    .metric-bearish {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 4px solid #FF0000;
        padding: 10px;
        border-radius: 5px;
    }
    .metric-neutral {
        background-color: rgba(255, 255, 0, 0.1);
        border-left: 4px solid #FFFF00;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.expander("Company Description", expanded=False):
        try:
            yf_data = yf.Ticker(symbol)
            description = yf_data.info.get("longBusinessSummary", "No description found for the selected symbol.")
            st.markdown(f"{description} ({symbol})")
        except:
            st.markdown(f"No description available for {symbol}")
    
    metrics_data = [
        [("P/E Ratio:", "peBasicExclExtraTTM"), ("PEG Ratio:", "pegTTM"), ("P/B Ratio:", "pb"), ("P/S Ratio:", "psTTM")],
        [("EPS (TTM):", "epsTTM"), ("Gross Margin (TTM):", "grossMarginTTM"), ("Net Margin (TTM):", "netProfitMarginTTM"), ("Operating Margin:", "operatingMarginTTM")],
        [("ROE (TTM):", "roeTTM"), ("ROA (TTM):", "roaTTM"), ("ROI (TTM):", "roiTTM"), ("5 Year EPS Growth:", "epsGrowth5Y")],
        [("Beta:", "beta"), ("Debt/Equity Ratio:", "totalDebt/totalEquityQuarterly"), ("Current Ratio:", "currentRatioQuarterly"), ("Quick Ratio:", "quickRatioQuarterly")],
        [("5D Return:", "5DayPriceReturnDaily"), ("13W Return:", "13WeekPriceReturnDaily"), ("52W Low:", "52WeekLow"), ("52W High:", "52WeekHigh")],
        [("Dividend Yield:", "currentDividendYieldTTM"), ("P/CF Ratio:", "pcfShareTTM"), ("52W Return:", "52WeekPriceReturnDaily"), ("Payout Ratio:", "payoutRatioTTM")]
    ]
    
    for metric_row in metrics_data:
        cols = st.columns([2, 2, 2, 2])
        for i, (label, key) in enumerate(metric_row):
            raw_value = metrics.get(key, 'N/A')
            rounded_value = round_digit(raw_value, 2)
            
            # Format percentage values
            display_value = rounded_value
            if "Return" in label or "Yield" in label or "Margin" in label:
                if rounded_value != "N/A":
                    display_value = f"{rounded_value}%"
            
            # Get sentiment and color
            sentiment, color = get_metric_sentiment(label, rounded_value)
            
            # Create colored container
            with cols[i]:
                st.markdown(f"""
                <div class="metric-{sentiment}">
                    <strong>{label}</strong><br>
                    <span style="font-size: 1.5em; color: {color}; font-weight: bold;">{display_value}</span>
                </div>
                """, unsafe_allow_html=True)

def info_card(symbol, get_stock_profile, get_stock_quote, get_stock_metrics, get_stock_lookup):
    profile = get_stock_profile(symbol)
    quote = get_stock_quote(symbol)
    metrics = get_stock_metrics(symbol)
    lookup = get_stock_lookup(symbol)
    
    if not lookup or not lookup.get('result'):
        st.error("Could not retrieve lookup data for this stock.")
        return
        
    lookup_data = lookup['result'][0]
    
    if not profile or not quote:
        st.error("Could not retrieve all data for this stock.")
        return

    display_toast_messages()
    create_info_display(symbol, profile, quote, metrics, lookup_data)
    create_financial_metrics_display(symbol, metrics)

def add_indicators_to_data(data, symbol, indicators, fperiod=None, finterval=None):
    """
    Add technical indicators to stock data.
    
    Parameters:
    - data: DataFrame with OHLCV data
    - symbol: Stock symbol string
    - indicators: List of tuples [(ind_type, ind_period), ...]
    - fperiod: Period for market indices (SP500, NASDAQ)
    - finterval: Interval for market indices (SP500, NASDAQ)
    
    Returns:
    - DataFrame with added indicator columns
    """
    
    data = data.copy()
    
    for ind_type, ind_period in indicators:
        try:
            if ind_type == "SP500":
                ind_col = f"{ind_type}-{fperiod}" if fperiod else f"{ind_type}-default"
                if fperiod and finterval:
                    data[ind_col] = met.SP500(period=fperiod, interval=finterval)
                else:
                    st.warning(f"Period and interval required for {ind_type}")
                    
            elif ind_type == "NASDAQ":
                ind_col = f"{ind_type}-{fperiod}" if fperiod else f"{ind_type}-default"
                if fperiod and finterval:
                    data[ind_col] = met.NASDAQ(period=fperiod, interval=finterval)
                else:
                    st.warning(f"Period and interval required for {ind_type}")
                    
            elif ind_type == "SMA":
                ind_col = f"{ind_type}-{ind_period}"
                data[ind_col] = met.MA(data['Close'], period=ind_period, ind_type='SMA')
                
            elif ind_type == "EMA":
                ind_col = f"{ind_type}-{ind_period}"
                data[ind_col] = met.MA(data['Close'], period=ind_period, ind_type='EMA')
                
            elif ind_type == "MACD":
                macd_data = met.MACD(data, short_period=12, long_period=26, signal_period=ind_period)
                data[f'MACD-{ind_period}'] = macd_data['macd']
                data[f'MACD_Signal-{ind_period}'] = macd_data['signal']
                data[f'MACD_Histogram-{ind_period}'] = macd_data['histogram']
                
            elif ind_type == "RSI":
                ind_col = f"{ind_type}-{ind_period}"
                data[ind_col] = met.RSI(data["Close"], period=ind_period)
                
            elif ind_type == "STO":
                ind_col = f"{ind_type}-{ind_period}"
                data[ind_col] = met.STO(data, period=ind_period)
                
            elif ind_type == "ATR":
                ind_col = f"{ind_type}-{ind_period}"
                data[ind_col] = met.ATR(data, period=ind_period)
                
            elif ind_type == "OBV":
                ind_col = f"{ind_type}-{ind_period}"
                data[ind_col] = met.OBV(data, period=ind_period)
                
            elif ind_type == "BOB":
                bb_data = met.BOB(data, period=ind_period)
                data[f'BB_Upper-{ind_period}'] = bb_data['upper']
                data[f'BB_Lower-{ind_period}'] = bb_data['lower']
                data[f'BB_Middle-{ind_period}'] = bb_data['middle']
                
            elif ind_type == "ADX":
                adx_data = met.ADX(data, period=ind_period)
                data[f'ADX-{ind_period}'] = adx_data['adx']
                data[f'+DI-{ind_period}'] = adx_data['+DI']
                data[f'-DI-{ind_period}'] = adx_data['-DI']
                
        except Exception as e:
            st.warning(f"Error calculating {ind_type}-{ind_period} for {symbol}: {e}")
            continue
    
    return data


def add_indicators_to_plot(fig, symbol_data, symbol, indicators, date_col, fperiod=None):
    """
    Add indicator traces to a plotly figure.
    
    Parameters:
    - fig: Plotly figure object
    - symbol_data: DataFrame with symbol data and indicators
    - symbol: Stock symbol string
    - indicators: List of tuples [(ind_type, ind_period), ...]
    - date_col: Name of the date column
    - fperiod: Period for market indices (used in naming)
    
    Returns:
    - Updated plotly figure
    """
    import plotly.graph_objects as go
    
    COLOR_PALETTE = ['#ff4081', '#ffd700', '#00e676', '#7c4dff', '#00bcd4', "#ff9900", '#e040fb', "#00fff2", '#ff1744', '#00e5ff']
    color_idx = 0
    
    for ind_type, ind_period in indicators:
        color = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]

        if ind_type == "SP500":
            col_name = f'SP500-{fperiod}' if fperiod else 'SP500-default'
            if col_name in symbol_data.columns:
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[col_name],
                    yaxis='y2',
                    name=f'{col_name}',
                    line=dict(color=color, width=2)
                ))

        elif ind_type == "NASDAQ":
            col_name = f'NASDAQ-{fperiod}' if fperiod else 'NASDAQ-default'
            if col_name in symbol_data.columns:
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[col_name],
                    yaxis='y2',
                    name=f'{col_name}',
                    line=dict(color=color, width=2)
                ))
        
        elif ind_type == "SMA":
            col_name = f'SMA-{ind_period}'
            if col_name in symbol_data.columns:
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[col_name],
                    name=f'{symbol} {col_name}',
                    line=dict(color=color, width=1.5, dash='dot')
                ))
                
        elif ind_type == "EMA":
            col_name = f'EMA-{ind_period}'
            if col_name in symbol_data.columns:
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[col_name],
                    name=f'{symbol} {col_name}',
                    line=dict(color=color, width=1.5, dash='dash')
                ))

        elif ind_type == "MACD":
            macd_col = f'MACD-{ind_period}'
            signal_col = f'MACD_Signal-{ind_period}'
            hist_col = f'MACD_Histogram-{ind_period}'
            
            if all(col in symbol_data.columns for col in [macd_col, signal_col, hist_col]):
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[macd_col],
                    yaxis='y2',
                    name=f'{symbol} MACD-{ind_period}',
                    line=dict(color=color, width=2)
                ))
                
                signal_color = COLOR_PALETTE[(color_idx + 1) % len(COLOR_PALETTE)]
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[signal_col],
                    yaxis='y2',
                    name=f'{symbol} MACD Signal-{ind_period}',
                    line=dict(color=signal_color, width=2, dash='dash')
                ))

                hist_color = COLOR_PALETTE[(color_idx + 2) % len(COLOR_PALETTE)]
                fig.add_trace(go.Bar(
                    x=symbol_data[date_col],
                    y=symbol_data[hist_col],
                    yaxis='y2',
                    name=f'{symbol} MACD Histogram-{ind_period}',
                    marker_color=hist_color,
                    opacity=0.7
                ))

                color_idx += 2
                
        elif ind_type == "RSI":
            col_name = f'RSI-{ind_period}'
            if col_name in symbol_data.columns:
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[col_name],
                    yaxis='y2',
                    name=f'{symbol} {col_name}',
                    line=dict(color=color, width=1.5)
                ))
                
        elif ind_type == "STO":
            col_name = f'STO-{ind_period}'
            if col_name in symbol_data.columns:
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[col_name],
                    name=f'{symbol} {col_name}',
                    line=dict(color=color, width=1.5)
                ))

        elif ind_type == "ATR":
            col_name = f'ATR-{ind_period}'
            if col_name in symbol_data.columns:
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[col_name],
                    yaxis='y2',
                    name=f'{symbol} {col_name}',
                    line=dict(color=color, width=1.5)
                ))

        elif ind_type == "OBV":
            col_name = f'OBV-{ind_period}'
            if col_name in symbol_data.columns:
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[col_name],
                    yaxis='y2',
                    name=f'{symbol} {col_name}',
                    line=dict(color=color, width=1.5)
                ))
                
        elif ind_type == "BOB":
            upper_col = f'BB_Upper-{ind_period}'
            lower_col = f'BB_Lower-{ind_period}'
            middle_col = f'BB_Middle-{ind_period}'
            
            if all(col in symbol_data.columns for col in [upper_col, lower_col, middle_col]):
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[upper_col],
                    name=f'{symbol} BB Upper-{ind_period}',
                    line=dict(color=color, width=1),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[lower_col],
                    name=f'{symbol} BB Lower-{ind_period}',
                    line=dict(color=color, width=1),
                    fill='tonexty',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[middle_col],
                    name=f'{symbol} BB Middle-{ind_period}',
                    line=dict(color=color, width=1.5, dash='dashdot')
                ))

        elif ind_type == "ADX":
            adx = f'ADX-{ind_period}'
            di_plus = f'+DI-{ind_period}'
            di_minus = f'-DI-{ind_period}'
            
            if all(col in symbol_data.columns for col in [adx, di_plus, di_minus]):

                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[adx],
                    yaxis='y2',
                    name=f'{symbol} ADX-{ind_period}',
                    line=dict(color=color, width=1),
                    showlegend=False
                ))

                pos_color = COLOR_PALETTE[(color_idx + 1) % len(COLOR_PALETTE)]
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[di_plus],
                    yaxis='y2',
                    name=f'{symbol} +DI-{ind_period}',
                    line=dict(color=pos_color, width=1),
                    showlegend=False
                ))

                neg_color = COLOR_PALETTE[(color_idx + 2) % len(COLOR_PALETTE)]
                fig.add_trace(go.Scatter(
                    x=symbol_data[date_col],
                    y=symbol_data[di_minus],
                    yaxis='y2',
                    name=f'{symbol} -DI-{ind_period}',
                    line=dict(color=neg_color, width=1),
                    showlegend=False
                ))

                color_idx += 2
        
        color_idx += 1
    
    return fig


def graph(fsymbols, fperiod, finterval):
    """
    Plots closing prices and selected moving averages for one or more stocks.
    Enhanced to handle both single and multiple stocks properly.
    """
    import yfinance as yf
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
    
    if isinstance(fsymbols, str):
        fsymbols = [fsymbols]
    
    try:
        df = yf.download(
            fsymbols,
            period=fperiod,
            interval=finterval,
            repair=True,
            group_by='ticker' if len(fsymbols) > 1 else None
        )
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return

    if df.empty:
        st.error(f"No data available for {fsymbols}")
        return

    if len(fsymbols) == 1 and not isinstance(df.columns, pd.MultiIndex):
        df = pd.concat({fsymbols[0]: df}, axis=1)

    plot_df = []
    indicators = st.session_state.get('indicators', [])

    for symbol in fsymbols:
        if isinstance(df.columns, pd.MultiIndex):
            try:
                data = df[symbol].copy()
            except KeyError:
                st.warning(f"No data for {symbol} in downloaded DataFrame")
                continue
        else:
            data = df.copy()
        
        if data.empty:
            st.warning(f"Empty data for {symbol}")
            continue

        data = data.reset_index()
        data['Symbol'] = symbol

        # Add indicators using the new function
        data = add_indicators_to_data(data, symbol, indicators, fperiod, finterval)
        
        plot_df.append(data)

    if not plot_df:
        st.error("No data available to plot.")
        return

    plot_df = pd.concat(plot_df, ignore_index=True)

    # Ensure we have a proper Date column
    date_columns = ['Date', 'Datetime', 'date', 'datetime', 'index']
    date_col = None
    for col in date_columns:
        if col in plot_df.columns:
            date_col = col
            break
    
    if date_col is None:
        plot_df = plot_df.reset_index()
        if 'index' in plot_df.columns:
            date_col = 'index'
        else:
            st.error("Could not find a date column in the data.")
            return

    fig = go.Figure()
    
    # Add main price lines for each symbol
    for symbol in fsymbols:
        symbol_data = plot_df[plot_df['Symbol'] == symbol].copy()
        if symbol_data.empty:
            continue
            
        # Add main price line
        fig.add_trace(go.Scatter(
            x=symbol_data[date_col],
            y=symbol_data['Close'],
            name=f'{symbol} Close',
            line=dict(width=2)
        ))
        
        # Add technical indicators using the new function
        fig = add_indicators_to_plot(fig, symbol_data, symbol, indicators, date_col, fperiod)

    fig.update_layout(
        title=f"Closing Prices and Technical Indicators: {', '.join(fsymbols)}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend_title="Symbol / Metric", 
        hovermode="x unified",
        yaxis2=dict(
            title="Indicators",
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def candlestick(fsymbol, fperiod, finterval):
    df = yf.download(fsymbol, period=fperiod, interval=finterval, repair=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    datetime_cols = [col for col in df.columns if col.lower() in ['date', 'datetime', 'timestamp', 'index']]
    if datetime_cols:
        x_col = datetime_cols[0]
    else:
        x_col = df.columns[0]

    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df[x_col],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=f'{fsymbol} Candlestick',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        opacity=0.9
    ))

    indicators = st.session_state.get('indicators', [])
    COLOR_PALETTE = ['#ffd700', '#7c4dff', "#00e1ff", "#ff9900", "#fb40ee", "#0080ff", "#febedd", "#f1f193", "#5900ff", "#ffffff"]
    
    for idx, (ind_type, ind_period) in enumerate(indicators):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

        if ind_type == "SP500":
            df[f'SP500-{fperiod}'] = met.SP500(period=fperiod, interval=finterval)
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[f'SP500-{fperiod}'],
                yaxis='y2',
                name=f'SP500-{fperiod}',
                line=dict(color=color, width=2)
            ))

        elif ind_type == "NASDAQ":
            df[f'NASDAQ-{ind_period}'] = met.NASDAQ(period=fperiod, interval=finterval)
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[f'NASDAQ-{fperiod}'],
                yaxis='y2',
                name=f'NASDAQ-{fperiod}',
                line=dict(color=color, width=2)
            ))

        
        elif ind_type == "SMA":
            df[f'SMA-{ind_period}'] = met.MA(df['Close'], period=ind_period, ind_type='SMA')
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[f'SMA-{ind_period}'],
                name=f'SMA-{ind_period}',
                line=dict(color=color, width=1.5, dash='dot')
            ))

        elif ind_type == "EMA":
            df[f'EMA-{ind_period}'] = met.MA(df['Close'], period=ind_period, ind_type='EMA')
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[f'EMA-{ind_period}'],
                name=f'EMA-{ind_period}',
                line=dict(color=color, width=1.5, dash='dash')
            ))
        elif ind_type == "MACD":
            macd_data = met.MACD(df, short_period=12, long_period=26, signal_period=ind_period)
            
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=macd_data['macd'],
                yaxis='y2',
                name=f'MACD-{ind_period}',
                line=dict(color=color, width=2),
            ))
            
            signal_color = COLOR_PALETTE[(idx + 1) % len(COLOR_PALETTE)]
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=macd_data['signal'],
                yaxis='y2',
                name=f'MACD Signal-{ind_period}',
                line=dict(color=signal_color, width=2, dash='dash'),
                
            ))
            
            hist_color = COLOR_PALETTE[(idx + 2) % len(COLOR_PALETTE)]
            fig.add_trace(go.Bar(
                x=df[x_col],
                y=macd_data['histogram'],
                yaxis='y2',
                name=f'MACD Histogram-{ind_period}',
                marker_color=hist_color,
                opacity=0.7,
            ))

            idx = idx + 2

        elif ind_type == "RSI":
            df[f'RSI-{ind_period}'] = met.RSI(df['Close'], period=ind_period)
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[f'RSI-{ind_period}'],
                yaxis='y2',
                name=f'RSI-{ind_period}',
                line=dict(color=color, width=1.5)
            ))

        elif ind_type == "STO":
            df[f'STO-{ind_period}'] = met.STO(df, period=ind_period)
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[f'STO-{ind_period}'],
                name=f'STO-{ind_period}',
                line=dict(color=color, width=1.5)
            ))

        elif ind_type == "ATR":
            df[f'ATR-{ind_period}'] = met.ATR(df, period=ind_period)
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[f'ATR-{ind_period}'],
                yaxis='y2',
                name=f'ATR-{ind_period}',
                line=dict(color=color, width=1.5)
            ))

        elif ind_type == "OBV":
            df[f'OBV-{ind_period}'] = met.OBV(df, period=ind_period)
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[f'OBV-{ind_period}'],
                yaxis='y2',
                name=f'OBV-{ind_period}',
                line=dict(color=color, width=1.5)
            ))

        elif ind_type == "BOB":
            bb_data = met.BOB(df, period=ind_period)
            
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=bb_data['upper'],
                name=f'BB Upper-{ind_period}',
                line=dict(color=color, width=1),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=bb_data['lower'],
                name=f'BB Lower-{ind_period}',
                line=dict(color=color, width=1),
                fill='tonexty',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=bb_data['middle'],
                name=f'BB Middle-{ind_period}',
                line=dict(color=color, width=1.5, dash='dashdot')
            ))
        
        elif ind_type == "ADX":
            adx_data = met.ADX(df, period=ind_period)

            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=adx_data['adx'],
                yaxis='y2',
                name=f'ADX-{ind_period}',
                line=dict(color=color, width=1),
                showlegend=True
            ))
            
            pos_color = COLOR_PALETTE[(idx + 1) % len(COLOR_PALETTE)]
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=adx_data['+DI'],
                yaxis='y2',
                name=f'+DI-{ind_period}',
                line=dict(color=pos_color, width=1),
                showlegend=True
            ))

            neg_color = COLOR_PALETTE[(idx + 2) % len(COLOR_PALETTE)]
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=adx_data['-DI'],
                yaxis='y2',
                name=f'-DI-{ind_period}',
                line=dict(color=neg_color, width=1),
                showlegend=True
            ))

            idx = idx + 2

        idx += 1

    fig.update_layout(
        title=f'{fsymbol} Candlestick Chart',
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=500,
        plot_bgcolor="#181A20",
        paper_bgcolor="#181A20",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
        xaxis=dict(gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a"),
        yaxis2=dict(
            title="Indicators",
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def correlation_heatmap(fsymbols, fperiod='1y'):
    """
    Enhanced correlation heatmap with financial-specific features
    """
    try:
        with st.spinner('Downloading market data...'):
            df = yf.download(fsymbols, period=fperiod, progress=False)
        
        if df.empty:
            st.error("No data available for correlation analysis.")
            return
            
        close_data = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df[['Close']]   
        returns = close_data.pct_change(fill_method=None).dropna()
        
        if returns.empty:
            st.error("No return data available for correlation analysis.")
            return
 
        correlation = returns.corr(method='pearson')
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=np.round(correlation.values, 2),
                texttemplate="%{text}",
                hoverinfo="x+y+z"
            )
        )
        
        fig.update_layout(
            title=f'Stock Return Correlations ({fperiod} period)',
            xaxis_title='Stocks',
            yaxis_title='Stocks',
            height=600,
            annotations=[
                dict(
                    text="",
                    xref="paper", yref="paper",
                    x=0.5, y=1.1, showarrow=False
                )
            ],
            coloraxis_colorbar=dict(
                title='Correlation',
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1 (Perfect Negative)", "-0.5", "0 (No Correlation)", "0.5", "1 (Perfect Positive)"]
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")


def get_darker_color(hex_color: str, factor: float = 0.8):
    """
    Given a hex color, returns a slightly darker hex color.
    Enhanced with better error handling.
    """
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return hex_color  # Return original if invalid format
            
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darker_rgb = [max(0, min(255, int(channel * factor))) for channel in rgb]
        return f'#{darker_rgb[0]:02x}{darker_rgb[1]:02x}{darker_rgb[2]:02x}'
    except (ValueError, IndexError):
        return hex_color  # Return original if conversion fails


def pie_chart(df=[], balance=0):
    """
    Creates a portfolio pie chart with enhanced error handling.
    """
    if (isinstance(df, list) and len(df) == 0) or (hasattr(df, 'empty') and df.empty):
        if balance == 0:
            st.write("Your portfolio is empty. No chart to display.")
            return
        else:
            # Show only cash
            fig = go.Figure(go.Pie(
                labels=['Cash'],
                values=[balance],
                marker=dict(colors=['#CFEFBA'])
            ))
            fig.update_layout(
                title_text="Portfolio Allocation (Cash Only)",
                title_x=0.5,
                margin=dict(t=30, l=0, r=0, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            return

    try:
        # Dictionary to hold the summed value and sector for each unique ticker
        ticker_data = {}
        
        # Calculate the total value for each ticker and its corresponding sector
        for _, row in df.iterrows():
            current_price = row.get("current_price")
            if pd.notna(current_price) and row.get("quantity") is not None:
                ticker = row["ticker"]
                stock_value = current_price * row["quantity"]
                
                # Sum the value for each ticker and get the sector once
                if ticker not in ticker_data:
                    profile = get_stock_profile(ticker)
                    sector = profile.get('finnhubIndustry', 'N/A') if profile else 'N/A'
                    ticker_data[ticker] = {'value': stock_value, 'sector': sector}
                else:
                    ticker_data[ticker]['value'] += stock_value

        if not ticker_data and balance == 0:
            st.write("No valid portfolio data to display.")
            return

        # Now, calculate the total value for each sector
        sector_totals = {}
        for ticker, data in ticker_data.items():
            sector = data['sector']
            if sector in sector_totals:
                sector_totals[sector] += data['value']
            else:
                sector_totals[sector] = data['value']
        
        # Prepare lists for the Plotly Sunburst chart
        labels = []
        parents = []
        values = []
        colors = []
        
        # Dynamically assign colors to each unique sector
        unique_sectors = sorted(list(set(sector_totals.keys())))
        
        # Use a built-in Plotly qualitative color scale
        sector_color_scale = p_colors.qualitative.Plotly
        sector_color_map = {
            sector: sector_color_scale[i % len(sector_color_scale)] 
            for i, sector in enumerate(unique_sectors)
        }

        # 1. Add the top-level categories: "Holdings" and "Cash"
        if ticker_data:
            labels.append("Holdings")
            parents.append("")
            values.append(sum(sector_totals.values()))
            colors.append("#E4E4DC")  # A distinct color for the overall holdings

        if balance > 0:
            labels.append("Cash")
            parents.append("")
            values.append(balance)
            colors.append("#CFEFBA")  # A distinct color for cash

        # 2. Add the sectors as a child of "Holdings" with their total values
        for sector in unique_sectors:
            labels.append(sector)
            parents.append("Holdings")
            values.append(sector_totals[sector])
            colors.append(sector_color_map[sector])

        # 3. Add the individual tickers as a child of their respective sectors
        for ticker, data in ticker_data.items():
            labels.append(ticker)
            parents.append(data['sector'])
            values.append(data['value'])
            sector_color = sector_color_map[data['sector']]
            ticker_color = get_darker_color(sector_color, 0.8)
            colors.append(ticker_color)

        # Create the sunburst chart
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            branchvalues="total",
            hoverinfo="label+value",
            textinfo="label+percent entry",
        ))

        fig.update_layout(
            title_text="Portfolio Allocation",
            title_x=0.41,
            margin=dict(t=30, l=0, r=0, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating portfolio pie chart: {e}")
        st.write("Debug info:", str(df.head() if hasattr(df, 'head') else df))


def indicator_system():
    if 'indicators' not in st.session_state:
        st.session_state.indicators = []
    
    c1, c2, c3 = st.columns([2, 1, 1])
    ind_type = c1.selectbox("Indicator", ["SP500", "NASDAQ", "SMA", "EMA", "MACD","RSI", "STO", "OBV", "ATR", "BOB", "ADX"], key="ind_type", label_visibility="collapsed")
    ind_period = c2.number_input("Period", min_value=2, max_value=200, value=20, key="ind_period", label_visibility="collapsed")
    
    if c3.button("Add", key="add_indicator", use_container_width=True):
        ind_tuple = (ind_type, int(ind_period))
        if ind_tuple not in st.session_state.indicators:
            st.session_state.indicators.append(ind_tuple)
            st.rerun()
    
    if st.session_state.indicators:
        st.markdown("Applied Indicators:")
        num_cols = min(len(st.session_state.indicators), 8)
        cols = st.columns(num_cols)
        for i, ind in enumerate(list(st.session_state.indicators)):
            with cols[i % num_cols]:
                ind_label = f"{ind[0]}-{ind[1]}"
                st.markdown(f"<div style='background:#2c2f36; padding:0.4rem; border-radius:0.5rem; text-align:center; margin-bottom:0.5rem;'>{ind_label}</div>", unsafe_allow_html=True)
                if st.button("Remove", key=f"remove_{ind_label}", use_container_width=True):
                    st.session_state.indicators.remove(ind)
                    st.rerun()

def render_graph(symbol, show, interval, period, chart_type, realtime):
    if not show:
        st.warning("Graph display is turned off. Check the 'Show Graph' box to see the chart.")
        return
    
    try:
        if chart_type == "Candlestick":
            candlestick(symbol, period, interval)
        elif chart_type == "Correlation Heatmap":
            correlation_heatmap(symbol, period)
        else:
            graph(symbol, period, interval)

        if realtime and is_market_open() == "True":
            st.caption("🟢 Real-time mode enabled. Auto-refreshing every 10 seconds.")
        elif realtime:
            market_status = is_market_open()
            st.caption(f"⚪ Real-time mode enabled, but {market_status}")
    except Exception as e:
        st.error(f"An error occurred while rendering the chart: {e}")


def sell_stock_from_holding(symbol, quantity):
    market_open = is_market_open()
    if market_open != "True":
        st.session_state.toast_message = f"Cannot execute this order, because {market_open}"
        st.session_state.toast_type = "error"
        return
    
    holdings_df, portfolio_df, transactions_df = load_data()
    stock_holdings = holdings_df[holdings_df['ticker'] == symbol].copy()
    
    if stock_holdings.empty:
        st.session_state.toast_message = f"You don't own any shares of {symbol}!"
        st.session_state.toast_type = "error"
        return
    
    available_quantity = stock_holdings['quantity'].sum()
    
    if available_quantity < quantity:
        st.session_state.toast_message = f"You only have {int(available_quantity)} shares of {symbol}!"
        st.session_state.toast_type = "error"
        return
    
    current_price = get_last_price(symbol)
    if current_price == "N/A":
        st.session_state.toast_message = "Could not get current price for this stock!"
        st.session_state.toast_type = "error"
        return
        
    total_revenue = current_price * quantity
    sell_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if 'timestamp' in stock_holdings.columns:
        stock_holdings = stock_holdings.sort_values('timestamp')
    else:
        stock_holdings = stock_holdings.sort_index() 
    
    remaining_to_sell = quantity
    indices_to_remove = []
    total_cost_basis = 0
    
    for idx, row in stock_holdings.iterrows():
        if remaining_to_sell <= 0:
            break
            
        if row['quantity'] <= remaining_to_sell:
            shares_sold = row['quantity']
            remaining_to_sell -= shares_sold
            indices_to_remove.append(idx)
            total_cost_basis += shares_sold * row['buyprice']
            
            matching_buys = transactions_df[
                (transactions_df['ticker'] == symbol) &
                (transactions_df['buyprice'] == row['buyprice']) &
                (transactions_df['quantity'] == row['quantity']) &
                (transactions_df['sold'] == 0)
            ]
            
            if not matching_buys.empty:
                transactions_df.loc[matching_buys.index[0], 'sold'] = 1
                transactions_df.loc[matching_buys.index[0], 'sellprice'] = current_price
                transactions_df.loc[matching_buys.index[0], 'sell_timestamp'] = sell_timestamp
            
        else:
            shares_sold = remaining_to_sell
            holdings_df.loc[idx, 'quantity'] -= remaining_to_sell
            total_cost_basis += shares_sold * row['buyprice']
            remaining_to_sell = 0
            
            matching_buys = transactions_df[
                (transactions_df['ticker'] == symbol) &
                (transactions_df['buyprice'] == row['buyprice']) &
                (transactions_df['quantity'] >= shares_sold) &
                (transactions_df['sold'] == 0)
            ]
            
            if not matching_buys.empty:
                original_qty = transactions_df.loc[matching_buys.index[0], 'quantity']
                sold_transaction = transactions_df.loc[matching_buys.index[0]].copy()
                sold_transaction['quantity'] = shares_sold
                sold_transaction['sold'] = 1
                sold_transaction['sellprice'] = current_price
                sold_transaction['sell_timestamp'] = sell_timestamp
                
                transactions_df.loc[matching_buys.index[0], 'quantity'] -= shares_sold
                transactions_df = pd.concat([transactions_df, pd.DataFrame([sold_transaction])], ignore_index=True)
    
    new_transaction = pd.DataFrame([{
        'ticker': symbol,
        'quantity': quantity,
        'buyprice': None,
        'buy_timestamp': None,
        'sold': 1,
        'sellprice': current_price,
        'sell_timestamp': sell_timestamp
    }])
    
    transactions_df = pd.concat([transactions_df, new_transaction], ignore_index=True)

    if indices_to_remove:
        holdings_df = holdings_df.drop(indices_to_remove).reset_index(drop=True)

    current_balance = portfolio_df["balance"].iloc[0]
    new_balance = round(current_balance + total_revenue, 2)
    portfolio_df.loc[0, "balance"] = new_balance

    save_data(holdings_df, portfolio_df, transactions_df)

    profit_loss = total_revenue - total_cost_basis
    pl_emoji = "📈" if profit_loss >= 0 else "📉"
    
    st.session_state.toast_message = f"✅ Sold {quantity} shares of {symbol} for ${total_revenue:.2f}! {pl_emoji} P&L: ${profit_loss:.2f} | Balance: ${new_balance:.2f}"
    st.session_state.toast_type = "success"


@st.cache_data(ttl=300)
def calculate_portfolio_metrics():
    holdings_df, portfolio_df, transactions_df = load_data()
    portfolio_value = get_portfolio_value()
    daily_change = get_daily_change()
    balance_change = get_daily_change("balance")
    alltime_change = get_alltime_change()
    alltime_change_pct = get_alltime_change("percent")
    
    return {
        'portfolio_value': portfolio_value,
        'daily_change': daily_change,
        'balance_change': balance_change,
        'alltime_change': alltime_change,
        'alltime_change_pct': alltime_change_pct,
        'portfolio_df': portfolio_df
    }

def process_holdings_display(holdings_df):
    if holdings_df.empty:
        return pd.DataFrame()
    
    holdings_display = []
    for _, row in holdings_df.iterrows():
        current_price = row.get("current_price", "N/A")
        if current_price == "N/A":
            current_price = get_last_price(row["ticker"])
        quantity = row["quantity"]
        buyprice = row["buyprice"]
        
        if current_price != "N/A" and isinstance(current_price, (int, float)):
            value = current_price * quantity
            gain_loss = (current_price - buyprice) * quantity
            gain_loss_pct = ((current_price - buyprice) / buyprice) * 100
        else:
            value = gain_loss = gain_loss_pct = "N/A"
        
        holdings_display.append({
            "Symbol": row["ticker"],
            "Quantity": int(quantity),
            "Buy Price": f"${buyprice:.2f}",
            "Current Price": f"${current_price:.2f}" if current_price != "N/A" else "N/A",
            "Market Value": f"${value:,.2f}" if value != "N/A" else "N/A",
            "Gain/Loss": f"${gain_loss:,.2f}" if gain_loss != "N/A" else "N/A",
            "Gain/Loss %": f"{gain_loss_pct:.2f}%" if gain_loss_pct != "N/A" else "N/A",
        })
    
    return pd.DataFrame(holdings_display)

def calculate_portfolio_history():
    holdings_df, portfolio_df, transactions_df = load_data()
    
    if transactions_df.empty:
        return pd.DataFrame()
    
    transactions_df['timestamp'] = pd.to_datetime(
        transactions_df['buy_timestamp'].fillna(transactions_df['sell_timestamp'])
    )
    
    transactions_df = transactions_df.sort_values('timestamp')
    
    portfolio_history = []
    current_holdings = {}
    cash_balance = 35000
    
    for _, transaction in transactions_df.iterrows():
        timestamp = transaction['timestamp']
        ticker = transaction['ticker']
        
        if transaction['sold'] == 0:
            if ticker not in current_holdings:
                current_holdings[ticker] = []
            current_holdings[ticker].append({
                'quantity': transaction['quantity'],
                'buy_price': transaction['buyprice']
            })
            cash_balance -= transaction['quantity'] * transaction['buyprice']
        else:
            if ticker in current_holdings and transaction['sellprice']:
                quantity_to_sell = transaction['quantity']
                revenue = quantity_to_sell * transaction['sellprice']
                cash_balance += revenue
                
                for holding in current_holdings[ticker]:
                    if quantity_to_sell <= 0:
                        break
                    if holding['quantity'] <= quantity_to_sell:
                        quantity_to_sell -= holding['quantity']
                        holding['quantity'] = 0
                    else:
                        holding['quantity'] -= quantity_to_sell
                        quantity_to_sell = 0
                
                current_holdings[ticker] = [h for h in current_holdings[ticker] if h['quantity'] > 0]
                if not current_holdings[ticker]:
                    del current_holdings[ticker]
        
        portfolio_value = cash_balance
        for ticker, holdings in current_holdings.items():
            current_price = get_last_price(ticker)
            if current_price != "N/A":
                for holding in holdings:
                    portfolio_value += holding['quantity'] * current_price
        
        portfolio_history.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash_balance': cash_balance
        })
    
    if portfolio_history:
        history_df = pd.DataFrame(portfolio_history)
        
        today = pd.Timestamp.now()
        if history_df['timestamp'].max() < today:
            current_portfolio_value = cash_balance
            for ticker, holdings in current_holdings.items():
                current_price = get_last_price(ticker)
                if current_price != "N/A":
                    for holding in holdings:
                        current_portfolio_value += holding['quantity'] * current_price
            
            history_df = pd.concat([
                history_df,
                pd.DataFrame([{
                    'timestamp': today,
                    'portfolio_value': current_portfolio_value,
                    'cash_balance': cash_balance
                }])
            ], ignore_index=True)
        
        return history_df
    
    return pd.DataFrame()

def create_performance_chart(timeframe='1M'):
    """
    Enhanced portfolio performance chart with technical indicators support.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
    from datetime import timedelta
    
    history_df = calculate_portfolio_history()
    
    if history_df.empty:
        return None
    
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    
    end_date = history_df['timestamp'].max()
    
    if timeframe == '1d':
        start_date = end_date - timedelta(days=1)
    elif timeframe == '3d':
        start_date = end_date - timedelta(days=3)
    elif timeframe == '1W':
        start_date = end_date - timedelta(days=7)
    elif timeframe == '1M':
        start_date = end_date - timedelta(days=30)
    elif timeframe == '3M':
        start_date = end_date - timedelta(days=90)
    elif timeframe == '6M':
        start_date = end_date - timedelta(days=180)
    elif timeframe == '1Y':
        start_date = end_date - timedelta(days=365)
    elif timeframe == '5Y':
        start_date = end_date - timedelta(days=1825)
    else:
        start_date = history_df['timestamp'].min()
    
    filtered_df = history_df[history_df['timestamp'] >= start_date].copy()
    
    if filtered_df.empty:
        filtered_df = history_df.copy()
    
    # Prepare data for technical indicators
    chart_data = filtered_df.copy()
    chart_data = chart_data.rename(columns={
        'portfolio_value': 'Close',
        'timestamp': 'Date'
    })
    
    chart_data['Open'] = chart_data['Close']
    chart_data['High'] = chart_data['Close']
    chart_data['Low'] = chart_data['Close']
    chart_data['Volume'] = 100  

    # Get indicators from session state
    indicators = st.session_state.get('indicators', [])
    
    # Add indicators to the data if any are selected
    if indicators:
        try:
            chart_data = add_indicators_to_data(
                chart_data, 
                'Portfolio', 
                indicators, 
                fperiod=timeframe, 
                finterval='1d'
            )
        except Exception as e:
            st.warning(f"Error adding indicators to portfolio chart: {e}")
    
    # Calculate performance metrics
    initial_value = filtered_df['portfolio_value'].iloc[0] if not filtered_df.empty else 35000
    final_value = filtered_df['portfolio_value'].iloc[-1] if not filtered_df.empty else initial_value
    change = final_value - initial_value
    change_pct = (change / initial_value * 100) if initial_value != 0 else 0
    
    # Create the figure
    fig = go.Figure()
    
    # Add main portfolio value line
    fig.add_trace(go.Scatter(
        x=chart_data['Date'],
        y=chart_data['Close'],
        yaxis="y",
        mode='lines',
        name='Portfolio Value',
        line=dict(color="#59ff00" if change >= 0 else '#ff4444', width=2),
    ))
    
    # Add cash balance line
    fig.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df['cash_balance'],
        yaxis="y2",
        mode='lines',
        name='Cash Balance',
        line=dict(color='#00bcd4', width=1, dash='dash'),
    ))
    
    if indicators:
        try:
            fig = add_indicators_to_plot(
                fig, 
                chart_data, 
                'Portfolio', 
                indicators, 
                'Date', 
                fperiod=timeframe
            )
        except Exception as e:
            st.warning(f"Error adding indicator traces to portfolio chart: {e}")
    
    layout_config = dict(
        title=dict(
            text=f'Portfolio Performance - {timeframe}',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Date',
        yaxis_title='Portfolio Value',
        hovermode='x unified',
        template='plotly_dark',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(gridcolor="#2a2a2a"),
        yaxis2=dict(
            title="Cash Balance",
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    # Add secondary y-axis if indicators are present
    if indicators:
        layout_config['yaxis2'] = dict(
            title="Indicators",
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
        )
    
    fig.update_layout(**layout_config)
    
    fig.update_xaxes(
        rangeslider_visible=False,
        type='date'
    )
    
    fig.update_yaxes(
        tickformat='$,.0f'
    )

    
    return fig, change, change_pct

@st.cache_data(ttl=3600)
def get_stock_news(symbol, early_date = get_date(1), late_date = get_date()):

    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={early_date}&to={late_date}&token={FINNHUB_API_KEY}"
    news = get_stock_data(url)
    return news

@st.cache_data(ttl=3600)
def get_insider_transactions(symbol, early_date = get_date(1), late_date = get_date()):

    url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol}&from={early_date}&to={late_date}&token={FINNHUB_API_KEY}"
    transactions = get_stock_data(url)
    return transactions

@st.cache_data(ttl=18000)
def get_insider_sentiment(symbol, early_date, late_date):

    url = f"https://finnhub.io/api/v1/stock/insider-sentiment?symbol={symbol}&from={early_date}&to={late_date}&token={FINNHUB_API_KEY}"
    sentiment_data = get_stock_data(url)
    return sentiment_data

@st.cache_data(ttl=18000)
def get_recommendation_trends(symbol):
    
    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={FINNHUB_API_KEY}"
    trends = get_stock_data(url)
    return trends

@st.cache_data(ttl=18000)
def get_eps_calendar(symbol, early_date, late_date):

    symbol_str = ""
    if symbol:
        symbol_str = f"&symbol={symbol}"
    url = f"https://finnhub.io/api/v1/calendar/earnings?from={early_date}&to={late_date}{symbol_str}&token={FINNHUB_API_KEY}"
    calendar = get_stock_data(url)
    return calendar


DatabaseManager.initialize_database(db_manager)