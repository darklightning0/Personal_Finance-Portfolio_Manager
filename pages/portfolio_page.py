import streamlit as st
import theme
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from utility import (
    load_data, 
    get_portfolio_value,
    get_daily_change,
    get_alltime_change,
    update_daily_portfolio,
    get_holdings_summary,
    get_transaction_history,
    create_performance_chart, 
    round_digit, 
    create_trading_interface, 
    indicator_system, 
    pie_chart, 
    correlation_heatmap,
    load_data,
    calculate_portfolio_history,
)

theme.custom_theme()
st.set_page_config(layout="wide")

st.title("My Portfolio")

update_daily_portfolio()

holdings_df, portfolio_df, transactions_df = load_data()

portfolio_value = get_portfolio_value()
daily_change = get_daily_change("value")
balance_change = get_daily_change("balance")
alltime_change = get_alltime_change("net")
alltime_change_pct = get_alltime_change("percent")

portfolio_value_col, balance_col = st.columns(2, gap="large", border=False)

with portfolio_value_col:
    yesterday_value = portfolio_df['yesterday_value'].iloc[0] if not portfolio_df.empty else 0
    daily_change_pct = (daily_change / yesterday_value * 100) if yesterday_value != 0 else 0
    
    st.metric(
        label="Portfolio Value",
        value=f"${round_digit(portfolio_value, 2)}",
        delta=f"{round_digit(daily_change_pct, 2)}%",
    )

with balance_col:
    yesterday_balance = portfolio_df['yesterday_balance'].iloc[0] if not portfolio_df.empty else 35000
    balance_change_pct = (balance_change / yesterday_balance * 100) if yesterday_balance != 0 else 0
    current_balance = portfolio_df['balance'].iloc[0] if not portfolio_df.empty else 35000
    
    st.metric(
        label="Your Balance",
        value=f"${round_digit(current_balance, 2)}",
        delta=f"{round_digit(balance_change, 2)} ({round_digit(balance_change_pct, 2)}%)",
    )

daily_change_col, alltime_change_col = st.columns(2, gap="large", border=False)

with daily_change_col:
    st.metric(
        label="Daily Change",
        value=f"${round_digit(daily_change, 2)}",
    )

with alltime_change_col:
    st.metric(
        label="Total Return",
        value=f"${round_digit(alltime_change, 2)}",
        delta=f"{round_digit(alltime_change_pct, 2)}%"
    )

if holdings_df.empty:
    st.info("📊 Your portfolio is empty. Start by searching and buying some stocks!")
else:
    holdings_summary_df = get_holdings_summary()
    
    if not holdings_summary_df.empty:
        holdings_display_data = []
        
        for _, row in holdings_summary_df.iterrows():
            ticker = row['ticker']
            total_quantity = row['total_quantity']
            avg_buy_price = row['avg_buy_price']
            current_price = row['current_price']
            
            if pd.notna(current_price) and current_price != 0:
                market_value = current_price * total_quantity
                gain_loss = (current_price - avg_buy_price) * total_quantity
                gain_loss_pct = ((current_price - avg_buy_price) / avg_buy_price) * 100
            else:
                market_value = gain_loss = gain_loss_pct = "N/A"
            
            holdings_display_data.append({
                "Symbol": ticker,
                "Quantity": int(total_quantity),
                "Avg Buy Price": f"${avg_buy_price:.2f}",
                "Current Price": f"${current_price:.2f}" if current_price != "N/A" else "N/A",
                "Market Value": f"${market_value:,.2f}" if market_value != "N/A" else "N/A",
                "Gain/Loss": f"${gain_loss:,.2f}" if gain_loss != "N/A" else "N/A",
                "Gain/Loss %": f"{gain_loss_pct:.2f}%" if gain_loss_pct != "N/A" else "N/A",
            })
        
        holdings_display_df = pd.DataFrame(holdings_display_data)
        
        st.data_editor(
            holdings_display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small", disabled=True),
                "Quantity": st.column_config.NumberColumn("Quantity", width="small", disabled=True),
                "Avg Buy Price": st.column_config.TextColumn("Avg Buy Price", width="medium", disabled=True),
                "Current Price": st.column_config.TextColumn("Current Price", width="medium", disabled=True),
                "Market Value": st.column_config.TextColumn("Market Value", width="medium", disabled=True),
                "Gain/Loss": st.column_config.TextColumn("Gain/Loss", width="medium", disabled=True),
                "Gain/Loss %": st.column_config.TextColumn("Gain/Loss %", width="medium", disabled=True),
            }
        )

with st.expander("Trade", expanded=False):
    st.subheader("Trade Holdings")
    create_trading_interface(holdings_df=holdings_df)

if not holdings_df.empty:
    current_balance = portfolio_df['balance'].iloc[0] if not portfolio_df.empty else 35000
    pie_chart(holdings_df, current_balance)
    
    unique_tickers = holdings_df['ticker'].unique().tolist()
    if len(unique_tickers) >= 2:
        correlation_heatmap(unique_tickers)
    else:
        st.info("Add more stocks to your portfolio to see correlation analysis (minimum 2 stocks required)")

st.markdown("### 📈 Portfolio Performance Over Time")

transaction_count = len(transactions_df) if not transactions_df.empty else 0

if transaction_count > 0:
    with st.container(height=150, border=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### Chart Settings")
            INTERVALS = ["1d", "3d", "1wk", "1mo", "3mo", "6mo", "1y", "5y"]
            interval = st.selectbox("Interval", options=list(INTERVALS), index=3)
                    
        with col2:
            st.markdown("##### Technical Indicators")
            indicator_system()
    
    chart_result = create_performance_chart(interval)
    
    if chart_result:
        fig, change, change_pct = chart_result
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.metric(
                label=f"{interval} Change",
                value=f"${change:,.2f}",
                delta=f"{change_pct:.2f}%"
            )
        
        with perf_col2:
            history_df = calculate_portfolio_history()
            if not history_df.empty:
                avg_value = history_df['portfolio_value'].mean()
                st.metric(
                    label="Average Portfolio Value",
                    value=f"${avg_value:,.2f}"
                )
            else:
                st.metric(
                    label="Average Portfolio Value",
                    value="N/A"
                )
        
        with perf_col3:
            if not history_df.empty:
                max_value = history_df['portfolio_value'].max()
                st.metric(
                    label="All-Time High",
                    value=f"${max_value:,.2f}"
                )
            else:
                st.metric(
                    label="All-Time High", 
                    value="N/A"
                )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Performance Data", expanded=False):
            history_df = calculate_portfolio_history()
            if not history_df.empty:
                display_df = history_df.copy()
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['portfolio_value'] = display_df['portfolio_value'].apply(lambda x: f"${x:,.2f}")
                display_df['cash_balance'] = display_df['cash_balance'].apply(lambda x: f"${x:,.2f}")
                display_df.columns = ['Date/Time', 'Portfolio Value', 'Cash Balance']
                
                recent_df = display_df.tail(50)
                st.dataframe(recent_df, use_container_width=True, hide_index=True)
                
                if len(display_df) > 50:
                    st.caption(f"Showing last 50 entries out of {len(display_df)} total records")
else:
    st.info("📊 No transaction history available. Start trading to see your portfolio performance over time!")

@st.cache_data(ttl=300)
def get_risk_free_rate():

    try:
        treasury = yf.Ticker("^TNX")  
        data = treasury.history(period="5d")
        if not data.empty:
            return data['Close'].iloc[-1] / 100  
        else:
            return 0.045 
    except:
        return 0.045 

@st.cache_data(ttl=3600)
def get_sector_info(ticker):
   
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        return sector, industry
    except:
        return 'Unknown', 'Unknown'

@st.cache_data(ttl=3600)
def get_stock_beta(ticker):
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('beta', 1.0)
    except:
        return 1.0

@st.cache_data(ttl=3600)
def get_market_data(period='1y'):
   
    try:
      
        market_tickers = ['^GSPC', '^IXIC', '^DJI', '^RUT']  
        market_data = yf.download(market_tickers, period=period, progress=False)
        
        if not market_data.empty and isinstance(market_data.columns, pd.MultiIndex):
          
            close_data = market_data['Close']
            close_data.columns = ['S&P 500', 'NASDAQ', 'Dow Jones', 'Russell 2000']
            return close_data
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def calculate_portfolio_risk_metrics():
    """Calculate comprehensive risk metrics for the portfolio using actual data"""
    history_df = calculate_portfolio_history()
    
    if history_df.empty or len(history_df) < 2:
        return None
    
    
    history_df = history_df.sort_values('timestamp')
    history_df['daily_return'] = history_df['portfolio_value'].pct_change()
    

    returns = history_df['daily_return'].dropna()
    
    if len(returns) < 2:
        return None
    
 
    risk_free_rate = get_risk_free_rate()
    risk_free_daily = risk_free_rate / 252
    
    
    volatility = returns.std() * np.sqrt(252)  
    avg_return = returns.mean() * 252 
    sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
    
  
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    
    var_95 = np.percentile(returns, 5) * np.sqrt(252)
    
  
    holdings_summary_df = get_holdings_summary()
    if not holdings_summary_df.empty:
        total_value = (holdings_summary_df['current_price'] * holdings_summary_df['total_quantity']).sum()
        weighted_beta = 0
        
        for _, row in holdings_summary_df.iterrows():
            weight = (row['current_price'] * row['total_quantity']) / total_value
            stock_beta = get_stock_beta(row['ticker'])
            weighted_beta += weight * stock_beta
        
        beta = weighted_beta
    else:
        beta = 1.0
    
    return {
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'avg_return': avg_return,
        'beta': beta,
        'risk_free_rate': risk_free_rate
    }

def calculate_factor_exposures():
    
    holdings_summary_df = get_holdings_summary()
    
    if holdings_summary_df.empty:
        return {}
    
 
    tickers = holdings_summary_df['ticker'].unique().tolist()
    
   
    total_value = (holdings_summary_df['current_price'] * holdings_summary_df['total_quantity']).sum()
    weights = {}
    for _, row in holdings_summary_df.iterrows():
        weight = (row['current_price'] * row['total_quantity']) / total_value
        weights[row['ticker']] = weight
    

    market_data = get_market_data()
    
    if market_data.empty:
        return {'market_correlation': {}}
    
    # Download portfolio stocks data
    try:
        portfolio_data = yf.download(tickers, period='1y', progress=False)
        if portfolio_data.empty:
            return {'market_correlation': {}}
        
        # Handle single vs multiple stocks
        if len(tickers) == 1:
            portfolio_returns = portfolio_data['Close'].pct_change().dropna()
            # Create a weighted portfolio return (which is just the single stock)
            portfolio_return_series = portfolio_returns
        else:
            portfolio_closes = portfolio_data['Close']
            portfolio_returns = portfolio_closes.pct_change().dropna()
            
            # Calculate weighted portfolio returns
            weighted_returns = []
            for date, returns_row in portfolio_returns.iterrows():
                weighted_return = sum(returns_row[ticker] * weights[ticker] for ticker in tickers if not pd.isna(returns_row[ticker]))
                weighted_returns.append(weighted_return)
            
            portfolio_return_series = pd.Series(weighted_returns, index=portfolio_returns.index)
        
        # Calculate market correlations
        market_returns = market_data.pct_change().dropna()
        
        correlations = {}
        for market_name in market_returns.columns:
            # Align dates
            common_dates = portfolio_return_series.index.intersection(market_returns.index)
            if len(common_dates) > 10:  # Need sufficient data points
                portfolio_aligned = portfolio_return_series.loc[common_dates]
                market_aligned = market_returns[market_name].loc[common_dates]
                
                correlation = portfolio_aligned.corr(market_aligned)
                if not pd.isna(correlation):
                    correlations[market_name] = correlation
        
        return {'market_correlation': correlations}
    
    except Exception as e:
        st.error(f"Error calculating factor exposures: {e}")
        return {'market_correlation': {}}

def perform_stress_testing():
    """Perform stress testing using actual portfolio data and historical scenarios"""
    holdings_summary_df = get_holdings_summary()
    
    if holdings_summary_df.empty:
        return {}
    
    current_value = get_portfolio_value()
    scenarios = {}
    
    # Get portfolio tickers and weights
    tickers = holdings_summary_df['ticker'].unique().tolist()
    total_value = (holdings_summary_df['current_price'] * holdings_summary_df['total_quantity']).sum()
    
    try:
        # Download 2 years of data to calculate historical volatilities
        portfolio_data = yf.download(tickers, period='2y', progress=False)
        
        if portfolio_data.empty:
            return {}
        
        # Handle single vs multiple stocks
        if len(tickers) == 1:
            stock_data = portfolio_data[['Close']].copy()
            stock_data.columns = [tickers[0]]
        else:
            stock_data = portfolio_data['Close']
        
        # Calculate historical volatilities and correlations
        returns = stock_data.pct_change().dropna()
        
        # Scenario 1: Market Crash (based on actual 2008/2020 crash patterns)
        crash_impacts = {}
        for ticker in tickers:
            if ticker in returns.columns:
                # Use actual volatility to estimate crash impact
                stock_volatility = returns[ticker].std() * np.sqrt(252)
                # High volatility stocks tend to fall more in crashes
                crash_multiplier = min(2.5, 1.5 + (stock_volatility - 0.2))  # Scale based on volatility
                crash_impacts[ticker] = -0.20 * crash_multiplier  # Base 20% decline, adjusted by volatility
            else:
                crash_impacts[ticker] = -0.25  # Default if no data
        
        crash_impact_total = 0
        for _, row in holdings_summary_df.iterrows():
            ticker = row['ticker']
            market_value = row['current_price'] * row['total_quantity']
            impact = crash_impacts.get(ticker, -0.25)
            crash_impact_total += market_value * impact
        
        scenarios['Market Crash (Historical Pattern)'] = {
            'impact_dollar': crash_impact_total,
            'impact_percent': (crash_impact_total / current_value) * 100,
            'new_value': current_value + crash_impact_total
        }
        
        # Scenario 2: Interest Rate Spike (growth vs value impact)
        rate_spike_impacts = {}
        for ticker in tickers:
            try:
                # Get stock info to determine if it's growth or value
                stock_info = yf.Ticker(ticker).info
                pe_ratio = stock_info.get('trailingPE', 20)  # Default moderate P/E
                
                # High P/E stocks (growth) are more sensitive to interest rates
                if pe_ratio > 30:
                    rate_spike_impacts[ticker] = -0.18  # Growth stocks hit harder
                elif pe_ratio > 20:
                    rate_spike_impacts[ticker] = -0.12  # Moderate impact
                else:
                    rate_spike_impacts[ticker] = -0.05  # Value stocks less affected
            except:
                rate_spike_impacts[ticker] = -0.10  # Default moderate impact
        
        rate_spike_total = 0
        for _, row in holdings_summary_df.iterrows():
            ticker = row['ticker']
            market_value = row['current_price'] * row['total_quantity']
            impact = rate_spike_impacts.get(ticker, -0.10)
            rate_spike_total += market_value * impact
        
        scenarios['Interest Rate Spike'] = {
            'impact_dollar': rate_spike_total,
            'impact_percent': (rate_spike_total / current_value) * 100,
            'new_value': current_value + rate_spike_total
        }
        
        # Scenario 3: Sector Rotation (based on actual sector exposure)
        sector_rotation_impacts = {}
        for _, row in holdings_summary_df.iterrows():
            ticker = row['ticker']
            sector, _ = get_sector_info(ticker)
            
            # Simulate rotation out of growth sectors into value sectors
            if sector in ['Technology', 'Communication Services', 'Consumer Discretionary']:
                sector_rotation_impacts[ticker] = -0.15  # Growth sectors decline
            elif sector in ['Financials', 'Energy', 'Utilities', 'Real Estate']:
                sector_rotation_impacts[ticker] = 0.08   # Value sectors benefit
            else:
                sector_rotation_impacts[ticker] = -0.02  # Neutral sectors slightly down
        
        sector_rotation_total = 0
        for _, row in holdings_summary_df.iterrows():
            ticker = row['ticker']
            market_value = row['current_price'] * row['total_quantity']
            impact = sector_rotation_impacts.get(ticker, -0.02)
            sector_rotation_total += market_value * impact
        
        scenarios['Sector Rotation (Growth→Value)'] = {
            'impact_dollar': sector_rotation_total,
            'impact_percent': (sector_rotation_total / current_value) * 100,
            'new_value': current_value + sector_rotation_total
        }
        
    except Exception as e:
        st.error(f"Error in stress testing: {e}")
        return {}
    
    return scenarios

with st.expander("🔎 Portfolio Risk & Factor Analytics", expanded=False):
    if not holdings_df.empty and transaction_count > 0:
        risk_metrics = calculate_portfolio_risk_metrics()
        
        if risk_metrics:
            st.markdown("#### 📊 Risk Metrics")
            
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            
            with risk_col1:
                volatility_color = "normal"
                if risk_metrics['volatility'] > 0.25:
                    volatility_color = "inverse"
                elif risk_metrics['volatility'] < 0.15:
                    volatility_color = "off"
                    
                st.metric(
                    label="Volatility (σ)",
                    value=f"{risk_metrics['volatility']:.1%}",
                    help="Annualized volatility. S&P 500 ~15%, High-growth stocks ~30%+"
                )
            
            with risk_col2:
                sharpe_color = "normal"
                if risk_metrics['sharpe_ratio'] > 1.5:
                    sharpe_color = "off"
                elif risk_metrics['sharpe_ratio'] < 0:
                    sharpe_color = "inverse"
                    
                st.metric(
                    label="Sharpe Ratio",
                    value=f"{risk_metrics['sharpe_ratio']:.2f}",
                    help="Risk-adjusted return. >1 = Good, >2 = Very Good, <0 = Poor"
                )
            
            with risk_col3:
                st.metric(
                    label="Max Drawdown",
                    value=f"{risk_metrics['max_drawdown']:.1%}",
                    help="Largest peak-to-trough decline in portfolio value"
                )
            
            with risk_col4:
                st.metric(
                    label="Value at Risk (95%)",
                    value=f"{risk_metrics['var_95']:.1%}",
                    help="Maximum expected annual loss with 95% confidence"
                )
            
            # Additional metrics
            risk_col5, risk_col6, risk_col7, risk_col8 = st.columns(4)
            
            with risk_col5:
                st.metric(
                    label="Portfolio Beta",
                    value=f"{risk_metrics['beta']:.2f}",
                    help="Sensitivity to market movements. 1.0 = same as market"
                )
            
            with risk_col6:
                st.metric(
                    label="Expected Annual Return",
                    value=f"{risk_metrics['avg_return']:.1%}",
                    help="Annualized return based on historical performance"
                )
            
            with risk_col7:
                st.metric(
                    label="Current Risk-Free Rate",
                    value=f"{risk_metrics['risk_free_rate']:.1%}",
                    help="Current 10-Year Treasury yield used in calculations"
                )
            
            # Risk interpretation
            st.markdown("#### 🎯 Risk Assessment")
            
            risk_level = "Low"
            risk_color = "🟢"
            if risk_metrics['volatility'] > 0.25:
                risk_level = "High"
                risk_color = "🔴"
            elif risk_metrics['volatility'] > 0.20:
                risk_level = "Moderate"
                risk_color = "🟡"
            
            sharpe_assessment = "Poor"
            if risk_metrics['sharpe_ratio'] > 1.5:
                sharpe_assessment = "Excellent"
            elif risk_metrics['sharpe_ratio'] > 1.0:
                sharpe_assessment = "Good"
            elif risk_metrics['sharpe_ratio'] > 0:
                sharpe_assessment = "Fair"
            
            st.markdown(f"""
            **Overall Risk Level:** {risk_color} {risk_level}
            
            **Risk-Adjusted Performance:** {sharpe_assessment}
            """)
        
        # Factor Exposure Analysis
        st.markdown("#### 🏭 Factor & Market Exposure")
        
        factor_data = calculate_factor_exposures()
        
        if factor_data.get('market_correlation'):
            correlations = factor_data['market_correlation']
            
            # Display market correlations
            st.markdown("**Market Index Correlations:**")
            
            corr_cols = st.columns(len(correlations))
            for i, (index_name, correlation) in enumerate(correlations.items()):
                with corr_cols[i]:
                    # Color code based on correlation strength
                    if abs(correlation) > 0.7:
                        color = "🔴" if correlation > 0 else "🔵"
                        strength = "Strong"
                    elif abs(correlation) > 0.3:
                        color = "🟡"
                        strength = "Moderate"
                    else:
                        color = "🟢"
                        strength = "Weak"
                    
                    st.metric(
                        label=index_name,
                        value=f"{correlation:.2f}",
                        help=f"{strength} correlation with {index_name}"
                    )
            
            # Interpretation
            max_corr_index = max(correlations, key=correlations.get)
            max_corr_value = correlations[max_corr_index]
            
            if max_corr_value > 0.7:
                st.info(f"📊 **High correlation with {max_corr_index}** ({max_corr_value:.2f}) - Your portfolio moves closely with this index")
            elif max_corr_value > 0.3:
                st.info(f"📊 **Moderate correlation with {max_corr_index}** ({max_corr_value:.2f}) - Some alignment with this market segment")
            else:
                st.success("📊 **Low market correlation** - Your portfolio shows good diversification across market indices")
        
        # Style Analysis based on actual stock data
        st.markdown("#### 📈 Style Factor Analysis")
        
        holdings_summary_df = get_holdings_summary()
        if not holdings_summary_df.empty:
            # Get actual P/E ratios and market caps for style classification
            growth_weight = 0
            value_weight = 0
            large_cap_weight = 0
            small_cap_weight = 0
            total_weight = 0
            
            style_data = []
            
            for _, row in holdings_summary_df.iterrows():
                ticker = row['ticker']
                weight = row['current_price'] * row['total_quantity']
                total_weight += weight
                
                try:
                    stock_info = yf.Ticker(ticker).info
                    pe_ratio = stock_info.get('trailingPE', None)
                    market_cap = stock_info.get('marketCap', None)
                    
                    # Style classification based on P/E ratio
                    if pe_ratio and pe_ratio > 25:
                        growth_weight += weight
                        style = "Growth"
                    elif pe_ratio and pe_ratio < 15:
                        value_weight += weight
                        style = "Value"
                    else:
                        style = "Blend"
                    
                    # Market cap classification
                    if market_cap:
                        if market_cap > 10e9:  # > $10B
                            large_cap_weight += weight
                            cap_style = "Large Cap"
                        elif market_cap > 2e9:  # $2B - $10B
                            cap_style = "Mid Cap"
                        else:  # < $2B
                            small_cap_weight += weight
                            cap_style = "Small Cap"
                    else:
                        cap_style = "Unknown"
                    
                    style_data.append({
                        'ticker': ticker,
                        'weight': weight / total_weight * 100,
                        'style': style,
                        'cap_style': cap_style,
                        'pe_ratio': pe_ratio
                    })
                
                except Exception as e:
                    style_data.append({
                        'ticker': ticker,
                        'weight': weight / total_weight * 100,
                        'style': 'Unknown',
                        'cap_style': 'Unknown',
                        'pe_ratio': None
                    })
            
            growth_pct = (growth_weight / total_weight) * 100 if total_weight > 0 else 0
            value_pct = (value_weight / total_weight) * 100 if total_weight > 0 else 0
            large_cap_pct = (large_cap_weight / total_weight) * 100 if total_weight > 0 else 0
            small_cap_pct = (small_cap_weight / total_weight) * 100 if total_weight > 0 else 0
            
            style_col1, style_col2, style_col3 = st.columns(3)
            
            with style_col1:
                st.metric("Growth Exposure", f"{growth_pct:.1f}%")
            
            with style_col2:
                st.metric("Value Exposure", f"{value_pct:.1f}%")
            
            with style_col3:
                other_pct = 100 - growth_pct - value_pct
                st.metric("Other/Blend", f"{other_pct:.1f}%")
            
            # Market Cap Analysis
            cap_col1, cap_col2, cap_col3 = st.columns(3)
            
            with cap_col1:
                st.metric("Large Cap", f"{large_cap_pct:.1f}%")
            
            with cap_col2:
                mid_cap_pct = 100 - large_cap_pct - small_cap_pct
                st.metric("Mid Cap", f"{mid_cap_pct:.1f}%")
            
            with cap_col3:
                st.metric("Small Cap", f"{small_cap_pct:.1f}%")
            
            # Style DataFrame
            if style_data:
                style_df = pd.DataFrame(style_data)
                st.markdown("**Individual Stock Analysis:**")
                
                # Format the style dataframe for display
                display_style_df = style_df.copy()
                display_style_df['weight'] = display_style_df['weight'].apply(lambda x: f"{x:.1f}%")
                display_style_df['pe_ratio'] = display_style_df['pe_ratio'].apply(lambda x: f"{x:.1f}" if x is not None else "N/A")
                display_style_df.columns = ['Ticker', 'Weight', 'Style', 'Market Cap', 'P/E Ratio']
                
                st.dataframe(display_style_df, use_container_width=True, hide_index=True)
            
            # Style interpretation
            if growth_pct > 60:
                st.info("📈 **Growth-tilted portfolio** - Higher potential returns but more volatile")
            elif value_pct > 60:
                st.info("📊 **Value-tilted portfolio** - More defensive, potentially lower returns")
            else:
                st.info("⚖️ **Balanced style exposure** - Mix of growth and value characteristics")
        
        # Stress Testing
        st.markdown("#### 🧪 Stress Testing")
        
        scenarios = perform_stress_testing()
        
        if scenarios:
            st.markdown("**Market Scenario Analysis:**")
            
            for scenario_name, data in scenarios.items():
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{scenario_name}**")
                    
                    with col2:
                        color = "normal"
                        if data['impact_percent'] < -15:
                            color = "inverse"
                        elif data['impact_percent'] > 5:
                            color = "off"
                        
                        st.metric(
                            label="Impact",
                            value=f"{data['impact_percent']:.1f}%",
                            delta=f"${data['impact_dollar']:,.0f}"
                        )
                    
                    with col3:
                        st.metric(
                            label="New Portfolio Value",
                            value=f"${data['new_value']:,.0f}"
                        )
            
            st.markdown("##### Stress Test Methodology")
            st.caption("""
            • **Market Crash**: Impact based on historical volatility patterns - high volatility stocks decline more
            • **Interest Rate Spike**: Growth stocks (high P/E ratios) are more sensitive to rate changes  
            • **Sector Rotation**: Simulates rotation from growth to value sectors using real sector data
            """)
            
            # Risk Summary
            worst_scenario = min(scenarios.values(), key=lambda x: x['impact_percent'])
            best_scenario = max(scenarios.values(), key=lambda x: x['impact_percent'])
            
            st.markdown("#### 🎯 Stress Test Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Worst Case Scenario",
                    value=f"{worst_scenario['impact_percent']:.1f}%",
                    delta=f"${worst_scenario['impact_dollar']:,.0f}",
                    help="Most negative impact across all stress test scenarios"
                )
            
            with col2:
                st.metric(
                    label="Best Case Scenario", 
                    value=f"{best_scenario['impact_percent']:.1f}%",
                    delta=f"${best_scenario['impact_dollar']:,.0f}",
                    help="Most positive impact across all stress test scenarios"
                )
        
        # Portfolio Diversification Analysis
        st.markdown("#### 🎲 Diversification Analysis")
        
        holdings_summary_df = get_holdings_summary()
        if not holdings_summary_df.empty:
            # Calculate concentration metrics
            total_portfolio_value = (holdings_summary_df['current_price'] * holdings_summary_df['total_quantity']).sum()
            weights = []
            
            for _, row in holdings_summary_df.iterrows():
                weight = (row['current_price'] * row['total_quantity']) / total_portfolio_value
                weights.append(weight)
            
            # Herfindahl-Hirschman Index (HHI) for concentration
            hhi = sum(w**2 for w in weights)
            
            # Effective number of holdings
            effective_holdings = 1 / hhi if hhi > 0 else 0
            
            # Largest position
            max_weight = max(weights) if weights else 0
            
            div_col1, div_col2, div_col3 = st.columns(3)
            
            with div_col1:
                st.metric(
                    label="Concentration Index (HHI)",
                    value=f"{hhi:.3f}",
                    help="Lower is better. 0.2+ = concentrated, 0.1- = diversified"
                )
            
            with div_col2:
                st.metric(
                    label="Effective Holdings",
                    value=f"{effective_holdings:.1f}",
                    help="Number of equally-weighted holdings that would have same concentration"
                )
            
            with div_col3:
                st.metric(
                    label="Largest Position",
                    value=f"{max_weight:.1%}",
                    help="Weight of largest single holding"
                )
            
            # Diversification assessment
            if hhi > 0.25:
                st.warning("⚠️ **Highly concentrated portfolio** - Consider adding more positions")
            elif hhi > 0.15:
                st.info("ℹ️ **Moderately concentrated** - Some concentration risk present")
            else:
                st.success("✅ **Well diversified** - Good risk distribution across holdings")
    
    else:
        st.info("📊 Add holdings and build transaction history to see risk analytics. Risk metrics require at least 2 days of portfolio data.")

with st.expander("📊 Advanced Portfolio Analytics", expanded=False):
    if not holdings_df.empty:
        st.markdown("#### Recent Transaction History")
        
        recent_transactions = get_transaction_history(limit=20)
        
        if not recent_transactions.empty:
            display_transactions = []
            
            for _, row in recent_transactions.iterrows():
                if row['sold'] == 1:
                    if pd.notna(row['buy_timestamp']):
                        transaction_type = "Partial Sale"
                        timestamp = row['sell_timestamp']
                        price = row['sellprice']
                    else:
                        transaction_type = "Sale"
                        timestamp = row['sell_timestamp']
                        price = row['sellprice']
                else:
                    transaction_type = "Buy"
                    timestamp = row['buy_timestamp']
                    price = row['buyprice']
                
                display_transactions.append({
                    "Date": pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M') if pd.notna(timestamp) else "N/A",
                    "Type": transaction_type,
                    "Symbol": row['ticker'],
                    "Quantity": int(row['quantity']),
                    "Price": f"${price:.2f}" if pd.notna(price) else "N/A",
                    "Total": f"${row['quantity'] * price:.2f}" if pd.notna(price) else "N/A"
                })
            
            transactions_display_df = pd.DataFrame(display_transactions)
            st.dataframe(transactions_display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No transaction history available")
        
        st.markdown("#### Portfolio Composition")
        
        holdings_summary_df = get_holdings_summary()
        if not holdings_summary_df.empty:
            total_portfolio_value = holdings_summary_df['total_quantity'].multiply(
                holdings_summary_df['current_price']
            ).sum()
            
            composition_data = []
            for _, row in holdings_summary_df.iterrows():
                market_value = row['total_quantity'] * row['current_price']
                percentage = (market_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                
                composition_data.append({
                    "Symbol": row['ticker'],
                    "Market Value": f"${market_value:,.2f}",
                    "Portfolio %": f"{percentage:.1f}%",
                    "Shares": int(row['total_quantity'])
                })
            
            composition_df = pd.DataFrame(composition_data)
            st.dataframe(composition_df, use_container_width=True, hide_index=True)
            
            # Concentration risk analysis
            st.markdown("#### Concentration Risk Analysis")
            max_weight = max([float(x[:-1]) for x in composition_df['Portfolio %']])
            
            if max_weight > 30:
                st.warning(f"⚠️ **High concentration risk**: {max_weight:.1f}% in single position")
            elif max_weight > 20:
                st.info(f"ℹ️ **Moderate concentration**: {max_weight:.1f}% in largest position")
            else:
                st.success(f"✅ **Well diversified**: Largest position {max_weight:.1f}%")
                
        # Portfolio Performance Benchmarking
        st.markdown("#### 📈 Performance Benchmarking")
        
        history_df = calculate_portfolio_history()
        if not history_df.empty and len(history_df) > 30:  # Need sufficient data
            # Calculate portfolio returns
            history_df = history_df.sort_values('timestamp')
            history_df['portfolio_return'] = history_df['portfolio_value'].pct_change()
            
            # Get benchmark data (S&P 500)
            try:
                end_date = datetime.now()
                start_date = history_df['timestamp'].min()
                
                benchmark = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
                if not benchmark.empty:
                    benchmark['benchmark_return'] = benchmark['Close'].pct_change()
                    
                    # Align dates
                    portfolio_dates = pd.to_datetime(history_df['timestamp']).dt.date
                    benchmark_dates = benchmark.index.date
                    
                    # Calculate cumulative returns for comparison
                    portfolio_cumulative = (1 + history_df['portfolio_return'].fillna(0)).cumprod() - 1
                    benchmark_cumulative = (1 + benchmark['benchmark_return'].fillna(0)).cumprod() - 1
                    
                    # Current performance vs benchmark
                    portfolio_total_return = portfolio_cumulative.iloc[-1] * 100
                    benchmark_total_return = benchmark_cumulative.iloc[-1] * 100
                    outperformance = portfolio_total_return - benchmark_total_return
                    
                    bench_col1, bench_col2, bench_col3 = st.columns(3)
                    
                    with bench_col1:
                        st.metric(
                            label="Portfolio Return",
                            value=f"{portfolio_total_return:.1f}%",
                            help="Total return since first transaction"
                        )
                    
                    with bench_col2:
                        st.metric(
                            label="S&P 500 Return",
                            value=f"{benchmark_total_return:.1f}%",
                            help="Benchmark return over same period"
                        )
                    
                    with bench_col3:
                        delta_color = "normal"
                        if outperformance > 5:
                            delta_color = "off"
                        elif outperformance < -5:
                            delta_color = "inverse"
                        
                        st.metric(
                            label="Outperformance",
                            value=f"{outperformance:+.1f}%",
                            help="Portfolio return vs S&P 500 benchmark"
                        )
                    
                    # Performance interpretation
                    if outperformance > 10:
                        st.success("🎯 **Strong outperformance** - Portfolio significantly beating market")
                    elif outperformance > 0:
                        st.info("📈 **Modest outperformance** - Portfolio slightly ahead of market")
                    elif outperformance > -5:
                        st.info("📊 **Market-like performance** - Portfolio tracking close to benchmark")
                    else:
                        st.warning("📉 **Underperformance** - Portfolio lagging behind market")
                        
            except Exception as e:
                st.error(f"Error calculating benchmark comparison: {e}")
        else:
            st.info("Need more transaction history (30+ days) for performance benchmarking")
                
    else:
        st.info("Add holdings to your portfolio to see advanced analytics")

with st.expander("🔧 Database Management", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Create Database Backup", use_container_width=True):
            try:
                from utility import backup_database
                backup_file = backup_database()
                if backup_file:
                    st.success(f"✅ Backup created: {backup_file}")
                else:
                    st.error("❌ Backup failed")
            except ImportError:
                st.error("❌ backup_database function not found in utility.py")
    
    with col2:
        total_holdings = len(holdings_df)
        total_transactions = len(transactions_df)
        unique_stocks = holdings_df['ticker'].nunique() if not holdings_df.empty else 0
        
        st.metric("Total Holdings", total_holdings)
        st.metric("Total Transactions", total_transactions)
        st.metric("Unique Stocks", unique_stocks)
    
    with col3:
        import os
        if os.path.exists("data/trading_portfolio.db"):
            file_size = os.path.getsize("data/trading_portfolio.db")
            st.metric("Database Size", f"{file_size / 1024:.1f} KB")
            
            mod_time = datetime.fromtimestamp(os.path.getmtime("data/trading_portfolio.db"))
            st.caption(f"Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("Database file not found")

with st.expander("🚨 Portfolio Alerts & Recommendations", expanded=False):
    if not holdings_df.empty:
        st.markdown("#### Portfolio Health Check")
        
        alerts = []
        recommendations = []
        
        holdings_summary_df = get_holdings_summary()
        if not holdings_summary_df.empty:
            total_portfolio_value = (holdings_summary_df['current_price'] * holdings_summary_df['total_quantity']).sum()
            
            # Check for concentration risk
            for _, row in holdings_summary_df.iterrows():
                weight = (row['current_price'] * row['total_quantity']) / total_portfolio_value
                if weight > 0.3:
                    alerts.append(f"🚨 **High concentration**: {row['ticker']} represents {weight:.1%} of portfolio")
                    recommendations.append(f"Consider reducing {row['ticker']} position to below 20%")
            
            # Check for small positions
            small_positions = []
            for _, row in holdings_summary_df.iterrows():
                weight = (row['current_price'] * row['total_quantity']) / total_portfolio_value
                if weight < 0.02:  # Less than 2%
                    small_positions.append(row['ticker'])
            
            if len(small_positions) > 3:
                alerts.append(f"⚠️ **Many small positions**: {len(small_positions)} holdings under 2%")
                recommendations.append("Consider consolidating small positions to reduce complexity")
            
            # Check portfolio size
            num_holdings = len(holdings_summary_df)
            if num_holdings < 5:
                alerts.append("⚠️ **Low diversification**: Portfolio has fewer than 5 holdings")
                recommendations.append("Consider adding more positions to improve diversification")
            elif num_holdings > 30:
                alerts.append("⚠️ **Over-diversification**: Portfolio has many holdings")
                recommendations.append("Consider consolidating to 10-25 core positions")
            
            # Display alerts and recommendations
            if alerts:
                st.markdown("**Active Alerts:**")
                for alert in alerts:
                    st.markdown(f"• {alert}")
            else:
                st.success("✅ **No major portfolio alerts** - Portfolio structure looks healthy")
            
            if recommendations:
                st.markdown("**Recommendations:**")
                for rec in recommendations:
                    st.markdown(f"• {rec}")
            
            # Cash allocation check
            current_balance = portfolio_df['balance'].iloc[0] if not portfolio_df.empty else 0
            total_value = total_portfolio_value + current_balance
            cash_pct = (current_balance / total_value) * 100 if total_value > 0 else 0
            
            st.markdown("#### 💰 Cash Allocation")
            st.metric("Cash Percentage", f"{cash_pct:.1f}%")
            
            if cash_pct > 20:
                st.info("💡 **High cash allocation** - Consider investing excess cash")
            elif cash_pct < 5:
                st.warning("⚠️ **Low cash reserves** - Consider keeping some cash for opportunities")
            else:
                st.success("✅ **Balanced cash allocation** - Good liquidity buffer")
    
    else:
        st.info("Build your portfolio to see personalized alerts and recommendations")