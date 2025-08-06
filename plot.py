
import os   
import metrics as met

#Finance
import yfinance as yf

#Visualization
#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
#import seaborn as sns
import streamlit as st

#Data and Calculations
import numpy as np
import ta
import pandas as pd
import datetime as dt


def graph(fsymbols, fperiod, finterval):
    """
    Plots closing prices and selected moving averages for one or more stocks.
    """
    if isinstance(fsymbols, str):
        fsymbols = [fsymbols]

    df = yf.download(fsymbols, period=fperiod, interval=finterval, repair=True, group_by='ticker')
    plot_df = []

    is_multilevel = isinstance(df.columns, pd.MultiIndex)
    
    for symbol in fsymbols:
        data = None 
        
        if is_multilevel:
            if symbol in df.columns.get_level_values(0):
                data = df[symbol].copy()
        else:
            data = df.copy()
        
        if data is None or data.empty or 'Close' not in data.columns:
            st.error(f"Error: 'Close' column not found for symbol {symbol} or data is empty. Skipping.")
            continue
            
        data = data.reset_index()
        data['Symbol'] = symbol

        indicators = st.session_state.get('indicators', [])
        
        for ind_type, ind_period in indicators:
            if ind_type == "SMA":
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
                bb_data = met.ADX(data, period=ind_period)
                data[f'ADX-{ind_period}'] = bb_data['adx']
                data[f'+DI-{ind_period}'] = bb_data['+DI']
                data[f'-DI-{ind_period}'] = bb_data['-DI']
        
        plot_df.append(data)

    if not plot_df:
        st.error("No data available to plot.")
        return

    plot_df = pd.concat(plot_df, ignore_index=True)

    if 'Date' not in plot_df.columns:
        plot_df = plot_df.reset_index()
        if 'index' in plot_df.columns:
            plot_df = plot_df.rename(columns={'index': 'Date'})

    fig = go.Figure()
    
    indicators = st.session_state.get('indicators', [])
    COLOR_PALETTE = ['#ff4081', '#ffd700', '#00e676', '#7c4dff', '#00bcd4', "#ff9900", '#e040fb', "#00fff2", '#ff1744', '#00e5ff']
    color_idx = 0
    
    for symbol in fsymbols:
        symbol_data = plot_df[plot_df['Symbol'] == symbol].copy()
        if symbol_data.empty:
            continue
            
        fig.add_trace(go.Scatter(
            x=symbol_data['Date'],
            y=symbol_data['Close'],
            name=f'{symbol} Close',
            line=dict(width=2)
        ))
        
        for ind_type, ind_period in indicators:
            color = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
            
            if ind_type == "SMA":
                col_name = f'SMA-{ind_period}'
                if col_name in symbol_data.columns:
                    fig.add_trace(go.Scatter(
                        x=symbol_data['Date'],
                        y=symbol_data[col_name],
                        name=f'{symbol} {col_name}',
                        line=dict(color=color, width=1.5, dash='dot')
                    ))
                    
            elif ind_type == "EMA":
                col_name = f'EMA-{ind_period}'
                if col_name in symbol_data.columns:
                    fig.add_trace(go.Scatter(
                        x=symbol_data['Date'],
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
                        x=symbol_data['Date'],
                        y=symbol_data[macd_col],
                        yaxis='y2',
                        name=f'{symbol} MACD-{ind_period}',
                        line=dict(color=color, width=2)
                    ))
                    
                    signal_color = COLOR_PALETTE[(color_idx + 1) % len(COLOR_PALETTE)]
                    fig.add_trace(go.Scatter(
                        x=symbol_data['Date'],
                        y=symbol_data[signal_col],
                        yaxis='y2',
                        name=f'{symbol} MACD Signal-{ind_period}',
                        line=dict(color=signal_color, width=2, dash='dash')
                    ))

                    hist_color = COLOR_PALETTE[(color_idx + 2) % len(COLOR_PALETTE)]
                    fig.add_trace(go.Bar(
                        x=symbol_data['Date'],
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
                        x=symbol_data['Date'],
                        y=symbol_data[col_name],
                        yaxis='y2',
                        name=f'{symbol} {col_name}',
                        line=dict(color=color, width=1.5)
                    ))
                    
            elif ind_type == "STO":
                col_name = f'STO-{ind_period}'
                if col_name in symbol_data.columns:
                    fig.add_trace(go.Scatter(
                        x=symbol_data['Date'],
                        y=symbol_data[col_name],
                        name=f'{symbol} {col_name}',
                        line=dict(color=color, width=1.5)
                    ))

            elif ind_type == "ATR":
                col_name = f'ATR-{ind_period}'
                if col_name in symbol_data.columns:
                    fig.add_trace(go.Scatter(
                        x=symbol_data['Date'],
                        y=symbol_data[col_name],
                        yaxis='y2',
                        name=f'{symbol} {col_name}',
                        line=dict(color=color, width=1.5)
                    ))

            elif ind_type == "OBV":
                col_name = f'OBV-{ind_period}'
                if col_name in symbol_data.columns:
                    fig.add_trace(go.Scatter(
                        x=symbol_data['Date'],
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
                        x=symbol_data['Date'],
                        y=symbol_data[upper_col],
                        name=f'{symbol} BB Upper-{ind_period}',
                        line=dict(color=color, width=1),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=symbol_data['Date'],
                        y=symbol_data[lower_col],
                        name=f'{symbol} BB Lower-{ind_period}',
                        line=dict(color=color, width=1),
                        fill='tonexty',
                        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=symbol_data['Date'],
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
                        x=symbol_data['Date'],
                        y=symbol_data[adx],
                        yaxis='y2',
                        name=f'{symbol} ADX-{ind_period}',
                        line=dict(color=color, width=1),
                        showlegend=False
                    ))

                    pos_color = COLOR_PALETTE[color_idx + 1 % len(COLOR_PALETTE)]
                    fig.add_trace(go.Scatter(
                        x=symbol_data['Date'],
                        y=symbol_data[di_plus],
                        yaxis='y2',
                        name=f'{symbol} +DI-{ind_period}',
                        line=dict(color=pos_color, width=1),
                        showlegend=False
                    ))

                    neg_color = COLOR_PALETTE[color_idx + 2 % len(COLOR_PALETTE)]
                    fig.add_trace(go.Scatter(
                        x=symbol_data['Date'],
                        y=symbol_data[di_minus],
                        yaxis='y2',
                        name=f'{symbol} -DI-{ind_period}',
                        line=dict(color=neg_color, width=1),
                        showlegend=False
                    ))

                    color_idx += 2
            
            color_idx += 1

    fig.update_layout(
        title=f"Closing Prices and Moving Averages: {', '.join(fsymbols)}",
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
        
        if ind_type == "SMA":
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
            df[f'ATR-{ind_period}'] = met.OBV(df, period=ind_period)
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

    df = yf.download(fsymbols, period=fperiod)
    returns = df['Close'].pct_change(fill_method=None).dropna()
    correlation = returns.corr(method='pearson')

    fig = px.imshow(correlation, text_auto=True, aspect='auto', color_continuous_scale='RdBu')
    fig.update_layout(
        title='Correlation Heatmap',
        xaxis_title='Stocks',
        yaxis_title='Stocks',
        coloraxis_colorbar=dict(title='Correlation Coefficient'),
    )
    st.plotly_chart(fig, use_container_width=True)

