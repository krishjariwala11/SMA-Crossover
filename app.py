import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sma_strategy import SMAStrategy

# Page configuration
st.set_page_config(
    page_title="SMA Crossover Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 SMA Crossover Strategy</h1>', unsafe_allow_html=True)
    
    # Sidebar for parameters
    st.sidebar.header("Strategy Parameters")
    
    # Input parameters
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter stock ticker symbol (e.g., AAPL, MSFT, GOOGL)")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        short_window = st.number_input("Short SMA Window", min_value=5, max_value=50, value=20, help="Short-term moving average period")
    with col2:
        long_window = st.number_input("Long SMA Window", min_value=10, max_value=200, value=50, help="Long-term moving average period")
    
    initial_cash = st.sidebar.number_input("Initial Cash ($)", min_value=1000, max_value=1000000, value=10000, step=1000, help="Starting capital for the strategy")
    
    # Position Sizing Parameters
    st.sidebar.subheader("Position Sizing")
    position_type = st.sidebar.selectbox(
        "Position Size Type",
        ["Fixed Shares", "Percentage of Portfolio"],
        help="Choose how to determine position size for each trade"
    )
    
    if position_type == "Fixed Shares":
        position_size = st.sidebar.number_input("Number of Shares per Trade", min_value=1, max_value=10000, value=100, help="Fixed number of shares to buy/sell in each trade")
        position_size_pct = None
    else:
        position_size = None
        position_size_pct = st.sidebar.slider("Portfolio Percentage per Trade (%)", min_value=1, max_value=100, value=100, help="Percentage of available cash to use in each trade")
    
    # Date inputs
    st.sidebar.subheader("Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    start_date = st.sidebar.date_input("Start Date", value=start_date, max_value=end_date)
    end_date = st.sidebar.date_input("End Date", value=end_date, max_value=end_date)
    
    # Validation
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        return
    
    if long_window <= short_window:
        st.sidebar.error("Long SMA window must be greater than short SMA window")
        return
    
    # Run strategy button
    run_strategy = st.sidebar.button("🚀 Run Strategy", type="primary", use_container_width=True)
    
    if run_strategy:
        try:
            with st.spinner("Fetching data and executing strategy..."):
                # Initialize strategy
                strategy = SMAStrategy(ticker, short_window, long_window, initial_cash, position_size, position_size_pct)
                
                # Fetch data
                data = strategy.fetch_data(start_date, end_date)
                
                if data.empty:
                    st.error(f"No data available for {ticker} in the specified date range")
                    return
                
                # Execute strategy
                portfolio_values, benchmark_values, trades, positions = strategy.execute_strategy(data)
                
                # Calculate metrics
                metrics = strategy.calculate_metrics()
                
                # Display results
                display_results(data, portfolio_values, benchmark_values, trades, positions, metrics, ticker, initial_cash, position_size, position_size_pct)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your parameters and try again.")
    st.markdown(
    """
    <style>
    .footer {
        position: bottom;
        left: 0;
        bottom: 0;
        width: 100%;
        # background-color: #f0f2f6;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 25px;
    }
    </style>
    <div class="footer">
        © Krish Jariwala | All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True)

def display_results(data, portfolio_values, benchmark_values, trades, positions, metrics, ticker, initial_cash, position_size, position_size_pct):
    """Display all results in organized sections"""
    
    # Strategy Configuration
    st.header("⚙️ Strategy Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ticker", ticker)
    with col2:
        if position_size is not None:
            st.metric("Position Size", f"{position_size} shares")
        else:
            st.metric("Position Size", f"{position_size_pct}% of portfolio")
    with col3:
        st.metric("Initial Capital", f"${initial_cash:,.2f}")
    
    st.divider()
    
    # Performance Metrics
    st.header("📊 Performance Metrics")
    
    # Create metric columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{metrics['Total_Return_%']}%", 
                 delta=f"{metrics['Total_Return_%'] - metrics['Benchmark_Return_%']:.2f}% vs Benchmark")
    
    with col2:
        st.metric("Annualized Return", f"{metrics['Annualized_Return_%']}%")
    
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe_Ratio']:.2f}")
    
    with col4:
        st.metric("Max Drawdown", f"{metrics['Max_Drawdown_%']}%")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Volatility", f"{metrics['Volatility_%']}%")
    
    with col2:
        st.metric("Win Rate", f"{metrics['Win_Rate_%']}%")
    
    with col3:
        st.metric("Total Trades", metrics['Total_Trades'])
    
    with col4:
        st.metric("Final Portfolio Value", f"${metrics['Final_Portfolio_Value']:,.2f}")
    
    st.divider()
    
    # Charts Section
    st.header("📈 Strategy Analysis")
    
    # Create subplots for comprehensive analysis
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Stock Price & SMA Lines', 'Portfolio vs Benchmark Performance', 'Portfolio Composition'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Chart 1: Stock price and SMA lines
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Stock Price', line=dict(color='black')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_Short'], name=f'SMA {data["SMA_Short"].name.split("_")[-1]}', 
                  line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_Long'], name=f'SMA {data["SMA_Long"].name.split("_")[-1]}', 
                  line=dict(color='red')),
        row=1, col=1
    )
    
    # Add buy/sell signals
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]
    
    fig.add_trace(
        go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', 
                  name='Buy Signal', marker=dict(color='green', size=8, symbol='triangle-up')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', 
                  name='Sell Signal', marker=dict(color='red', size=8, symbol='triangle-down')),
        row=1, col=1
    )
    
    # Chart 2: Portfolio vs Benchmark
    fig.add_trace(
        go.Scatter(x=portfolio_values['Date'], y=portfolio_values['Portfolio_Value'], 
                  name='Strategy Portfolio', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=benchmark_values['Date'], y=benchmark_values['Benchmark_Value'], 
                  name='Buy & Hold Benchmark', line=dict(color='blue')),
        row=2, col=1
    )
    
    # Chart 3: Portfolio composition
    fig.add_trace(
        go.Scatter(x=portfolio_values['Date'], y=portfolio_values['Cash'], 
                  name='Cash', fill='tonexty', line=dict(color='lightblue')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=portfolio_values['Date'], y=portfolio_values['Share_Value'], 
                  name='Stock Value', fill='tonexty', line=dict(color='orange')),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"SMA Crossover Strategy Analysis for {ticker}",
        title_x=0.5
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Trade Details
    st.header("💼 Trade Details")
    
    if not trades.empty:
        # Create enhanced trades display with profit/loss calculation
        trades_display = trades.copy()
        trades_display['Date'] = trades_display['Date'].dt.strftime('%Y-%m-%d')
        trades_display['Price'] = trades_display['Price'].round(2)
        trades_display['Shares'] = trades_display['Shares'].astype(int)
        trades_display['Value'] = trades_display['Value'].round(2)
        
        # Calculate profit/loss for each trade
        trades_display['Profit_Loss'] = 0.0
        trades_display['Profit_Loss_%'] = 0.0
        
        # Group trades by buy-sell pairs
        buy_trades = trades_display[trades_display['Action'] == 'BUY'].copy()
        sell_trades = trades_display[trades_display['Action'] == 'SELL'].copy()
        
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            # Calculate profit/loss for completed trades
            for idx, sell_trade in sell_trades.iterrows():
                # Find the corresponding buy trade (most recent buy before this sell)
                buy_trades_before = buy_trades[buy_trades['Date'] < sell_trade['Date']]
                if len(buy_trades_before) > 0:
                    buy_trade = buy_trades_before.iloc[-1]  # Most recent buy
                    
                    # Calculate profit/loss based on price difference
                    buy_price = float(buy_trade['Price'])
                    sell_price = float(sell_trade['Price'])
                    shares_traded = min(float(buy_trade['Shares']), float(sell_trade['Shares']))
                    
                    # Calculate profit/loss
                    profit_loss = (sell_price - buy_price) * shares_traded
                    profit_loss_pct = ((sell_price - buy_price) / buy_price) * 100
                    
                    # Update the sell trade with profit/loss info using the original index
                    trades_display.loc[trades_display['Date'] == sell_trade['Date'], 'Profit_Loss'] = round(profit_loss, 2)
                    trades_display.loc[trades_display['Date'] == sell_trade['Date'], 'Profit_Loss_%'] = round(profit_loss_pct, 2)
        
        # Reorder columns for better display
        column_order = ['Date', 'Action', 'Price', 'Shares', 'Value', 'Profit_Loss', 'Profit_Loss_%']
        trades_display = trades_display[column_order]
        
        # Rename columns for better readability
        trades_display.columns = ['Date', 'Action', 'Price ($)', 'Shares', 'Value ($)', 'P&L ($)', 'P&L (%)']
        
        st.dataframe(trades_display, use_container_width=True)
        
        # Trade summary with profit/loss metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Buy Trades", len(trades[trades['Action'] == 'BUY']))
        with col2:
            st.metric("Total Sell Trades", len(trades[trades['Action'] == 'SELL']))
        with col3:
            total_pl = trades_display['P&L ($)'].sum()
            st.metric("Total P&L", f"${total_pl:,.2f}", 
                     delta=f"{total_pl:+.2f}" if total_pl != 0 else "0.00")
        with col4:
            st.metric("Total Trading Volume", f"${trades['Value'].sum():,.2f}")
        
        # # Profit/Loss Analysis
        # st.subheader("💰 Profit/Loss Analysis")
        
        # if len(sell_trades) > 0:
        #     profitable_trades = sell_trades[sell_trades['Profit_Loss'] > 0]
        #     losing_trades = sell_trades[sell_trades['Profit_Loss'] < 0]
            
        #     col1, col2, col3 = st.columns(3)
        #     with col1:
        #         st.metric("Profitable Trades", len(profitable_trades))
        #     with col2:
        #         st.metric("Losing Trades", len(losing_trades))
        #     with col3:
        #         if len(sell_trades) > 0:
        #             win_rate = (len(profitable_trades) / len(sell_trades)) * 100
        #             st.metric("Win Rate", f"{win_rate:.1f}%")
            
        #     # Show best and worst trades
        #     if len(sell_trades) > 0:
        #         best_trade = sell_trades.loc[sell_trades['Profit_Loss'].idxmax()]
        #         worst_trade = sell_trades.loc[sell_trades['Profit_Loss'].idxmin()]
                
        #         col1, col2 = st.columns(2)
        #         with col1:
        #             st.success(f"🏆 Best Trade: {best_trade['Date']} - ${best_trade['Profit_Loss']:,.2f} ({best_trade['Profit_Loss_%']:+.2f}%)")
        #         with col2:
        #             if worst_trade['Profit_Loss'] < 0:
        #                 st.error(f"📉 Worst Trade: {worst_trade['Date']} - ${worst_trade['Profit_Loss']:,.2f} ({worst_trade['Profit_Loss_%']:+.2f}%)")
        #             else:
        #                 st.info(f"📊 Worst Trade: {worst_trade['Date']} - ${worst_trade['Profit_Loss']:,.2f} ({worst_trade['Profit_Loss_%']:+.2f}%)")
    else:
        st.info("No trades were executed during this period.")
    
    st.divider()
    
    # Detailed Performance Analysis
    st.header("📋 Detailed Performance Analysis")
    
    # Create performance comparison table
    performance_data = {
        'Metric': ['Initial Investment', 'Final Portfolio Value', 'Final Benchmark Value', 
                  'Total Return', 'Benchmark Return', 'Excess Return', 'Annualized Return',
                  'Volatility', 'Sharpe Ratio', 'Maximum Drawdown', 'Win Rate'],
        'Strategy': [f"${initial_cash:,.2f}", f"${metrics['Final_Portfolio_Value']:,.2f}", 
                    f"${metrics['Final_Benchmark_Value']:,.2f}", f"{metrics['Total_Return_%']}%",
                    f"{metrics['Benchmark_Return_%']}%", 
                    f"{metrics['Total_Return_%'] - metrics['Benchmark_Return_%']:.2f}%",
                    f"{metrics['Annualized_Return_%']}%", f"{metrics['Volatility_%']}%",
                    f"{metrics['Sharpe_Ratio']:.2f}", f"{metrics['Max_Drawdown_%']}%",
                    f"{metrics['Win_Rate_%']}%"],
        'Benchmark': [f"${initial_cash:,.2f}", f"${metrics['Final_Benchmark_Value']:,.2f}",
                     f"${metrics['Final_Benchmark_Value']:,.2f}", f"{metrics['Benchmark_Return_%']}%",
                     f"{metrics['Benchmark_Return_%']}%", "0.00%",
                     f"{metrics['Benchmark_Return_%']}%", "N/A", "N/A", "N/A", "N/A"]
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)
    
    # Strategy insights
    st.subheader("💡 Strategy Insights")
    
    if metrics['Total_Return_%'] > metrics['Benchmark_Return_%']:
        st.success(f"✅ The SMA crossover strategy outperformed the buy-and-hold benchmark by "
                  f"{metrics['Total_Return_%'] - metrics['Benchmark_Return_%']:.2f} percentage points.")
    else:
        st.warning(f"⚠️ The SMA crossover strategy underperformed the buy-and-hold benchmark by "
                  f"{metrics['Benchmark_Return_%'] - metrics['Total_Return_%']:.2f} percentage points.")
    
    if metrics['Sharpe_Ratio'] > 1:
        st.success(f"✅ Good risk-adjusted returns with a Sharpe ratio of {metrics['Sharpe_Ratio']:.2f}")
    elif metrics['Sharpe_Ratio'] > 0:
        st.info(f"ℹ️ Positive risk-adjusted returns with a Sharpe ratio of {metrics['Sharpe_Ratio']:.2f}")
    else:
        st.warning(f"⚠️ Negative risk-adjusted returns with a Sharpe ratio of {metrics['Sharpe_Ratio']:.2f}")
    
    if abs(metrics['Max_Drawdown_%']) < 20:
        st.success(f"✅ Relatively low maximum drawdown of {metrics['Max_Drawdown_%']:.2f}%")
    else:
        st.warning(f"⚠️ High maximum drawdown of {metrics['Max_Drawdown_%']:.2f}% - consider risk management")

    

if __name__ == "__main__":
    main()
