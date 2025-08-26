import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SMAStrategy:
    def __init__(self, ticker, short_window, long_window, initial_cash, position_size=None, position_size_pct=None):
        """
        Initialize SMA Crossover Strategy
        
        Parameters:
        ticker (str): Stock ticker symbol
        short_window (int): Short-term SMA window
        long_window (int): Long-term SMA window
        initial_cash (float): Initial cash amount
        position_size (int, optional): Fixed number of shares per trade
        position_size_pct (float, optional): Percentage of portfolio per trade
        """
        self.ticker = ticker
        self.short_window = short_window
        self.long_window = long_window
        self.initial_cash = initial_cash
        self.position_size = position_size
        self.position_size_pct = position_size_pct
        self.cash = initial_cash
        self.shares = 0
        self.positions = []
        self.trades = []
        self.portfolio_values = []
        self.benchmark_values = []
        
    def fetch_data(self, start_date, end_date):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(self.ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for {self.ticker}")
                
            # Calculate SMAs
            data['SMA_Short'] = data['Close'].rolling(window=self.short_window).mean()
            data['SMA_Long'] = data['Close'].rolling(window=self.long_window).mean()
            
            # Generate signals based on actual crossovers
            data['Signal'] = 0
            
            # Detect crossovers
            for i in range(1, len(data)):
                # Previous day's SMAs
                prev_short = data['SMA_Short'].iloc[i-1]
                prev_long = data['SMA_Long'].iloc[i-1]
                
                # Current day's SMAs
                curr_short = data['SMA_Short'].iloc[i]
                curr_long = data['SMA_Long'].iloc[i]
                
                # Golden Cross: Short SMA crosses above Long SMA (Buy Signal)
                if prev_short <= prev_long and curr_short > curr_long:
                    data.iloc[i, data.columns.get_loc('Signal')] = 1
                
                # Death Cross: Short SMA crosses below Long SMA (Sell Signal)
                elif prev_short >= prev_long and curr_short < curr_long:
                    data.iloc[i, data.columns.get_loc('Signal')] = -1
            
            # Remove NaN values
            data = data.dropna()
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")
    
    def execute_strategy(self, data):
        """Execute the SMA crossover strategy"""
        portfolio_values = []
        benchmark_values = []
        trades = []
        positions = []
        
        initial_price = data['Close'].iloc[0]
        shares = 0
        cash = self.initial_cash
        
        for i, (date, row) in enumerate(data.iterrows()):
            current_price = row['Close']
            signal = row['Signal']
            
            # Execute trades based on signals
            if signal == 1 and shares == 0:  # Buy signal
                # Calculate position size based on strategy
                if self.position_size is not None:
                    # Fixed shares strategy - ensure integer shares
                    max_shares_possible = int(cash / current_price)
                    trade_shares = min(self.position_size, max_shares_possible)
                elif self.position_size_pct is not None:
                    # Percentage of portfolio strategy - ensure integer shares
                    trade_shares = int((cash * self.position_size_pct / 100) / current_price)
                else:
                    # Default: use all available cash - ensure integer shares
                    trade_shares = int(cash / current_price)
                
                if trade_shares > 0:
                    trade_value = trade_shares * current_price
                    shares = trade_shares
                    cash -= trade_value
                    
                    trades.append({
                        'Date': date,
                        'Action': 'BUY',
                        'Price': current_price,
                        'Shares': trade_shares,
                        'Value': trade_value
                    })
                
            elif signal == -1 and shares > 0:  # Sell signal
                trade_value = shares * current_price
                cash += trade_value
                
                trades.append({
                    'Date': date,
                    'Action': 'SELL',
                    'Price': current_price,
                    'Shares': shares,
                    'Value': trade_value
                })
                shares = 0
            
            # Calculate current portfolio value
            portfolio_value = cash + (shares * current_price)
            portfolio_values.append({
                'Date': date,
                'Portfolio_Value': portfolio_value,
                'Cash': cash,
                'Shares': shares,
                'Share_Value': shares * current_price
            })
            
            # Calculate benchmark value (buy and hold)
            benchmark_value = (self.initial_cash / initial_price) * current_price
            benchmark_values.append({
                'Date': date,
                'Benchmark_Value': benchmark_value
            })
            
            # Record positions
            positions.append({
                'Date': date,
                'Cash': cash,
                'Shares': shares,
                'Portfolio_Value': portfolio_value,
                'Benchmark_Value': benchmark_value
            })
        
        self.portfolio_values = pd.DataFrame(portfolio_values)
        self.benchmark_values = pd.DataFrame(benchmark_values)
        self.trades = pd.DataFrame(trades)
        self.positions = pd.DataFrame(positions)
        
        return self.portfolio_values, self.benchmark_values, self.trades, self.positions
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if self.portfolio_values.empty:
            return {}
        
        final_portfolio = self.portfolio_values['Portfolio_Value'].iloc[-1]
        final_benchmark = self.benchmark_values['Benchmark_Value'].iloc[-1]
        
        # Total return
        total_return = ((final_portfolio - self.initial_cash) / self.initial_cash) * 100
        benchmark_return = ((final_benchmark - self.initial_cash) / self.initial_cash) * 100
        
        # Annualized return
        days = (self.portfolio_values['Date'].iloc[-1] - self.portfolio_values['Date'].iloc[0]).days
        annualized_return = ((final_portfolio / self.initial_cash) ** (365/days) - 1) * 100
        
        # Volatility
        daily_returns = self.portfolio_values['Portfolio_Value'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_returns = daily_returns - (risk_free_rate / 252)
        sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative = self.portfolio_values['Portfolio_Value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Win rate - calculate based on actual profit/loss from buy-sell pairs
        if not self.trades.empty:
            buy_trades = self.trades[self.trades['Action'] == 'BUY']
            sell_trades = self.trades[self.trades['Action'] == 'SELL']
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                profitable_count = 0
                total_sell_trades = 0
                
                for _, sell_trade in sell_trades.iterrows():
                    # Find the corresponding buy trade (most recent buy before this sell)
                    buy_trades_before = buy_trades[buy_trades['Date'] < sell_trade['Date']]
                    if len(buy_trades_before) > 0:
                        buy_trade = buy_trades_before.iloc[-1]  # Most recent buy
                        
                        # Calculate if this trade was profitable
                        buy_price = buy_trade['Price']
                        sell_price = sell_trade['Price']
                        
                        if sell_price > buy_price:
                            profitable_count += 1
                        total_sell_trades += 1
                
                if total_sell_trades > 0:
                    win_rate = (profitable_count / total_sell_trades) * 100
                else:
                    win_rate = 0
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        return {
            'Total_Return_%': round(total_return, 2),
            'Benchmark_Return_%': round(benchmark_return, 2),
            'Annualized_Return_%': round(annualized_return, 2),
            'Volatility_%': round(volatility, 2),
            'Sharpe_Ratio': round(sharpe_ratio, 2),
            'Max_Drawdown_%': round(max_drawdown, 2),
            'Win_Rate_%': round(win_rate, 2),
            'Final_Portfolio_Value': round(final_portfolio, 2),
            'Final_Benchmark_Value': round(final_benchmark, 2),
            'Total_Trades': len(self.trades) if not self.trades.empty else 0
        }
