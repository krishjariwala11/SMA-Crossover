# 📈 SMA Crossover Strategy  with backtesting Application

A comprehensive Python application that implements a Simple Moving Average (SMA) crossover trading strategy with a beautiful Streamlit web interface.


Check out the live app here: 
[![Streamlit App](https://img.shields.io/badge/🚀%20Open%20App-Streamlit-red?style=for-the-badge&logo=streamlit)](https://sma-krish.streamlit.app/)

## 🚀 Features

- **Interactive Strategy Parameters**: Customize short/long SMA windows, initial cash, and date ranges
- **Real-time Data**: Fetches live stock data using Yahoo Finance API
- **Comprehensive Analysis**: 
  - Stock price charts with SMA lines and buy/sell signals
  - Portfolio performance vs. benchmark comparison
  - Portfolio composition breakdown
  - Detailed trade history
- **Performance Metrics**: 
  - Total and annualized returns
  - Sharpe ratio and volatility
  - Maximum drawdown
  - Win rate and trade statistics
- **Beautiful Visualizations**: Interactive Plotly charts for better user experience
- **Responsive Design**: Optimized for both desktop and mobile viewing

## 📋 Requirements

- Python 3.8 or higher
- Internet connection for data fetching
- Required packages (see requirements.txt)

## 🛠️ How to Use

1. **Configure strategy parameters** in the sidebar:
   - **Stock Ticker**: Enter the stock symbol (e.g., AAPL, MSFT, GOOGL)
   - **Short SMA Window**: Short-term moving average period (default: 20)
   - **Long SMA Window**: Long-term moving average period (default: 50)
   - **Initial Cash**: Starting capital for the strategy
   - **Date Range**: Select start and end dates for backtesting

2. **Click "Run Strategy"** to execute the analysis

3. **Review results**:
   - Performance metrics at the top
   - Interactive charts showing strategy analysis
   - Trade details and history
   - Comprehensive performance comparison

## 📊 Strategy Logic

The SMA crossover strategy works as follows:

- **Buy Signal**: When the short-term SMA crosses above the long-term SMA
- **Sell Signal**: When the short-term SMA crosses below the long-term SMA
- **Position Management**: 

  - Partial positions or leverage is supported, which includes the number of assets, with a percentage of the available cash.

## 📈 Performance Metrics Explained

- **Total Return**: Overall percentage gain/loss from the strategy
- **Annualized Return**: Yearly return rate (annualized from the actual period)
- **Sharpe Ratio**: Risk-adjusted return measure (higher is better)
- **Volatility**: Annualized standard deviation of returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Benchmark Comparison**: Performance vs. buy-and-hold strategy

## 🔧 Customization

You can modify the strategy by editing `sma_strategy.py`:

- Change signal generation logic
- Add additional technical indicators
- Implement position sizing rules
- Add stop-loss mechanisms
- Include transaction costs

## 📱 Streamlit Features

- **Responsive Layout**: Automatically adjusts to different screen sizes
- **Interactive Elements**: Hover over charts for detailed information
- **Real-time Updates**: Instant results when parameters change
- **Professional Styling**: Clean, modern interface design

## ⚠️ Important Notes

- **Historical Data**: Results are based on historical data and may not predict future performance
- **No Transaction Costs**: The strategy assumes no trading fees or slippage
- **Data Availability**: Depends on Yahoo Finance data availability
- **Risk Disclaimer**: This is for educational purposes only - not financial advice

## 🐛 Troubleshooting

- **No Data Error**: Check if the ticker symbol is correct and data is available for the selected date range
- **Installation Issues**: Ensure you have the correct Python version and all dependencies installed
- **Performance Issues**: Large date ranges may take longer to process

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Yahoo Finance API](https://finance.yahoo.com/)
- [Technical Analysis Resources](https://www.investopedia.com/technical-analysis-4689657)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## 📄 License

This project is closed source and Krish Jariwala reserves all the rights.

---

**Happy Trading! 📈💰**


