# PyStockAI
Algorithmic stock price analysis and prediction.

This is merely a project to help me better learn python, not an actual useful utility.

Built in Linux Mint 20.1 within a VM on a mini PC with no CUDA cores.

## 2022-12-04 UPDATE:
### retrieve_stock_data.py (v0.1)
  ```usage: retrieve_stock_data.py [-h] -t  [-s] [-b] [-a] [-d] [-n] [-v] [-V]

  retrieve_stock_data.py: Obtain historical stock data and store for analysis.

  optional arguments:

    -h, --help:       Show help message and exit
    -t, --ticker:     Ticker abbreviation (e.g. AMZN, required)
    -s, --source:     Select data source; av = alphavantage, y1 = yahoo_fin, y2 = yFinance; default = av)
    -b, --begin_date: Beginning date for analysis set (e.g. 2021-04-20, default = one year ago from present date)
    -a, --all_time:   Use all available data (supersedes -b)
    -d, --data_type:  Data type to retrieve (daily, daily_adj, intraday or intraday_ext; default=intraday)
    -n, --interval:   Time interval between data points (intraday only; 1min, 5min, 15min, 30min, 60min); default=5min)
    -v, --version:    Show program version
    -V, --verbose:    Increase output verbosity
  ```

* Recent changes:
  * Changed -d default to daily_adj due to change in paywall at alphavantage
  * Built in column renaming from raw output from alphavantage (need to also do for yahoo sources)

* To-do list:
 * Fix yfinance and yahoo_fin options for data retrieval as alphavantage seems to have a paywall with moving target


### basic_analysis.py (v0.1)
* Cuurent functions:
  * Add technical indicator values to existing data from retrieve_stock_data.py. Requires current data available. First run retrieve_stock_data.py for ticker of interest.
    * -h, --help:       Show help message and exit
    * -t, --ticker:     Ticker abbreviation (e.g. AMZN, required)
    * -v, --version:    Show program version
    * -V, --verbose:    Increase output verbosity

* Recent changes:
  * Improved file handling:
    * Supplied ticker symbol (-t) causes program to look for available data in ticker abbreviation subdirectory, report on age of available files, select most recent between daily or daily_adj (use only one).

   * Copy latest data obtained by retrieve_stock_data.py
   * Calculate technical indicators
     * SMA (Simple Moving Average -- Automatically calculate for SMA = 5, 10, 20, 30, 60, 200) -- talib function MA or SMA
     * EMA (Exponential Moving Average) -- talib function
     * MACD (Moving Average Convergence/Divergence) -- talib function
     * RSI (Relative Strength Index) -- talib function
     * STOCH (Stochastic Oscillator) -- talib function
     * STOCHRSI (Stochastic Relative Strength Index) -- talib function
     * ADX (Average Directional Movement Index) -- talib function
     * BBANDS (Bollinger Bands) -- talib function
     * AD (Chaikin A/D Line) -- talib function
     * DX (Directional Movement Index) -- talib function
   * Add buy/sell function, test function over time, calculate best success strategy


 * tensor_analysis.py -- NOT YET BUILT --
   * Copy latest data from basic_analysis.py
   * Train/run model using different strategies
   * Predict next 1-10 days with confidence interval
   * Add buy/sell function, test function over time, calculate best success strategy
 * hmm_analysis.py -- NOT YET BUILT --
   * Similar analysis as in tenstor_analysis.py
   * Use HMM approach
 * Update visualize.py
   * Visualizations for each of the above
   * Allow output to private website
   * Multiple plots per screen
   * Date of analysis, raw data used on screen
   * Useful calculated values on screen
   * Change name of script to something more useful
 * Build monitor_analysis.py
   * Maintains analyses on current list of stock symbols
   * Uploads latest data to private website
   * Can run from cron

## Important notes
 * For those running a computer without CUDA architecture (requires CPU instead of GPU for ML processing), will need to compile tensorfow from source on your particular system
 * Models generated by CPU are known to be slightly inconsistent with those generated by GPU
 * alpha_vantage
   * Stable API
   * Same data as yfinance
   * Many technical indicators can be queried directly
   * Can pay for min 75 requests per minute with no daily limits for $49.99/mo
   * Many useful data types behind paywall
     * Pay if your model justifies the cost
 * yfinance
   * No longer stable source of data
   * Minimum amount of useful data for building models

## Installing
 1. Clone repo `git clone https://github.com/lvandrews/PyStockAI.git`
 1. Install python3 (apt, yum, etc)
 1. Install tensorflow and keras (see above about compiling for CPU)
     * Need to add directions in dependencies directory
 1. Install ta-lib dependencies in dependencies directory, follow instructions in associated README files
 1. Install python requirements with pip `pip install -r requirements`
 
## Obtain and analyze data
 1. Run retrieve_stock_data.py to obtain data for one particular stock
 1. Run basic_model.py to build out technical indicators
 1. Run tensor_analysis.py to build out ML-based predictions
 1. Run hmm_analysis.py to build out HMM-based predictions (script not yet built)
 1. Run visualize.py for graphical output and technical details
 1. Run monitor_analysis.py to update all symbols in analysis list (script not yet built)
     * Need to build list of symbols and maintenance scripts
 1. Observe output on private website (need to determine how to do this...)
 
