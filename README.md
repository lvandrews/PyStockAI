# PyStockAI
Algorithmic stock price analysis and prediction.

This is merely a project to help me better learn python, not an actual useful utility.

Built in Linux Mint 20.1 within a VM on a mini PC with no CUDA cores.

## 2022-12-04 UPDATES:
### retrieve_stock_data.py (v0.1)
  ```usage: retrieve_stock_data.py [-h] -t  [-s] [-b] [-a] [-d] [-n] [-v] [-V]

  retrieve_stock_data.py: Obtain historical stock data and store for analysis.

  optional arguments:

    -h, --help:       Show help message and exit
    -t, --ticker:     Ticker abbreviation (e.g. DOW, required)
    -s, --source:     Select data source; av = alphavantage, y1 = yahoo_fin, y2 = yFinance; default = av)
    -b, --begin_date: Beginning date for analysis set (e.g. 2021-04-20, default = one year ago from present date)
    -a, --all_time:   Use all available data (supersedes -b)
    -d, --data_type:  Data type to retrieve (daily, daily_adj, intraday or intraday_ext; default=intraday)
    -n, --interval:   Time interval between data points (intraday only; 1min, 5min, 15min, 30min, 60min); default=5min)
    -v, --version:    Show program version
    -V, --verbose:    Increase output verbosity
  ```
Notes:
 * Use of source "av" requires user to obtain API key from https://www.alphavantage.co/
 * Free tier API available
 * Store API key in "creds.json" file within PyStockAI/scripts/
 ```creds.json file contents:
 
 {"av_api_key":"API_KEY_HERE"}
 ```

Recent changes:
 * Changed -d default to daily_adj due to change in paywall at alphavantage
 * Built in column renaming from raw output from alphavantage (need to also do for yahoo sources)

To-do list:
 * Fix yfinance and yahoo_fin options for data retrieval as alphavantage seems to have a paywall with moving target
 * Consider adding logger function using module such as [loguru](https://github.com/Delgan/loguru) which may enable automatic user notification when a script hits an error -- particularly useful if using cron to keep analysis up to date.

### basic_analysis.py (v0.1)
  ```usage: basic_analysis.py [-h] -t  [-v] [-V]

  basic_analysis.py: Add technical indicator values to existing data from retrieve_stock_data.py.
  Requires current data available. First run retrieve_stock_data.py for ticker of interest.

  optional arguments:

    -h, --help:       Show help message and exit
    -t, --ticker:     Ticker abbreviation (e.g. DOW, required)
    -s, --strategy:   Select strategy; determines which indicators to use (default is ALL)
    -v, --version:    Show program version
    -V, --verbose:    Increase output verbosity
  ```

Recent changes:
 * Improved file handling:
   * Supplied ticker symbol (-t) causes program to look for available data in ticker abbreviation subdirectory, report on age of available files, select most recent between daily or daily_adj (use only one).
 * Copies latest data obtained by retrieve_stock_data.py
 * Calculates all technical indicators in ta-lib

To-do list:
 * Test available technical indicators for utility (as automated function)
 * Use results of above test to distill technical indicator calculations to save space, time
 * Add buy/sell function, test function over time, calculate best success strategy

### visualize.py (v0.1)
  ```usage: visualize.py [-h] -t  [-s] [-e] [-v] [-V]

  visualize.py: Generate visualizations from analysis of stock data.

  optional arguments:
    -h, --help      show this help message and exit
    -t , --ticker   Ticker abbreviation (e.g. DOW, required)
    -s , --start    Start date (e.g. 2020-04-20, default = a year ago)
    -e , --end      End date (e.g. 2021-04-20, default = today)
    -v, --version   show program version
    -V, --verbose   increase output verbosity
  ```
Recent changes:
 * none
 
To-do list:
 * Useful visualizations for output from basic_analysis.py
 * Allow output to private website
 * Multiple plots per screen
 * Date of analysis, raw data used on screen
 * Useful calculated values on screen
 * Change name of script to something more useful

### make_html.py (v0.1)
  ```usage: make_html.py [-h] -t  [-v] [-V]

  make_html.py: Build or update html page for viewing stock data.

  optional arguments:
    -h, --help      show this help message and exit
    -t , --ticker   Ticker abbreviation (e.g. DOW, required)
    -v, --version   show program version
    -V, --verbose   increase output verbosity
  ```
Recent changes:
 * none

To-do list:
 * Make output appealing
 * Decide where output should go

### tensor_analysis.py -- NOT YET BUILT --
 * Copy latest data from basic_analysis.py
 * Train/run model using different strategies
 * Predict next 1-10 days with confidence interval
 * Add buy/sell function, test function over time, calculate best success strategy

### hmm_analysis.py -- NOT YET BUILT --
 * Similar analysis as in tenstor_analysis.py
 * Use HMM approach


### monitor_analysis.py -- NOT YET BUILT --
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
 1. Install [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/) (see above about compiling for CPU)
     * Need to add directions in dependencies directory
 1. Install [ta-lib](https://github.com/mrjbq7/ta-lib) dependencies in dependencies directory, follow instructions in [PyStockAI/dependencies/README.ta-lib-0.4.0-src.txt](https://github.com/lvandrews/PyStockAI/blob/main/dependencies/README.ta-lib-0.4.0-src.txt)
 1. Install python requirements with pip `pip install -r requirements`
 
## Obtain and analyze data
 1. Run retrieve_stock_data.py to obtain data for one particular stock
 1. Run basic_model.py to build out technical indicators
 1. Run tensor_analysis.py to build out ML-based predictions (script not yet built, optional)
 1. Run other ML-based options if I get around to making them (scripts not yet build, optional)
 1. Run hmm_analysis.py to build out HMM-based predictions (script not yet built, optional)
 1. Run visualize.py for graphical output and technical details
 1. Run monitor_analysis.py to update all symbols in analysis list (script not yet built, optional)
     * Need to build list of symbols and maintenance scripts
 1. Observe output on private website (script not yet built, optional)
 
