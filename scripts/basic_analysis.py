#!/usr/bin/env python3

# 2022-10-01

#setup(
#    name='basic_analysis.py',
#    version='0.1.0',
#    description='Add basic data analysis to stock data',
#    long_description=readme,
#    author='Lela Andrews',
#    author_email='lelavioletandrews@gmail.com',
#    url='https://github.com/lvandrews',
#    license=license,
#    packages=find_packages(exclude=('tests', 'docs'))
#)

# Program description and version
desctext='basic_analysis.py: Add technical indicator values to existing data from retrieve_stock_data.py. Requires current data available. First run retrieve_stock_data.py for ticker of interest.'
vers='basic_analysis.py v0.1'

# Parse inputs or provide help
import argparse, sys, json, time, os, pprint, glob
import datetime as dt
from os.path import dirname, abspath
import pandas as pd
import pandas_ta as ta
import talib
import shutil

# Define date string, repo directories, credentials
date_now_notime = time.strftime("%Y-%m-%d")
d0 = pd.to_datetime(date_now_notime)
repodir = dirname(dirname(abspath(__file__)))
datadir = os.path.join(repodir,"data")
modeldir = os.path.join(repodir,"models")
scriptdir = os.path.join(repodir,"scripts")

# Initialize parser
parser = argparse.ArgumentParser(description=desctext)
parser.add_argument("-t", "--ticker", help="Ticker abbreviation (e.g. DOW, required)", type=str, metavar="", required=True)
parser.add_argument("-s", "--strategy", help="Select strategy; default=ALL", choices=["ALL", "custom"], type=str, metavar="", default="ALL")
parser.add_argument("-v", "--version", help="Show program version", action="version", version=vers)
parser.add_argument("-V", "--verbose", help="Increase output verbosity", action="store_true")

# Print help if no arguments supplied
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)
    
# Read arguments from the command line and provide useful feedback
args = parser.parse_args()
if args.verbose:
    print("Verbosity turned on")

# Parse inputs and set ticker to uppercase if lowercase was entered
ticker = args.ticker.upper()
ticker_datadir = os.path.join(datadir,ticker,"")
strategy = args.strategy
print("\nBASIC ANALYSIS for:", ticker, "...\n", date_now_notime, "\n")

# Check for most recent daily or daily_adj file
lof_daily = glob.glob(os.path.join(ticker_datadir,"",f"{ticker}_*_daily.csv"))
if len(lof_daily) > 0:
    daily_newest = max(lof_daily)
    daily_newest_cdate = dt.datetime.fromtimestamp(os.path.getmtime(daily_newest))
    daily_newest_age = (d0 - daily_newest_cdate)
    daily_fname = os.path.basename(daily_newest)
    print("Latest daily data file is", daily_newest_age.days, "days old:", daily_fname)
    daily_data = daily_newest
    xx = 1
else:
    print("No daily data found...")
    xx = 0
    
lof_daily_adj = glob.glob(os.path.join(ticker_datadir,"",f"{ticker}_*_daily_adj.csv"))
if len(lof_daily_adj) > 0:
    daily_adj_newest = max(lof_daily_adj)
    daily_adj_newest_cdate = dt.datetime.fromtimestamp(os.path.getmtime(daily_adj_newest))
    daily_adj_newest_age = (d0 - daily_adj_newest_cdate)
    daily_adj_fname = os.path.basename(daily_adj_newest)
    print("Latest daily_adj data file is", daily_adj_newest_age.days, "days old:", daily_adj_fname)
    daily_data = daily_adj_newest
    xx = xx + 1
else:
    print("No daily_adj data found...")
    xx = xx + 0

if xx == 0:
    print("\nNo daily data found for", ticker, ". Exiting...\n")
    quit()

if xx == 2:
    daily_data = max(daily_newest,daily_adj_newest)

daily_data_fname = os.path.basename(daily_data)
daily_output = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_daily_basic_analysis.csv")

# Check if intraday exists
lof_intraday = glob.glob(os.path.join(ticker_datadir,"",f"{ticker}_*_intraday.csv"))
if len(lof_intraday) > 0:
    intraday_newest = max(lof_intraday)
    intraday_newest_cdate = dt.datetime.fromtimestamp(os.path.getmtime(intraday_newest))
    intraday_newest_age = (d0 - intraday_newest_cdate)
    intraday_fname = os.path.basename(intraday_newest)
    print("Latest intraday data file is", intraday_newest_age.days, "days old:", intraday_fname)
    intraday_data = intraday_newest
    intraday_data_fname = os.path.basename(intraday_data)
    intraday_output = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_intraday_basic_analysis.csv")
else:
    print("No intraday data found...")
    intraday_data_fname = "NONE"

# Report output:
print("\nDaily data source:   ", daily_data_fname, "\nIntraday data source:", intraday_data_fname, "\nStrategy:", strategy, "\n")


# Define available strategies
# Use ALL strategy for now (built-in to ta-lib). Refine to custom strategy in the future to conserve processing time and storage space. Move to separate file for defs if this becomes useful.
#quit()
# Custom Strategy definition
CustomStrategy = ta.Strategy(
    name="Custom Strategy",
    description="SMA20,SMA200, BBANDS, RSI, MACD and Volume SMA 20",
    ta=[
        {"kind": "sma", "length": 20},
        {"kind": "sma", "length": 100},
        {"kind": "ema", "length": 10},
        {"kind": "bbands", "length": 20},
        {"kind": "rsi"},
        {"kind": "macd", "fast": 8, "slow": 21},
        {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
    ]
)

# Function to signal when to buy and sell an asset
def buy_sell(signal):
    sigPriceBuy = []
    sigPriceSell = []
    flag = -1
    for i in range(0,len(signal)):
        #if sma30 > sma100  then buy else sell
            if signal['SMA30'][i] > signal['SMA100'][i]:
                if flag != 1:
                    sigPriceBuy.append(signal[ticker][i])
                    sigPriceSell.append(np.nan)
                    flag = 1
                else:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(np.nan)
                #print('Buy')
            elif signal['SMA30'][i] < signal['SMA100'][i]:
                if flag != 0:
                    sigPriceSell.append(signal[ticker][i])
                    sigPriceBuy.append(np.nan)
                    flag = 0
                else:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(np.nan)
                #print('sell')
            else: #Handling nan values
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
  
    return (sigPriceBuy, sigPriceSell)

# Create dataframes from inputs and add technical indicators, write to outputs
# Daily
print("Processing daily data...")
df_daily = pd.DataFrame()
df_daily = pd.read_csv(daily_data)
df_daily.set_index(pd.DatetimeIndex(df_daily['Date']), inplace=True)

if strategy == "ALL":
    df_daily.ta.strategy(ta.AllStrategy)
elif strategy == "custom":
    df_daily.ta.strategy(CustomStrategy)

df_daily.ta.log_return(cumulative=True, append=True)
df_daily.ta.percent_return(cumulative=True, append=True)
df_daily['SMA10'] = df_daily['Close'].rolling(window=10, min_periods=0).mean()
df_daily['SMA30'] = df_daily['Close'].rolling(window=30, min_periods=0).mean()
df_daily['SMA150'] = df_daily['Close'].rolling(window=150, min_periods=0).mean()

print(df_daily)

df_daily.to_csv(daily_output)

### REORT END
print("\n  --- DONE ---\n")
quit()

# Intraday
if len(lof_intraday) > 0:
    print("Processing intraday data...")
    df_intraday = pd.DataFrame()
    df_intraday = pd.read_csv(intraday_data)
    df_intraday.set_index(pd.DatetimeIndex(df_intraday['Date']), inplace=True)
    if strategy == "ALL":
        df_intraday.ta.strategy(ta.AllStrategy)
    elif strategy == "custom":
        df_intraday.ta.strategy(CustomStrategy)
    df_intraday.ta.log_return(cumulative=True, append=True)
    df_intraday.ta.percent_return(cumulative=True, append=True)

print(df_intraday)

df_intraday.to_csv(intraday_output)


### REORT END
print("\n  --- DONE ---\n")
quit()

