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
import argparse, sys, json, time, os, pprint
import datetime as dt
from os.path import dirname, abspath
import pandas as pd
import pandas_ta as ta
import talib
import shutil

# Define date string, repo directories, credentials
date_now_notime = time.strftime("%Y-%m-%d")
repodir = dirname(dirname(abspath(__file__)))
datadir = os.path.join(repodir,"data")
modeldir = os.path.join(repodir,"models")
scriptdir = os.path.join(repodir,"scripts")

# Initialize parser
parser = argparse.ArgumentParser(description=desctext)
parser.add_argument("-t", "--ticker", help="Ticker abbreviation (e.g. AMZN, required)", type=str, metavar="", required=True)
#parser.add_argument("-s", "--source", help="Select data source; default=AlphaVantage", choices=["alphavantage", "yahoo"], type=str, metavar="", default="alphavantage")
parser.add_argument("-v", "--version", help="show program version", action="version", version=vers)
parser.add_argument("-V", "--verbose", help="increase output verbosity", action="store_true")

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
ticker_input_daily = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_daily.csv")
ticker_input_intraday = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_intraday.csv")

# Check that inputs exist -- consider moving as def to util.py
if not os.path.isfile(ticker_input_daily):
    most_recent_daily = "MISSING"
    print("Missing input file:",ticker_input_daily)
    print("\nExiting\n")
    quit()
else:
    most_recent_daily = "AVAILABLE"
    

if not os.path.isfile(ticker_input_intraday):
    most_recent_intraday = "MISSING"
    print("Missing input file:",ticker_input_intraday)
    print("\nExiting\n")
    quit()
else:
    most_recent_intraday = "AVAILABLE"

print("\nInput files exist:\n",ticker_input_daily,"\n",ticker_input_intraday,"\n")

# Copy inputs to basic analysis files
ticker_output_daily = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_daily_basic_analysis.csv")
ticker_output_intraday = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_intraday_basic_analysis.csv")
#shutil.copyfile(ticker_input_daily,ticker_output_daily)
#shutil.copyfile(ticker_input_intraday,ticker_output_intraday)

# Create dataframes from inputs and add technical indicators, write to outputs
df_daily = pd.DataFrame()
df_daily = pd.read_csv(ticker_input_daily)
df_daily.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'}, inplace=True)
df_daily.ta.log_return(cumulative=True, append=True)
df_daily.ta.percent_return(cumulative=True, append=True)

print(df_daily)

df_daily.to_csv(ticker_output_daily)















quit()
