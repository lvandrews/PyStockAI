#!/usr/bin/env python3

# 2022-10-01

#setup(
#    name='retrieve_stock_data.py',
#    version='0.1.0',
#    description='Data retrieval script for PyStockAI',
#    long_description=readme,
#    author='Lela Andrews',
#    author_email='lelavioletandrews@gmail.com',
#    url='https://github.com/lvandrews',
#    license=license,
#    packages=find_packages(exclude=('tests', 'docs'))
#)

# Program description and version
desctext='retrieve_stock_data.py: Obtain historical stock data and store for analysis.'
vers='retrieve_stock_data.py v0.1'

# Import libraries, find date
import argparse, sys, json, time, os, pprint
import datetime as dt
from alpha_vantage.timeseries import TimeSeries
from os.path import dirname, abspath
import yahoo_fin.stock_info as si
import yfinance as yf
import pandas as pd

today=dt.date.today()
year_ago=today - dt.timedelta(days=365)

# Define date string, repo directories, credentials
date_now_notime = time.strftime("%Y-%m-%d")
repodir = dirname(dirname(abspath(__file__)))
datadir = os.path.join(repodir,"data")
modeldir = os.path.join(repodir,"models")
scriptdir = os.path.join(repodir,"scripts")
av_creds = os.path.join(scriptdir,"creds.json")

# Initialize parser
parser = argparse.ArgumentParser(description=desctext)
parser.add_argument("-t", "--ticker", help="Ticker abbreviation (e.g. AMZN, required)", type=str, metavar="", required=True)
parser.add_argument("-s", "--source", help="Select data source; default = y1; y1 = yahoo_fin, y2 = yFinance", choices=["av", "y1", "y2"], type=str, metavar="", default="av")
parser.add_argument("-b", "--begin_date", help="Beginning date for analysis set (e.g. 2021-04-20, default = one year ago from present date)", type=str, metavar='', default=year_ago)
parser.add_argument("-a", "--all_time", help="Use all available data (supersedes -b)", action="store_true")
parser.add_argument("-d", "--data_type", help="Data type to retrieve (daily, daily_adj, intraday or intraday_ext; default=intraday)", choices=["daily", "daily_adj", "intraday", "intraday_ext"], type=str, metavar="", default="daily_adj")
parser.add_argument("-n", "--interval", help="Time interval between data points (intraday only; 1min, 5min, 15min, 30min, 60min); default=5min)", choices=["1min", "5min", "15min", "30min", "60min"], type=str, metavar="", default="5min")
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
dtype = args.data_type
ticker_output_dir = os.path.join(datadir,ticker,"")
ticker_data_filename = os.path.join(ticker_output_dir,"",f"{ticker}_{date_now_notime}_{dtype}.csv")
time_window = dtype
source = args.source
intvl = args.interval

# Make output directory if doesn't exist'
if not os.path.exists(ticker_output_dir):
    os.mkdir(ticker_output_dir)
       
# Print important model parameters
print("Ticker symbol:", ticker)
print("Data type:", dtype)
if dtype == "intraday":
    print("Time interval:", intvl)
if source == "av":
    print("API source: AlphaVantage")
elif source == "y1":
    print("API source: Yahoo Finance (yahoo_fin)")
elif source == "y2":
    print("API source: Yahoo Finance (yFinance)")    
   
# Retrieve data function AlphaVantage
if source == "av":
    def save_dataset(symbol, dtype):
        credentials = json.load(open(av_creds, 'r'))
        api_key = credentials['av_api_key']
        ts = TimeSeries(key=api_key, output_format='pandas')
        if dtype == 'intraday':
            data, meta_data = ts.get_intraday(symbol, interval=intvl, outputsize='full')
        elif dtype == 'daily':
            data, meta_data = ts.get_daily(symbol, outputsize='full')
        elif dtype == 'daily_adj':
            data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')
        elif dtype == 'intraday_ext':
            data, meta_data = ts.get_intraday_extended(symbol, interval='15min')

        data.to_csv(ticker_data_filename)
        
    save_dataset(ticker, dtype)
    
    # Rename data column headers
    df = pd.DataFrame()
    df = pd.read_csv(ticker_data_filename)

    if dtype == 'daily_adj':
        df.rename(columns={'date': 'Date', '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. adjusted close': 'AdjClose', '6. volume': 'Volume', '7. dividend amount': 'Dividend Amount', '8. split coefficient': 'Split Coefficient'}, inplace=True)
    elif dtype == 'daily':
        df.rename(columns={'date': 'Date', '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'}, inplace=True)
    elif dtype == 'intraday':
        df.rename(columns={'date': 'Date', '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'}, inplace=True)
    
    # Write output to file
    df.to_csv(ticker_data_filename)
    print(df)

    ### END ALPHAVANTAGE FUNCTION ###

# Retrieve data function Yahoo
if source == "y1":
    def save_dataset(symbol, time_window):
   
       # Load data from Yahoo Finance using specified range or all time
    #if args.all_time:
        df = si.get_data(ticker)

    #elif args.begin_date:
     #   df = si.get_data(ticker, start_date=args.begin_date, end_date=today)
#        credentials = json.load(open(av_creds, 'r'))
#        api_key = credentials['av_api_key']
#        print(symbol, time_window)
#        ts = TimeSeries(key=api_key, output_format='pandas')
#        if time_window == 'intraday':
#            data, meta_data = ts.get_intraday(symbol, interval='1min', outputsize='full')
#        elif time_window == 'daily':
#            data, meta_data = ts.get_daily(symbol, outputsize='full')
#        elif time_window == 'adj_daily':
#            data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')
#        elif time_window == 'intraday_ext':
#            data, meta_data = ts.get_time_series_intraday_extended(symbol, interval='15min')
#
        df.to_csv(ticker_data_filename)

    save_dataset(ticker, dtype)


# Retrieve data function Yahoo -- NEED TO FIX AS ALPHAVANTAGE NOW CHARGING FOR DAILY DATA --
if source == "y2":
    def save_dataset(symbol, time_window):

       # Load data from Yahoo Finance using specified range or all time
#        yf.download(ticker)
        df = yf.Ticker(ticker)
        df.history(period="max")
    


#        credentials = json.load(open(av_creds, 'r'))
#        api_key = credentials['av_api_key']
#        print(symbol, time_window)
#        ts = TimeSeries(key=api_key, output_format='pandas')
#        if time_window == 'intraday':
#            data, meta_data = ts.get_intraday(symbol, interval='1min', outputsize='full')
#        elif time_window == 'daily':
#            data, meta_data = ts.get_daily(symbol, outputsize='full')
#        elif time_window == 'adj_daily':
#            data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')
#        elif time_window == 'intraday_ext':
#            data, meta_data = ts.get_time_series_intraday_extended(symbol, interval='15min')
#
        d.to_csv(ticker_data_filename)

    save_dataset(ticker, dtype)
    
    
