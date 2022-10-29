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
desctext = 'retrieve_stock_data.py: Obtain historical stock data and store for analysis.'
vers='retrieve_stock_data.py v0.1'

# Parse inputs or provide help
import argparse, sys, json, time, os, pprint
import datetime as dt
from alpha_vantage.timeseries import TimeSeries
from os.path import dirname, abspath
from yahoo_fin import stock_info as si

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
parser.add_argument("-d", "--data_type", help="Data type to retrieve (daily, daily_adj (p), intraday or intraday_ext; default=intraday)", choices=["daily", "adj_daily", "intraday", "intraday_ext"], type=str, metavar="", default="intraday")
parser.add_argument("-n", "--interval", help="Time interval between data points (intraday only; 1min, 5min, 15min, 30min, 60min); default=5min)", choices=["1min", "5min", "15min", "30min", "60min"], type=str, metavar="", default="5min")
parser.add_argument("-s", "--source", help="Select data source; default=AlphaVantage", choices=["alphavantage", "yahoo"], type=str, metavar="", default="alphavantage")
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
if source == "alphavantage":
    print("API source: AlphaVantage")
elif source == "yahoo":
    print("API source: Yahoo Finance")
    
# Retrieve data function AlphaVantage
if source == "alphavantage":
    def save_dataset(symbol, time_window):
        credentials = json.load(open(av_creds, 'r'))
        api_key = credentials['av_api_key']
        ts = TimeSeries(key=api_key, output_format='pandas')
        if time_window == 'intraday':
            data, meta_data = ts.get_intraday(symbol, interval=intvl, outputsize='full')
        elif time_window == 'daily':
            data, meta_data = ts.get_daily(symbol, outputsize='full')
        elif time_window == 'adj_daily':
            print("\nadj_daily data disabled due to paywall at AlphaVantage\n ----- EXITING -----\n")
            quit()
            #data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')
        elif time_window == 'intraday_ext':
            print("\nintraday_ext data disabled\n ----- EXITING -----\n")
            quit()
        #    data, meta_data = ts.get_intraday_extended(symbol, interval='15min')

        data.to_csv(ticker_data_filename)
        
    save_dataset(ticker, dtype)

# Retrieve data function Yahoo
if source == "yahoo":
    print("\nYahoo finance calls disabled at this time\n ----- EXITING -----\n")
    quit()
#    def save_dataset(symbol, time_window):
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
#       data.to_csv(ticker_data_filename)

#    save_dataset(ticker, dtype)
    
