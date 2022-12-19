#!/usr/bin/env python3

# 2021-04-05

#setup(
#    name='visualize.py',
#    version='0.1.0',
#    description='Visualize script for PyStockAI',
#    long_description=readme,
#    author='Lela Andrews',
#    author_email='lelavioletandrews@gmail.com',
#    url='https://github.com/lvandrews',
#    license=license,
#    packages=find_packages(exclude=('tests', 'docs'))
#)

# Parse inputs or provide help
import argparse, sys, time, os, pprint
import datetime as dt
from os.path import dirname, abspath
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Date strings
today=dt.date.today()
year_ago=today - dt.timedelta(days=365)
month_ago=today - dt.timedelta(days=30)
date_now = time.strftime("%Y-%m-%d_%H-%M-%S")
date_now_notime = time.strftime("%Y-%m-%d")
dt_string = time.strftime("%b-%d-%Y %I:%M:%S %p")

# Program description
desctext = 'visualize.py: Generate visualizations from analysis of stock data.'

# Define date string, repo directories, credentials
date_now_notime = time.strftime("%Y-%m-%d")
repodir = dirname(dirname(abspath(__file__)))
datadir = os.path.join(repodir,"data")
modeldir = os.path.join(repodir,"models")
scriptdir = os.path.join(repodir,"scripts")

# Initialize parser
parser = argparse.ArgumentParser(description=desctext)
parser.add_argument("-t", "--ticker", help="Ticker abbreviation (e.g. DOW, required)", type=str, metavar='', required=True)
parser.add_argument("-s", "--start", help="Start date (e.g. 2020-04-20, default = a year ago)", type=str, metavar='', default=year_ago)
parser.add_argument("-e", "--end", help="End date (e.g. 2021-04-20, default = today)", type=str, metavar='', default=today)
parser.add_argument("-v", "--version", help="Show program version", action="version", version="%(prog)s 0.1")
parser.add_argument("-V", "--verbose", help="Increase output verbosity", action="store_true")

# Print help if no arguments supplied
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

# Read arguments from the command line
args = parser.parse_args()
    
# Provide useful feedback
#parser.parse_args()
args = parser.parse_args()
if args.verbose:
    print("Verbosity turned on")

### NEED TO IMPROVE FILE HANDLING TO USE NEWEST AVAILABLE BASIC ANALYSES -- PRESENTLY LOOKS FOR TODAY ONLY
# Parse inputs and set ticker to uppercase if lowercase was entered
ticker = args.ticker.upper()
ticker_datadir = os.path.join(datadir,ticker,"")
daily_basic_analysis = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_daily_basic_analysis.csv")
if os.path.isfile(daily_basic_analysis):
    daily_basic_analysis_fname = os.path.basename(daily_basic_analysis)
else:
    daily_basic_analysis_fname = "MISSING. Check if inputs exist."
    
intraday_basic_analysis = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_intraday_basic_analysis.csv")
if os.path.isfile(intraday_basic_analysis):
    intraday_basic_analysis_fname = os.path.basename(intraday_basic_analysis)
else:
    intraday_basic_analysis_fname = "MISSING. Check if inputs exist."

# Report output:
print("\nDaily basic analysis source:   ", daily_basic_analysis_fname, "\nIntraday basic analysis source:", intraday_basic_analysis_fname, "\n")

# Initialize visualizers
plt.style.use('classic')
#sns.set()

yag = pd.to_datetime(year_ago, format='%Y-%m-%d')
mag = pd.to_datetime(month_ago, format='%Y-%m-%d')
#print(yag)
#quit()

# Generate output if daily_basic_analysis exists
if os.path.isfile(daily_basic_analysis):
    print("Generating visualizations of", ticker, "daily basic analysis input...\n")
    df_daily = pd.DataFrame()
    df_daily = pd.read_csv(daily_basic_analysis)
    #df_daily.set_index(pd.DatetimeIndex(df_daily['Date']), inplace=True)
    df_daily['Date'] = pd.to_datetime(df_daily['Date'], format='%Y-%m-%d')
    date = df_daily['Date']
    values = df_daily[['Open','Close','High','Low']]
    df_daily_short = df_daily.iloc[0:21]
    #df_daily_1yr = df_daily[df_daily['Date'] > date_now_notime - pd.Timedelta('365')]
    #print(df_daily_1yr)
    basic_output_sns = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_daily_basic_visualization_sns.png")
    basic_output_plt = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_daily_basic_visualization_plt.png")
    basic_output_plt_short = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_daily_basic_visualization_plt_shortterm.png")
    #df1 = pd.DataFrame(df_daily, df_daily.index, ["Open", "Close", "High", "Low"])
    #ax = sns.scatterplot(data=df1)
    #p01 = df_daily[['Date','Open','Close','High','Low']].set_index('Date').plot(figsize=(8,8))
    #df_daily_1yr = df_daily.loc[(df_daily['Date'] >= year_ago) & (df_daily['Date'] <= date_now_notime)]
    #df_daily_1yr = df_daily.query("Date >= yag and Date < = date_now_notime")
    
    #fig, axs = plt.subplots(figsize=(8, 6))
    #axs.set(ylabel="Value ($)",xlabel="Date")
    #axs.plot(date, values)
    #plt.xlim(mag)
    #plt.ylim(175,275)
    #plt.xticks(rotation = 45)
    #plt.legend(values, loc='upper left')
    #ax.plot(df_daily[['Date','Open','Close','High','Low']].set_index('Date').plot(figsize=(8,8)), df_daily[['Date']])
#    axs[0, 0].plot(df_daily[['Open','Close','High','Low']].set_index('Date').plot(figsize=(8,8)), df_daily[['Date']])
#    axs[0, 0].set_title('Axis [0, 0]')
#    axs[0, 1].plot(x, y, 'tab:orange')
#    axs[0, 1].set_title('Axis [0, 1]')
#    axs[1, 0].plot(x, -y, 'tab:green')
#    axs[1, 0].set_title('Axis [1, 0]')
#   axs[1, 1].plot(x, -y, 'tab:red')
#    axs[1, 1].set_title('Axis [1, 1]')

    #for ax in axs.flat:
        #ax.set(xlabel='x-label', ylabel='y-label')
    
#    p01 = df_daily[['Date','Open','Close','High','Low']].set_index('Date').plot(figsize=(8,8))
#    p01 = plt.ylabel('Value ($)')
#    p01 = plt.xlim(yag)
#    p01.invert_xaxis()
#    p01.set_xlim([dt.date(year_ago), dt.date(date_now_notime)])
#    plt.savefig(basic_output_plt)

    # Long-term plot (all time)
    fig, ax = plt.subplots(4, 1, figsize=(14, 18), sharex='col')
    #lt_plot_title = "Long-term data: {ticker}"
    df_daily[['Date','Open','Close','High','Low']].set_index('Date').plot(ax=ax[0],).legend(bbox_to_anchor=(1, 1.001), loc='upper left', borderaxespad=0, fontsize=12)
    fig.suptitle("Long-term data: " + ticker, fontsize=18)
    fig.subplots_adjust(right=0.8,left=0.05)
    ax1 = df_daily[['Date','RSI_14']].set_index('Date').plot(ax=ax[1])
#    ax1 = df_daily[['Date','RSI_14','STOCHRSIk_14_14_3_3','STOCHRSId_14_14_3_3']].set_index('Date').plot(ax=ax[1])
    X = ax1.axhline(70, color="grey")
    ax1.axhline(30, color="grey")
    ax1.set_ylim(ymin=0, ymax=100)
    ax1.axhspan(70, 100, color = "red", alpha=0.1)
    ax1.axhspan(0, 30, color = "green", alpha=0.1)
    ax1.legend(bbox_to_anchor=(1, 1.001), loc='upper left', borderaxespad=0, fontsize=12)
    df_daily[['Date','High','Low','SMA10','SMA30','SMA150']].set_index('Date').plot(ax=ax[2]).legend(bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0, fontsize=12)
    df_daily[['Date','OHLC4','BBU_5_2.0','BBL_5_2.0']].set_index('Date').plot(ax=ax[3]).legend(bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0, fontsize=12)
    
    plt.savefig(basic_output_plt)
    
    # Short-term plot (30 days)
    fig, ax = plt.subplots(4, 1, figsize=(14, 18), sharex='col')
    df_daily_short[['Date','Open','Close','High','Low']].set_index('Date').plot(ax=ax[0]).legend(bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0, fontsize=12)
    fig.suptitle("30 day data: " + ticker, fontsize=18)
    fig.subplots_adjust(right=0.8,left=0.05)
    ax1 = df_daily_short[['Date','RSI_14']].set_index('Date').plot(ax=ax[1])
#    ax1 = df_daily_short[['Date','RSI_14','STOCHRSIk_14_14_3_3','STOCHRSId_14_14_3_3']].set_index('Date').plot(ax=ax[1])
    ax1.axhline(70, color="grey")
    ax1.axhline(30, color="grey")
    ax1.legend(bbox_to_anchor=(1, 1.001), loc='upper left', borderaxespad=0, fontsize=12)
    ax1.set_ylim(ymin=0, ymax=100)
    ax1.axhspan(70, 100, color = "red", alpha=0.1)
    ax1.axhspan(0, 30, color = "green", alpha=0.1)
    df_daily_short[['Date','High','Low','SMA10','SMA30','SMA150']].set_index('Date').plot(ax=ax[2]).legend(bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0, fontsize=12)
    df_daily_short[['Date','Close','OHLC4','BBU_5_2.0','BBM_5_2.0','BBL_5_2.0']].set_index('Date').plot(ax=ax[3]).legend(bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0, fontsize=12)
    
    plt.savefig(basic_output_plt_short)


quit()
# Generate output if intraday_basic_analysis exists
if os.path.isfile(intraday_basic_analysis):
    df_intraday = pd.DataFrame()
    df_intraday = pd.read_csv(intraday_basic_analysis)



### END
print("\n  --- DONE ---\n")
quit()

## OLD CODE BELOW HERE ###

# Add plot RSI, close from code example, update x axis to date, scale according to time input
# Or... All time plot plus last 1 year plot?

# Import additional libraries
#%matplotlib inline

# Print useful information
print("Ticker symbol:", ticker)
print("Date range:", year_ago, " - ", today)

# Create some random data
#rng = np.random.RandomState(0)
#x_rng = np.linspace(0, 10, 500)
#y_rng = np.cumsum(rng.randn(500, 6), 0)

# Get some stock data for the last year for AVXL
df = si.get_data(ticker, start_date=year_ago, end_date=today)

df["date"] = df.index
df["difference"] = df["close"] - df["open"]

# Plot the data with Matplotlib defaults
#plt.plot(x_rng, y_rng)
#plt.legend('ABCDEF', ncol=2, loc='upper left');
#plt.plot_date(df.date, df.difference)

df1 = pd.DataFrame(df, df.index, ["open", "close", "high", "low"])
ax = sns.scatterplot(data=df1)

#ax = sns.scatterplot(x=df.index,y=df.difference,hue=df.close)
ax.legend(loc='upper left')
#sns.pairplot(df, hue='difference')

plt.show()


quit()
import plotly.graph_objects as go

fig = go.Figure(
    data=go.Ohlc(
        x=df_out.index,
        open=df_out["open"],
        high=df_out["high"],
        low=df_out["low"],
        close=df_out["close"],
    )
)
fig.show()
