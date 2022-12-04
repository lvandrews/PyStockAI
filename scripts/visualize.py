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
import argparse, sys, json, time, os, pprint
import datetime as dt
from os.path import dirname, abspath
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as mplt
import seaborn as sbn

# Date strings
today=dt.date.today()
year_ago=today - dt.timedelta(days=365)
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
parser.add_argument("-t", "--ticker", help="Ticker abbreviation (e.g. AMZN, required)", type=str, metavar='', required=True)
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

# Generate output if daily_basic_analysis exists
if os.path.isfile(daily_basic_analysis):
    print("Generating visualizations of", ticker, "daily basic analysis input...\n")
    df_daily = pd.DataFrame()
    df_daily = pd.read_csv(daily_basic_analysis)



quit()
# Generate output if intraday_basic_analysis exists
if os.path.isfile(intraday_basic_analysis):
    df_intraday = pd.DataFrame()
    df_intraday = pd.read_csv(intraday_basic_analysis)



### END
print("\n  --- DONE ---\n")
quit()

## OLD CODE BELOW HERE ###


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
ticker_basicanal_daily = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_daily_basic_analysis.csv")
ticker_basicanal_intraday = os.path.join(ticker_datadir,"",f"{ticker}_{date_now_notime}_intraday_basic_analysis.csv")
df_daily = pd.DataFrame()
df_daily = pd.read_csv(ticker_basicanal_daily)








quit()
### 

# Add plot RSI, close from code example, update x axis to date, scale according to time input
# Or... All time plot plus last 1 year plot?

###


# Import additional libraries
import matplotlib.pyplot as plt
plt.style.use('classic')
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

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
