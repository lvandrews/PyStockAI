#!/usr/bin/env python3

# 2021-04-05

#setup(
#    name='visualize',
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
import argparse, sys
import datetime as dt
import time

# Date strings
today=dt.date.today()
year_ago=today - dt.timedelta(days=365)
date_now = time.strftime("%Y-%m-%d_%H-%M-%S")
date_now_notime = time.strftime("%Y-%m-%d")
dt_string = time.strftime("%b-%d-%Y %I:%M:%S %p")

# Program description
desctext = 'visualize.py: Generate visualizations of stock data.'

# Initialize parser
parser = argparse.ArgumentParser(description=desctext)

parser.add_argument("-t", "--ticker", help="Ticker abbreviation (e.g. AMZN, required)", type=str, metavar='', required=True)
parser.add_argument("-s", "--start", help="Start date (e.g. 2020-04-20, default = a year ago)", type=str, metavar='', default=year_ago)
parser.add_argument("-e", "--end", help="End date (e.g. 2021-04-20, default = today)", type=str, metavar='', default=today)
parser.add_argument("-v", "--version", help="show program version", action="version", version="%(prog)s 0.1")
parser.add_argument("-V", "--verbose", help="increase output verbosity", action="store_true")


# Print help if no arguments supplied
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

# Read arguments from the command line
args = parser.parse_args()
    
# Provide useful feedback
args = parser.parse_args()
if args.verbose:
    print("Verbosity turned on")
    
parser.parse_args()

# Set ticker symbol to uppercase if lowercase was entered
ticker = args.ticker.upper()

# Import additional libraries
import matplotlib.pyplot as plt
plt.style.use('classic')
#%matplotlib inline
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si
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
df = si.get_data("AVXL", start_date=year_ago, end_date=today)

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