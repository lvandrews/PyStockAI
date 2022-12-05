#!/usr/bin/env python3

# 2022-11-10

#setup(
#    name='make_html.py',
#    version='0.1.0',
#    description='Build basic html page to view stock data.',
#    long_description=readme,
#    author='Lela Andrews',
#    author_email='lelavioletandrews@gmail.com',
#    url='https://github.com/lvandrews',
#    license=license,
#    packages=find_packages(exclude=('tests', 'docs'))
#)

# Program description and version
desctext='make_html.py: Build or update html page for viewing stock data.'
vers='make_html.py v0.1'

# Parse inputs or provide help
import argparse, sys, json, time, os, pprint, codecs
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
outdir = os.path.join(repodir,"docs")

# Initialize parser
parser = argparse.ArgumentParser(description=desctext)
parser.add_argument("-t", "--ticker", help="Ticker abbreviation (e.g. DOW, required)", type=str, metavar="", required=True)
#parser.add_argument("-s", "--source", help="Select data source; default=AlphaVantage", choices=["alphavantage", "yahoo"], type=str, metavar="", default="alphavantage")
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
html_datadir = os.path.join(outdir,"data","")
ticker_html = os.path.join(html_datadir,"",f"{ticker}.html")

# Check that input dir exists -- consider moving as def to util.py
if not os.path.isdir(ticker_datadir):
    print("Source directory does not exist:",ticker_datadir)
    print("Ticker input was:",ticker)
    print("\nExiting\n")
    quit()
else:
    print("Building HTML output from available data.\nTicker input:",ticker,"\n")

# Open/create HTML file in write mode
f = open(ticker_html, 'w')
  
# HTML code
html_out = f"""
<html>
<head>
<title>Title</title>
</head>
<body>
<h2>{ticker}</h2>
<h3>{date_now_notime}</h3>
<p>No data today :(</p>
  
</body>
</html>
"""
  
# Write output and close
f.write(html_out)
f.close()

# viewing html files
# below code creates a 
# codecs.StreamReaderWriter object
#file = codecs.open(html_out, 'r', "utf-8")
  
# using .read method to view the html 
# code from our object
#print(file.read())

