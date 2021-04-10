#!/usr/bin/env python3
#
# Script modified from tutorial here: https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras
# 2021-02-26

#setup(
#    name='predict-train-test',
#    version='0.1.0',
#    description='Predict/train/test script for PyStockAI',
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

today=dt.date.today()
year_ago=today - dt.timedelta(days=365)

# Program description
desctext = 'predict-train-test.py: use Tensorflow2 to model stock performance and predict future performance.'

# Initialize parser
parser = argparse.ArgumentParser(description=desctext)

parser.add_argument("-t", "--ticker", help="Ticker abbreviation (e.g. AMZN, required)", type=str, metavar='', required=True)
parser.add_argument("-e", "--epoch", help="Epochs to train (integer, default = 5)", type=int, metavar='', default="5")
parser.add_argument("-s", "--test_size", help="Test ratio size, 0.2 is 20%% (decimal, default = 0.2)", type=float, metavar='', default="0.2")
parser.add_argument("-w", "--window_size", help="Window length used to predict (integer, default = 50)", type=int, metavar='', default="50")
parser.add_argument("-l", "--lookup_step", help="Lookup step, 1 is the next day (integer, default = 1)", type=int, metavar='', default="1")
parser.add_argument("-b", "--begin_date", help="Beginning date for analysis set (e.g. 2021-04-20, default = one year ago from present date)", type=str, metavar='', default=year_ago)
parser.add_argument("-a", "--all_time", help="Use all available data (supercedes -b)", action="store_true")
parser.add_argument("-k", "--keep_results", help="Keep output dataframe (saves as .csv to results directory)", action="store_true")

parser.add_argument("-v", "--version", help="show program version", action="version", version="%(prog)s 0.1")
parser.add_argument("-V", "--verbose", help="increase output verbosity", action="store_true")


# Print help if no arguments supplied
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

# Read arguments from the command line and provide useful feedback
args = parser.parse_args()
if args.verbose:
    print("Verbosity turned on")
    
parser.parse_args()

# Set ticker symbol to uppercase if lowercase was entered
ticker = args.ticker.upper()

# Print important model parameters
print("Ticker symbol:", ticker)
print("Epochs to train:", args.epoch)
print("Test size:", args.test_size)
print("Window size:", args.window_size)
print("Lookup step:", args.lookup_step)
print("RNN cell: LSTM")

########################
### PREDICTING THE MODEL
# Import libraries and silence annoying tensorflow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import numpy as np
import pandas as pd
import random
import shutil
import time
import matplotlib.pyplot as plt
import alpaca_trade_api as ata
from pytz import timezone
import seaborn as sns
sns.set()

# Create these folders if they does not exist (including ticker-specific subdirectories within results, data and logs)
cwd = os.getcwd()
outdir_results = os.path.join(cwd, "results", ticker)
outdir_logs = os.path.join(cwd, "logs", ticker)
outdir_data = os.path.join(cwd, "data", ticker)
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir(outdir_results):
    os.mkdir(outdir_results)
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir(outdir_logs):
    os.mkdir(outdir_logs)
if not os.path.isdir("data"):
    os.mkdir("data")
if not os.path.isdir(outdir_data):
    os.mkdir(outdir_data)

# Date strings
date_now = time.strftime("%Y-%m-%d_%H-%M-%S")
date_now_notime = time.strftime("%Y-%m-%d")
dt_string = time.strftime("%b-%d-%Y %I:%M:%S %p")

# Set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

# Preparing the dataset
def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

# Clear output directories
for files in os.listdir(outdir_results):
  path = os.path.join(outdir_results, files)
  try:
    shutil.rmtree(path)
  except OSError:
    os.remove(path)
for files in os.listdir(outdir_logs):
  path = os.path.join(outdir_logs, files)
  try:
    shutil.rmtree(path)
  except OSError:
    os.remove(path) 
for files in os.listdir(outdir_data):
  path = os.path.join(outdir_data, files)
  try:
    shutil.rmtree(path)
  except OSError:
    os.remove(path) 

# Ticker data filename (yahoo finance)
ticker_data_filename = os.path.join(outdir_data, f"{ticker}_{date_now_notime}.csv")

# Need to figure out how "ticker" is operating in below code
def load_data(ticker, window_size=args.window_size, scale=True, shuffle=True, lookup_step=args.lookup_step, split_by_date=True,
                test_size=args.test_size, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        window_size (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
            to False will split datasets in a random way
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """

    # Load data from Yahoo Finance using specified range or all time
    if args.all_time:
      df = si.get_data(ticker)

    elif args.begin_date:
      df = si.get_data(ticker, start_date=args.begin_date, end_date=today)
  
    ### THIS PART ONLY FOR RUNNING ON COMMAND LINE
    # see if ticker is already a loaded stock from yahoo finance
    # if isinstance(ticker, str):
        # load it from yahoo_fin library
        # print("Loading new data")
        # df = si.get_data(ticker, start_date=args.begin_date, end_date=today)
    # elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        # print("Data already loaded")
        # df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    # Add date as a new column to end of df (first column also contains date as Unnamed column 0)
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=window_size)
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == window_size:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `window_size` sequence with `lookup_step` sequence
    # for instance, if window_size=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    return result

# Model creation
def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

######################
### TRAINING THE MODEL

# Window size or the sequence length
WINDOW_SIZE = args.window_size

# Lookup step, 1 is the next day
LOOKUP_STEP = args.lookup_step

# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"

# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"

# whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

# test ratio size, 0.2 is 20%
TEST_SIZE = args.test_size

# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

### model parameters
N_LAYERS = 2

# LSTM cell
CELL = LSTM

# 256 LSTM neurons
UNITS = 256

# 40% dropout
DROPOUT = 0.4

# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters
# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = args.epoch

# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now_notime}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{WINDOW_SIZE}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"

# Redirect data output
ticker_data_filename = os.path.join(outdir_data, f"{ticker}_{date_now_notime}.csv")

# load the data
data = load_data(ticker, WINDOW_SIZE, scale=SCALE, split_by_date=SPLIT_BY_DATE, 
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                feature_columns=FEATURE_COLUMNS)
                
# save the dataframe
data["df"].to_csv(ticker_data_filename)

# construct the model
model = create_model(WINDOW_SIZE, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
                    
# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join(outdir_results, model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=0)
tensorboard = TensorBoard(log_dir=os.path.join(outdir_logs, model_name))

# train the model and save the weights whenever we see 
# a new optimal model using ModelCheckpoint
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)

# Would love to have an automated tensorboard command that generates useful plots without having to open browser    
#tensorboard --logdir="outdir_logs"

#####################    
### TESTING THE MODEL

def plot_graph(test_df):
    """
    This function plots true close price (red) true future price (green),
    predicted close price (blue), and true high/low (yellow)
    """
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='g')
    plt.plot(test_df[f'adjclose'], c='r')
    plt.plot(test_df[f'high'], c='y')
    plt.plot(test_df[f'low'], c='y')
    plt.fill_between(test_df.index, test_df[f'high'], test_df[f'low'], alpha=0.1)
    plt.title(ticker)
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend(["Predicted Future Price", f"True Future Price ({LOOKUP_STEP} Days)", "Actual Close Price", "Actual High/Low"], loc='upper left')
    plt.show()
    
def get_final_df(model, data):
    """
    This function takes the `model` and `data` dict to 
    construct a final dataframe that includes the features along 
    with true and predicted prices of the testing dataset
    """
    # if predicted future price is higher than the current, 
    # then calculate the true future price minus the current price, to get the buy profit
    buy_profit  = lambda current, true_future, pred_future: true_future - current if pred_future > current else 0
    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, true_future, pred_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit, 
                                    final_df["adjclose"], 
                                    final_df[f"adjclose_{LOOKUP_STEP}"], 
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit, 
                                    final_df["adjclose"], 
                                    final_df[f"adjclose_{LOOKUP_STEP}"], 
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    return final_df
    
def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-WINDOW_SIZE:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price
    
# load optimal model weights from results folder
model_path = os.path.join(outdir_results, model_name) + ".h5"
model.load_weights(model_path)

# evaluate the model
loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
if SCALE:
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae
    
# Define final_df    
final_df = get_final_df(model, data)

# predict the future price
future_price = predict(model, data)

# we calculate the accuracy by counting the number of positive profits
accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
# calculating total buy & sell profit
total_buy_profit  = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()
# total profit by adding sell & buy together
total_profit = total_buy_profit + total_sell_profit
# dividing total profit by number of testing samples (number of trades)
profit_per_trade = total_profit / len(final_df)

# printing metrics
print(f"Future price after {LOOKUP_STEP} days is ${future_price:.2f}")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error: $",mean_absolute_error)
print("Accuracy score:", accuracy_score)
print("Total buy profit: $",total_buy_profit)
print("Total sell profit: $",total_sell_profit)
print("Total profit: $",total_profit)
print("Profit per trade: $",profit_per_trade)

# Get the final dataframe for the testing set, save if -k is passed
keep_csv = os.path.join(outdir_results, f"{date_now_notime}_{ticker}_e-{EPOCHS}_s-{TEST_SIZE}_w-{WINDOW_SIZE}_l-{LOOKUP_STEP}_final_df.csv")
if args.keep_results:
  final_df.to_csv(keep_csv)

# plot true/pred prices graph -- want to save as .png and have local html hosting monitored stocks
plot_graph(final_df)

#ax = sns.lineplot(final_df)
#ax.legend(loc='upper left')
#plt.show()

