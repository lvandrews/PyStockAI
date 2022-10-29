#!/usr/bin/env python3

# 2022-10-05

#setup(
#    name='basic_model.py',
#    version='0.1.0',
#    description='Basic tf model script for PyStockAI',
#    long_description=readme,
#    author='Lela Andrews',
#    author_email='lelavioletandrews@gmail.com',
#    url='https://github.com/lvandrews',
#    license=license,
#    packages=find_packages(exclude=('tests', 'docs'))
#)

# Program description and version
desctext = 'basic_model.py: Basic model stock analysis based on past performance.'
vers='basic_model.py v0.1'

import time, argparse, os, sys
#from tensorflow import set_random_seed
#set_random_seed(4)
import matplotlib.pyplot as plt
from datetime import datetime

# Define date string, repo directories, util.py file
date_now_notime = time.strftime("%Y-%m-%d")
repodir = os.path.join("..",os.getcwd())
#repodir = dirname(dirname(abspath(__file__)))
datadir = os.path.join(repodir,"data")
modeldir = os.path.join(repodir,"models")
scriptdir = os.path.join(repodir,"scripts")
from util import csv_to_dataset, history_points

# Initialize parser
parser = argparse.ArgumentParser(description=desctext)
parser.add_argument("-i", "--input", help="Input file (required)", type=str, metavar="", required=True)
parser.add_argument("-e", "--epoch", help="ML training epochs", type=int, metavar="", default=20)
parser.add_argument("-d", "--days", help="Days of data to use", type=int, metavar="", default=20)
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

# Parse input
infile = args.input
epoc = args.epoch

# Print input
print("Input:", infile)
print("Epochs:", epoc)
print("")

# Load tensorflow and keras
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
np.random.seed(4)

# dataset

ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(infile)

test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

print(ohlcv_train.shape)
print(ohlcv_test.shape)


# model architecture

lstm_input = Input(shape=(history_points, 5), name='lstm_input')
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(64, name='dense_0')(x)
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('linear', name='linear_output')(x)

model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=epoc, shuffle=True, validation_split=0.1)

# evaluation

y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict(ohlcv_histories)
y_predicted = y_normaliser.inverse_transform(y_predicted)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

model.save(f'basic_model.h5')
