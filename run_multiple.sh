#!/bin/bash

# Script to run main.py and automatically commit results when done with a supplied commit message
echo "[Data preparation] Generating vocabulary and clearing test set..."
cd twitter-datasets
./build.sh
cd ..

git checkout  b534615463216a41109f5a3903ee37abdeb0d1a7
echo "[Main run] Running main.py for Single LSTM RNN..."
python3 main.py > log_single_lstm_run
echo "[Main run] ...finished running main.py, changing commit and running again..."
git checkout  5c83e51743304e8b6d06094690e3e38684a92ac3
echo "[Main run] Running main.py for Double Conv..."
python3 main.py > log_double_conv_run
echo "[Main run] ...finished running main.py, done."
git checkout single_lstm
