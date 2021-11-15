#!/bin/bash

mkdir -p results

grep "Train" out/training.log | grep -o '[0-9]\+' > results/epochs.dat

# Loss
grep -A 8 "Train" out/training.log | grep "Loss:" | awk '{print $2}' > results/train-loss.dat
grep -A 8 "Test" out/training.log | grep "Loss:" | awk '{print $2}' > results/test-loss.dat
paste -d, results/epochs.dat results/*-loss.dat > results/loss.csv

# RMSE
grep -A 8 "Train" out/training.log | grep "RMSE:" | awk '{print $2}' > results/train-rmse.dat
grep -A 8 "Test" out/training.log | grep "RMSE:" | awk '{print $2}' > results/test-rmse.dat
paste -d, results/epochs.dat results/*-rmse.dat > results/rmse.csv

python plot.py
