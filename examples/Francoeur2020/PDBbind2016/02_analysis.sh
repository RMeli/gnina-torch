#!/bin/bash

mkdir -p results

grep "Train" out/training.log | grep -o '[0-9]\+' > results/epochs.dat

# Loss
grep -A 9 "Train" out/training.log | grep "Loss:" | awk '{print $2}' > results/train-loss.dat
grep -A 9 "Test" out/training.log | grep "Loss:" | awk '{print $2}' > results/test-loss.dat
paste -d, results/epochs.dat results/*-loss.dat > results/loss.csv

# ROC-AUC
grep -A 9 "Train" out/training.log | grep "ROC AUC:" | awk '{print $3}' > results/train-ROC-AUC.dat
grep -A 9 "Test" out/training.log | grep "ROC AUC:" | awk '{print $3}' > results/test-ROC-AUC.dat
paste -d, results/epochs.dat results/*-ROC-AUC.dat > results/ROC-AUC.csv\
