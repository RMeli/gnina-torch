#!/bin/bash

mkdir -p results

for i in 1 2 3
do
    out="out${i}"

    grep "Train" ${out}/training.log | grep -o '[0-9]\+' > results/epochs.dat

    # Loss
    grep -A 7 "Train" ${out}/training.log | grep "Loss:" | awk '{print $2}' > ${out}/train-loss.dat
    grep -A 7 "Test" ${out}/training.log | grep "Loss:" | awk '{print $2}' > ${out}/test-loss.dat

    # RMSE
    grep -A 7 "Train" ${out}/training.log | grep "RMSE:" | awk '{print $2}' > ${out}/train-rmse.dat
    grep -A 7 "Test" ${out}/training.log | grep "RMSE:" | awk '{print $2}' > ${out}/test-rmse.dat
done

for m in "loss" "rmse"
do
    paste -d, results/epochs.dat out*/train-${m}.dat > results/train-${m}.csv
    paste -d, results/epochs.dat out*/test-${m}.dat > results/test-${m}.csv
done

#python plot.py
