#!/bin/bash

mkdir -p results

for aug in "augmentation" "no-augmentation"
do
    for i in 0 1 2
    do
        dir="${aug}-${i}"

        grep "Train" ${dir}/training.log | grep -o '[0-9]\+' > results/epochs.dat

        grep -A 5 "Train" ${dir}/training.log | grep "ROC AUC" | awk '{print $3}' > ${dir}/train.dat
        grep -A 5 "Test" ${dir}/training.log | grep "ROC AUC" | awk '{print $3}' > ${dir}/test.dat
    done
done

for aug in "augmentation" "no-augmentation"
do
    paste -d, results/epochs.dat ${aug}-*/train.dat > results/${aug}-train.csv
    paste -d, results/epochs.dat ${aug}-*/test.dat > results/${aug}-test.csv
done

python plot.py
