#!/bin/bash

mkdir -p results

for aug in "augmentation" "no-augmentation"
do
    for i in 0 1 2
    do
        dir="${aug}-${i}"

        grep -A 3 "Train" ${dir}/training.log | grep "ROC AUC" | awk '{print $3}' > ${dir}/test.dat
        grep -A 3 "Test" ${dir}/training.log | grep "ROC AUC" | awk '{print $3}' > ${dir}/train.dat
    done

done

for aug in "augmentation" "no-augmentation"
do
    paste -d, ${aug}-*/train.dat > results/${aug}-train.csv
    paste -d, ${aug}-*/test.dat > results/${aug}-test.csv
done

python plot.py
