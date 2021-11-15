#!/bin/bash

# + ---------------------------------------------------------------------------------- +
# | Training of GNINA scoring function (default2017 model) on the CSAR datast.         |
# |                                                                                    |
# | This is an application test with the goal of reproducing the results from Ragoza   |
# | et al. (J. Chem. Inf. Model. 2017, 57, 942−957), in particular the ROC AUC as a    |
# | function of the number of epochs with and without data augmentation similarly to   |
# | Fig. 3.                                                                            |
# + ---------------------------------------------------------------------------------- +

epochs=250

# Training on CSAR data with data augmentation
for i in 0 1 2
do
    outdir="augmentation-${i}"
    mkdir -p ${outdir}

    nohup \
    python -m gnina.training \
        models-master/data/csar/alltrain${i}.types \
        --testfile models-master/data/csar/alltest${i}.types \
        --data_root models-master/data/csar/ \
        --iterations ${epochs} \
        --batch_size 10 \
        --balanced \
        --test_every 2 \
        --checkpoint_every 1 \
        --num_checkpoints 1 \
        --out_dir ${outdir} \
        --silent &
done

# Training on CSAR data without data augmentation
for i in 0 1 2
do
    outdir="no-augmentation-${i}"
    mkdir -p ${outdir}

    nohup \
    python -m gnina.training \
        models-master/data/csar/alltrain${i}.types \
        --testfile models-master/data/csar/alltest${i}.types \
        --data_root models-master/data/csar/ \
        --iterations ${epochs} \
        --batch_size 10 \
        --balanced \
        --random_translation 0.0 \
        --no_random_rotation \
        --test_every 2 \
        --checkpoint_every 1 \
        --num_checkpoints 1 \
        --out_dir ${outdir} \
        --silent &
done
