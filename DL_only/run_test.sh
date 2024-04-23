#!/bin/bash

#py=/home/ktoride/.conda/envs/torch/bin/python
py=/home/ktoride/run_python.sh

leads=(\
ml_old_unet4-256_month01_lead3_lr1.5e-05 \
ml_old_unet4-256_month01_lead6_lr1.2e-05 \
ml_old_unet4-256_month01_lead9_lr8.0e-06 \
#ml_old_unet4-256_month01_lead12_lr8.0e-06 \
)

for base in "${leads[@]}"
do
    for i in {1..5}
    do
        exp=${base}_$i
        #$py 2-1_test.py $exp
        $py 2-1_test_old.py $exp
        # $py 3-1_test_real.py $exp
    done
done
