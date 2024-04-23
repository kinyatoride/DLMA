#!/bin/bash
# 1.0e-6 1.2e-6 1.5e-6 2.0e-6 2.5e-6 3.0e-6 4.0e-6 5.0e-6 6.0e-6 8.0e-6
# 1.0e-5 1.2e-5 1.5e-5 2.0e-5 2.5e-5 3.0e-5 4.0e-5 5.0e-5 6.0e-5 8.0e-5
# 1.0e-4 1.2e-4 1.5e-4 2.0e-4 2.5e-4 3.0e-4 4.0e-4 5.0e-4 6.0e-4 8.0e-4


py=/home/ktoride/run_python.sh
#lead=12
#lr=1.0e-5
#$py 1_train_ensembles.py --lead $lead -lr $lr --i-ens 2 --n-ens 1

#--- Lead time ---

#lead=3
#for lr in 1.5e-5 
#do
#   $py 1_train_ensembles.py --lead $lead -lr $lr --i-ens 3 --n-ens 7
#done

#lead=6
#for lr in 1.2e-5
#do
#    $py 1_train_ensembles.py --lead $lead -lr $lr --i-ens 3 --n-ens 7
#done

#lead=9
#for lr in 8.0e-6
#do
#    $py 1_train_ensembles.py --lead $lead -lr $lr --i-ens 3 --n-ens 7
#done

lead=12
for lr in 6.0e-6 8.0e-6 1.0e-5
do
   $py 1_train_ensembles.py --lead $lead -lr $lr --i-ens 0 --n-ens 3
done

