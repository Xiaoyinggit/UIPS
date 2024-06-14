#!/bin/bash

random_seed=$1

rm -rf summary
mkdir data
mkdir save_path
mkdir summary

do
echo '--------'
echo '--------'
echo $random_seed
echo '--------'
mkdir summary/seed_$random_seed

python3 train_mlp.py  --model_name 'UIPS' --usedata 'yahoo_split'  --train_batch_size '512'  --epochs '10' --lr '0.005'  --UIPS_para 'lambda_-0.5,eta_-0.5,normalize_phi_sa-0.0,cappingFirstEpoch-1.0,cappingThre-0.05,lambdaDiff-0.0,gamma-20.0,eta_2-100' --random_seed $random_seed
echo '--------'
echo '--------'
echo '--------'
echo '--------'
done

