#!/bin/bash

random_seed=$1



mkdir data
mkdir save_path
mkdir summary
mkdir summary/seed_$random_seed


echo '--------'
echo '--------'
echo $random_seed
echo '--------'
mkdir save_path
python3 train_mlp.py  --model_name 'UIPS' --usedata 'Kuai'  --train_batch_size '512'  --epochs '20' --lr '0.000001'  --UIPS_para 'lambda_-50.0,eta_-1.0,normalize_phi_sa-0,cappingFirstEpoch-1.0,cappingThre-100,lambdaDiff-0.65,gamma-100.0,eta_2-100' --random_seed $random_seed
mv save_path seed_$random_seed/UIPS
echo '--------'
echo '--------'
echo '--------'
echo '--------'
