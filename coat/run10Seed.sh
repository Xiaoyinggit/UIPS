#!/bin/bash

random_seed = $1
mkdir data
mkdir save_path
mkdir summary


do
echo '--------'
echo '--------'
echo $random_seed
echo '--------'
mkdir summary/seed_$random_seed
python3 train_mlp.py  --model_name 'UIPS' --usedata 'coat'  --train_batch_size '512'  --epochs '20' --lr '0.005'  --UIPS_para 'lambda_-10.0,eta_-1.0,normalize_phi_sa-0.0,cappingFirstEpoch-1.0,cappingThre-1,lambdaDiff-0.0,gamma-100.0,eta_2-100.0' --random_seed $random_seed
echo '--------'
echo '--------'
echo '--------'
echo '--------'
done
