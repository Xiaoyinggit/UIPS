#!/bin/bash

random_seed=$1
usedata=$2

mkdir beta_hat10
mkdir wiki_syndata
mkdir seed_$random_seed
mkdir -p summary/seed_$random_seed

#    ./run10Seed1.sh  1234 wiki_syndata/syn_Wiki_100_with_prob1.pkl

echo '--------'
echo '--------'
echo $random_seed
echo '--------'
mkdir save_path
python3 pi/train_pi.py  --model_name 'UIPS' --train_batch_size '512'  --epochs '20' --lr '0.00001'  --UIPS_para 'lambda_-50.0,eta_-1,normalize_phi_sa-0,gamma-10.0,eta_2-10000'  --random_seed $random_seed --optimal_beta_model_path 'beta_hat10/lr_0.00001_size_100/save_path/ckptTop20' --usedata $usedata
mv save_path seed_$random_seed/UIPS
echo '--------'
echo '--------'
echo '--------'
echo '--------'


