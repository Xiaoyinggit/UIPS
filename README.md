# UIPS
we use old edition KuaiRec dataset, which contain an unused video ID=1225

## Process Dataset
the process files:
- kuai/process/KuaidataProcess.py
- yahoo/yahooPreprocess/yahooProcess.py
- coat/coatProcess/coatProcess.py
- wiki/beta_start/process: process data for beta_start trainning 

- wiki/beta_start/gene_syn_data_prob.py: process data for beta_hat tranining, change the temp parameter($\tau$) in para to get each dataset

- wiki/beta_start/gene_syn_data_mse.py: generate data for MSE, change the temp parameter($\tau$) in OPUN_para can get each dataset

## KuaiRec
### Model
this folder contains different models, including UIPS.
### RUN
./run10Seed.sh $randomseed


## Yahoo
### Model
this folder contains different models, including UIPS.
### RUN
./run10Seed.sh $randomseed


## Coat
### Model
this folder contains different models, including UIPS.
### RUN
./run10Seed.sh $randomseed


## Wiki
### beta_star
- process.py: process dataset to train beta_start
- beta_start_model.py,beta_start_model_v2.py: beta_start model
- gene_syn_data.py, gene_syn_data_mse.py: generate corresponding dataset.
### beta_hat
- beta_hat_model.py: structure of beta_hat model
- train_beta_hat.py: train beta hat
### pi
- model: contains different models, including UIPS.
- train_pi.py: train each model
### RUN
- ./run10Seed($\tau$).sh  randomseed data_path 

