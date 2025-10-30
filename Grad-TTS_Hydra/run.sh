#!/bin/bash
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export MKL_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

export CUDA_VISIBLE_DEVICES=3

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate CUDA122_Grad-TTS  

HYDRA_FULL_ERROR=1 python train.py -m --config-name=config +data=data model.Masking.a=0.05  model.Masking.b=20  training.n_timesteps=10 training.load_decoder=True training.load_encoder=True training.train_encoder=False


