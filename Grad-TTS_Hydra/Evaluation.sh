export CUDA_VISIBLE_DEVICES=3

HYDRA_FULL_ERROR=1 python  ground_truth_mel.py -m --config-name=config_eval +data=data model.Masking.a=0.00001
