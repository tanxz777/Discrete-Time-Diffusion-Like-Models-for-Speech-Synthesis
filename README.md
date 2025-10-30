# Discrete-Time-Diffusion-like-Models-for-Speech-Synthesis

This is the code for Paper: Discrete-Time Diffusion-Like Models for Speech Synthesis. <br>
This repository is based on original [Grad-TTS implementation by Huawei Noah’s Ark Lab](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS). Compared with the original code, main changes lies in model/diffusion.py <br>
Hydra is used for managing Hyperparameters.<br>

## usage
*Inference* for four different systems:Gaussian Additive Noise, Gaussian Multiplicative Noise, Blurring Noise, Blurring+Gaussian Additive Noise can be done by **1)** renaming diffusion_GAM.py, diffusion_GMM.py, diffusion_blur.py, diffusion_warm.py to diffusion.py in 'Grad-TTS_Hydra/model' folder **2)** choose correponding checkpoint repository name in 'eval.checkpoint_dir' field in Grad-TTS_Hydra/config/config_eval.yaml file  **3)** run ./eval_all.sh <br>
*Training* can be done by both replacing the corresponding diffusion_*.py file and changing relevant hyperparameters in 'Grad-TTS_Hydra/config/config.yaml' file.

