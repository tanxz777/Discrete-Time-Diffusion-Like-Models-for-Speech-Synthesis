source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate Grad-TTS-EVAL

job_name='GPM_Hydra'    #Hydra job name
task_prefix='model.Masking.d:'       #Hyper parameter name 

echo "$(date)"

#for value in 0.1 0.2 0.3; do   #name is the Hyper Paremeter name
#    ./tools/MCD/MCD.sh "$job_name" "model.Masking.a:0.05__model.Masking.b:0.01__model.Masking.c:40__model.Masking.d:${value}"
#done

#for value in 10 20 30 40; do   #name is the Hyper Paremeter name
#    ./tools/MCD/MCD.sh "$job_name" "model.Masking.a:0.04__model.Masking.b:0.01__model.Masking.c:${value}__model.Masking.d:0.2"
#done

#./tools/MCD/MCD.sh "$job_name" 'model.Masking.a:0.04__model.Masking.b:0.01__model.Masking.c:0__model.Masking.d:0.0__training.learning_rate:5e-05__training.load_decoder:False__training.load_encoder:False__training.n_epochs:800__training.train_encoder:True'

./tools/MCD/MCD.sh "$job_name" 'model.Masking.a:0.04__model.Masking.b:0.01__model.Masking.c:0__model.Masking.d:0.0__training.learning_rate:5e-05__training.load_decoder:True__training.load_encoder:True__training.n_epochs:2400__training.start_checkpoint:1595__training.train_encoder:True'

echo "$(date)"

