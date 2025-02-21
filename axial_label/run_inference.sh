#!/bin/bash --login
#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=8
#SBATCH --mem-per-cpu=304800M   # memory per CPU core
#SBATCH -J "gemma_axial"   # job name
#SBATCH -e ./errors/%j_err.txt
#SBATCH -o ./outputs/%j_output.txt
#SBATCH --qos=cs

# nvidia-smi
torchrun --nproc_per_node 8 label_posts.py \
    --tokenizer_path model_save \
    --max_seq_len 4960 --max_batch_size 16 \
    --system_prompt_file axial_prompt.txt \
    --input_csv_path all_codes.csv
