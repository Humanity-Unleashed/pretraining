#!/bin/bash
#SBATCH --job-name=serve_litellm        
#SBATCH --account=barak_lab
#SBATCH --partition=seas_compute                   
#SBATCH --output /n/netscratch/kempner_sham_lab/Lab/rosieyzh/humun/logs/%A.out           # Output file
#SBATCH --error /n/netscratch/kempner_sham_lab/Lab/rosieyzh/humun/logs/%A.err             # Error file          
#SBATCH --time=72:00:00                    
#SBATCH --nodes=1                             
#SBATCH --ntasks-per-node=1                                  
#SBATCH --cpus-per-task=64               
#SBATCH --mem=50GB

source ~/.bashrc
conda deactivate
conda activate humun

python humun_econ_transformer/data/preprocess_fred_data.py --dataset_path=datasets/all_FRED_merged.csv --metadata_path=datasets/all_fred_metadata.csv --output_folder=datasets --context_window=42 --prediction_window=6 --num_eval_chunks=1