#!/bin/bash
#SBATCH --job-name=upload_gc      
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

python datasets/upload_gc_folder.py