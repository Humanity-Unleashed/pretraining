#!/bin/bash
#SBATCH --job-name=upload_gc      
#SBATCH --account=<YOUR ACCOUNT HERE>
#SBATCH --partition=<YOUR PARTITION HERE>               
#SBATCH --output <YOUR OUTPUT FILE PATH HERE>           # Output file
#SBATCH --error <YOUR ERROR FILE PATH HERE>            # Error file          
#SBATCH --time=72:00:00                    
#SBATCH --nodes=1                             
#SBATCH --ntasks-per-node=1                                  
#SBATCH --cpus-per-task=64               
#SBATCH --mem=50GB

source ~/.bashrc
conda deactivate
conda activate humun

python datasets/upload_gc_folder.py