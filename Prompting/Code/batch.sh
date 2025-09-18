#!/bin/bash
#SBATCH -c 20
#SBATCH --gres=gpu:2
#SBATCH --output=job-logs/output.txt
#SBATCH --time=4-00:00:00
#SBATCH -w gnode077

mkdir -p /scratch/llm4adr/cache
# conda activate ~/LLM4ADR/research/
# cd ~/LLM4ADR/0_shot/Decision_0_shot

python3 gemma-3-4b-it_TB.py