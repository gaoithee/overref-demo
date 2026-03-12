#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="olmo2-refusal"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --output=slurm_outputs/olmo2_refusal_eval.out
#SBATCH --cpus-per-task=8
 
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"
 
conda init bash
conda activate overref-olmo23   # venv with transformers>=4.57
 
mkdir -p slurm_outputs results/olmo2
 
python /u/scandussio/overref-demo/run_experiment.py \
    --config config_olmo2 \
    --checkpoints base sft dpo final \
    --datasets or_bench
 
echo "DONE!"
