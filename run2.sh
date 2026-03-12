#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="olmo2"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --output=slurm_outputs/olmo_false_reject_eval.out
#SBATCH --cpus-per-task=8

# Standard preamble for debugging
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"

conda init bash
conda activate overref-demo

mkdir -p slurm_outputs

python run_experiment.py \
    --checkpoints base sft dpo final \
    --datasets or_bench false_reject

echo "DONE!"
