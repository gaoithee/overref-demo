#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="olmo1-refusal"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --output=slurm_outputs/olmo1_refusal_eval.out
#SBATCH --cpus-per-task=8

echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"

source /u/scandussio/overref-demo/.venv/bin/activate

mkdir -p slurm_outputs results/olmo1

python /u/scandussio/overref-demo/run_experiment.py \
    --config config \
    --checkpoints base sft dpo final \
    --datasets or_bench false_reject wildguard harmbench jailbreakbench toxicchat beavertails

echo "DONE!"