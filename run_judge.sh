#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="olmo-judge"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=60G
#SBATCH --output=slurm_outputs/olmo_judge.out
#SBATCH --cpus-per-task=8

echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"

conda init bash
conda activate overref-olmo23   # transformers>=4.57, works for gpt-oss-20b too

mkdir -p slurm_outputs

# Run judge on all three families.
# Use --resume to safely restart if interrupted — already-judged rows are skipped.
python /u/scandussio/overref-demo/run_judge.py \
    --results-dirs results/olmo1 results/olmo2 results/olmo3 \
    --resume

echo "DONE!"