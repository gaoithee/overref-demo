#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="olmo3-refusal"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --output=slurm_outputs/olmo3_refusal_eval.out
#SBATCH --cpus-per-task=8

echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"

export HF_HOME=/share/ai-lab/scandussio/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

source /u/scandussio/overref-demo/.venv-olmo23/bin/activate

mkdir -p slurm_outputs results/olmo3

python /u/scandussio/overref-demo/run_experiment.py \
    --config config_olmo3 \
    --checkpoints base sft dpo final \
    --datasets wildguard harmbench jailbreakbench beavertails
echo "DONE!"