#!/bin/bash
#SBATCH --job-name=eval_satdesc
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_satdesc_%j.out
#SBATCH --error=logs/eval_satdesc_%j.err

cd $PWD/../_v2vgot_repo_tmp
source /home/$USER/miniconda3/etc/profile.d/conda.sh
bash LLaVA/scripts/v1_5/eval/eval_v2vgot_satdesc_full_all.sh
