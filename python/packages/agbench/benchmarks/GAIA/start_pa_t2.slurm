#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 5
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --job-name=run_gaia
#SBATCH --output=log_pa_t2.out

echo $SLURM_JOB_NODELIST

# source /lustre/fs1/home/jfioresi/miniforge3/envs/autogen/bin/activate
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/lustre/fs1/home/jfioresi/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/lustre/fs1/home/jfioresi/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/lustre/fs1/home/jfioresi/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/lustre/fs1/home/jfioresi/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/lustre/fs1/home/jfioresi/miniforge3/etc/profile.d/mamba.sh" ]; then
    . "/lustre/fs1/home/jfioresi/miniforge3/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<
mamba activate autogen

### Running the program ###
srun bash run_pa_t2.sh
