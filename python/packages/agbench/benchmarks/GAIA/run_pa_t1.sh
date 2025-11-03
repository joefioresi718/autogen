module load apptainer

export APPTAINER_OVERLAY=/home/jo869742/llm/autogen/python/packages/agbench/benchmarks/GAIA/apptainer-overlays/agbench-pa-8G.img
export APPTAINER_SIF=/home/jo869742/llm/agbench.sif
export CONTAINER_RUNTIME=apptainer
export PIP_CACHE_DIR=/workspace/.cache/pip
export XDG_CACHE_HOME=/workspace/.cache
export TMPDIR=/workspace/tmp
export ORT_DISABLE_CPU_AFFINITY=1
export OMP_NUM_THREADS=4           # set to the CPUs you actually want to use
export MKL_NUM_THREADS=4           # match OMP_NUM_THREADS
export OMP_PROC_BIND=false

ssh -N -f -L 11434:localhost:11434 jo869742@evc101
agbench run Tasks/gaia_validation_level_1__ParallelAgents.jsonl
agbench tabulate Results/gaia_validation_level_1__ParallelAgents/
