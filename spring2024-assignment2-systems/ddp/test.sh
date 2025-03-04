#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2
#SBATCH --mem=8G
#SBATCH --time=00:02:00

# Activate conda environment


eval "$(conda shell.bash hook)"
# Change conda environment name, if necessary
conda activate llmscratch
# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-4}  # Default to 4 if not set
echo "MASTER_PORT: ${MASTER_PORT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"

# Execute command for each task
srun python multinode.py