#!/bin/bash
#SBATCH --job-name=train_vits     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
# /!\ Caution, in the following line, "multithread" refers to hyperthreading.
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=1:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/vits%j.out # output file name
#SBATCH --error=logs/vits%j.out  # error file name
#SBATCH --qos=qos_gpu-dev         # we are submitting a test job

#set -x
#cd ${SLURM_SUBMIT_DIR}

script_name=$1

# Define and create a unique scratch directory for this job
SCRATCH_DIRECTORY=/gpfsscratch/rech/vfw/uur64jb/vits/$SLURM_JOB_NAME
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

cp ${SLURM_SUBMIT_DIR}/script_name ${SCRATCH_DIRECTORY}

echo "running..."
time sleep 5

# module purge
# module load 

# srun python ./mnist_example.py -s&

# wait

# Finish the script
exit 0