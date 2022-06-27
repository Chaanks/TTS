#!/bin/bash
#SBATCH --array=0-10%1
#SBATCH -A vfw@gpu
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
# /!\ Caution, in the following line, "multithread" refers to hyperthreading.
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/vits.%x.%j.out # output file name
#SBATCH --error=logs/vits.%x.%j.out  # error file name
#SBATCH --qos=qos_gpu-t3         # we are submitting a test job

#set -x

# Define and create a unique scratch directory for this job
SCRATCH_DIRECTORY=/gpfsscratch/rech/vfw/uur64jb/vits/$SLURM_JOB_NAME
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

cp ${SLURM_SUBMIT_DIR}/$1 ${SCRATCH_DIRECTORY}

module purge
module load anaconda-py3/2021.05

conda activate coqui
echo "Make sure pytorch is installed:"
python -c "import torch; print(f'torch version: {torch.__version__}')"

echo "running..."
time sleep 2

srun python $1


# Finish the script
exit 0
