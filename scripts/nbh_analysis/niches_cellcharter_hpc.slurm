#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1                          # List of available GPUs: https://wiki.bwhpc.de/e/Helix/Hardware#Compute_Nodes
#SBATCH --time=48:00:00
#SBATCH --mem=120gb
#SBATCH --job-name=cellcharter                # Specify name for your job
#SBATCH --mail-user=fz305@uni-heidelberg.de   # Specify email address for job notifications
#SBATCH --mail-type=END,FAIL,REQUEUE          

DIM="3Dnbrs" # "2D" | "3Dfull" | "3Dnbrs"

## Define paths
cd $(ws_find thesis)/master_thesis/                                     # Paths specifications can be absolute or relative to this directory
script=$PWD/scripts/nbh_analysis/cellcharter_hpc.py                     # Specify path to python script

## Environment setup
module load devel/cuda/12.1                                                   # cuda version used to install pytorch
export OMP_NUM_THREADS=${SLURM_NTASKS}
source /gpfs/bwfor/home/hd/hd_hd/hd_fz305/.py_venvs/cellcharter/bin/activate  # Activate python venv

## Run
if [[ $DIM == "3Dfull" ]]; then
    python $script -i $PWD/data/input/cellcharter_3D.h5ad -o $output $PWD/data/output/cellcharter_model_3D/
elif [[ $DIM == "3Dnbrs" ]]; then
    for i in {10..140..10}
    do
        echo "Processing file $i"
        python $script -i $PWD/data/input/3Dnbrs/cellcharter_s$i.h5ad -o $output $PWD/data/output/3Dnbrs/cellcharter_model_s$i/
    done
else
    for i in {0..151}
    do 
        echo "Processing file $i"
        python $script -i $PWD/data/input/sections/cellcharter_z$i.h5ad -o $output $PWD/data/output/sections/cellcharter_model_z$i/
    done
fi