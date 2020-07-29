#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=1:00:00
#SBATCH --job-name=simon_cool
#SBATCH --output=plot_out.txt
#SBATCH --mail-type=FAIL
 
cd /project/s/sievers/simont/mars_2019_tools

module load gcc
gcc -O3 -o libalbatrostools.so -fPIC --shared albatrostools.c -fopenmp 
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/project/s/sievers/simont/mars_2019_tools 

module load intelpython3


python base_auto.py