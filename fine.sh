#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=64000
#SBATCH --time=06:00:00


module load gcc/9.3
module load fftw/3.3.8-gcc9
module load openblas/openblas_gcc9
module load libffi/3.3
module load python/3.8.2-gcc9

gcc -O3 -o libalbatrostools.so -fPIC --shared albatrostools.c -fopenmp

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/simont/mars

python3 fine_cross.py -o /home/simont/mars/plots -d /data/cynthia/albatros/mars2019/baseband -p 20 -f -c 1563708358 1563711358

