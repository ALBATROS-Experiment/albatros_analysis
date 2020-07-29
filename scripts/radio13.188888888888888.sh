#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64000
#SBATCH --time=00:30:00
module load gcc/9.3
module load fftw/3.3.8-gcc9
module load openblas/openblas_gcc9
module load libffi/3.3
module load python/3.8.2-gcc9
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/simont/mars
python3 /home/simont/mars/radio.py 13188888.888888888