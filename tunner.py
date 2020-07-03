##tune the stuffffff
import numpy as np
import os

start = 13.1e6
stop = 13.6e6
step = 0.01e6
n_freq = int(stop - start)/step))
freq_range = np.linspace(start,stop, n_freq)

os.system("module load gcc/9.3")

os.system("gcc -O3 -o libalbatrostools.so -fPIC --shared albatrostools.c -fopenmp")


for freq in freq_range:
    with open("scripts/radio"+str(freq/1e6)+".sh",'w') as script:
        script.write("#!/bin/sh\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n#SBATCH --mem=64000\n#SBATCH --time=00:30:00\n")
        script.write("module load gcc/9.3\nmodule load fftw/3.3.8-gcc9\nmodule load openblas/openblas_gcc9\nmodule load libffi/3.3\nmodule load python/3.8.2-gcc9\n")
        script.write("export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/simont/mars\n")
        script.write("python3 /home/simont/mars/radio.py " + str(freq))
    os.system("sbatch /home/imont/mars/scripts/radio"+str(freq/1e6)+".sh")

