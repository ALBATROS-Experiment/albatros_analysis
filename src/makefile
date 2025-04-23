CC := gcc
CFLAGS := -fPIC -O3 -std=c99
LIBS := -lfftw3_omp -lm
objects := lib_unpacking.so lib_correlations_cpu.so#Always build unpacking

ifeq (${USE_GPU}, 1)
objects += libpycufft.so
objects += lib_correlations_gpu.so
CFLAGS += -mcpu=native #On MIST
else
objects += libmath.so
objects += libfftw.so
CFLAGS += -march=native #On NIAGARA

endif

all: $(objects)

lib_unpacking.so: correlations/unpacking.c
	$(CC) $(CFLAGS) -shared -fopenmp -o correlations/$@ $^
lib_correlations_cpu.so: correlations/correlations_cpu.c
	$(CC) $(CFLAGS) -shared -fopenmp -o correlations/$@ $^
libmath.so: utils/math_utils.c
	$(CC) $(CFLAGS) -shared -fopenmp -o utils/$@ $^
libfftw.so: utils/mkfftw.c
	$(CC) $(CFLAGS) -shared -fopenmp -o utils/$@ $^ $(LIBS)
libpycufft.so: utils/pycufft.cu
	nvcc -shared -o utils/$@ $^ -Xcompiler -fPIC -lgomp -lcufft
lib_correlations_gpu.so: correlations/correlations_gpu.cu
	nvcc -shared -o correlations/$@ -g -G $^ -Xcompiler -fPIC -lcublas -lcudart
clean:
	rm -f $(wildcard */*.so)
