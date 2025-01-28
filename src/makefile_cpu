CC := gcc
CFLAGS := -fPIC -O3 -std=c99 -march=native
LIBS := -lfftw3_omp -lm
objects := lib_unpacking.so lib_correlations_cpu.so libmath.so libfftw.so
all: $(objects)

lib_unpacking.so: correlations/unpacking.c
	$(CC) $(CFLAGS) -shared -fopenmp -o correlations/$@ $^
lib_correlations_cpu.so: correlations/correlations_cpu.c
	$(CC) $(CFLAGS) -shared -fopenmp -o correlations/$@ $^
libmath.so: utils/math_utils.c
	$(CC) $(CFLAGS) -shared -fopenmp -o utils/$@ $^
libfftw.so: utils/mkfftw.c
	$(CC) $(CFLAGS) -shared -fopenmp -o utils/$@ $^ $(LIBS)

clean:
	rm -f $(wildcard */*.so)


