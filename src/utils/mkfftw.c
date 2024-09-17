#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>
#include <omp.h>
#define MKFFTW_FLAG FFTW_ESTIMATE
// module load python/3.9.10 gcc/11.3.0 fftw/3.3.10
// gcc -std=c99 -O3 -march=native -shared -o libfftw.so -fPIC -fopenmp mkfftw.c -lfftw3_omp -lm
int set_threads(int nthreads)
{
  if (nthreads<0) {
    nthreads=omp_get_num_threads();
  }
  if(fftw_init_threads()){
    fftw_plan_with_nthreads(nthreads);
    // printf("Set FFTW to have %d threads.\n",nthreads);
  }
  else{
    printf("something went wrong during thread init");
  }
}

//gcc-4.9 -I/Users/sievers/local/include -fopenmp -std=c99 -O3 -shared -fPIC -o libmkfftw.so mkfftw.c -L/Users/sievers/local/lib -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3  -lm -lgomp
//gcc-9 -I/usr/local/include -fopenmp -std=c99 -O3 -shared -fPIC -o libmkfftw.so mkfftw.c -L/usr/local/lib -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3  -lm -lgomp


//gcc -I{HIPPO_FFTW_DIR}/include -fopenmp -std=c99 -O3 -shared -fPIC -o libmkfftw.so mkfftw.c -L${HIPPO_FFTW_DIR}/lib    -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3  -lm -lgomp
//gcc -fopenmp -std=c99 -O3 -shared -fPIC -o libmkfftw.so mkfftw.c -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3 -lgomp -lpthread

void many_fft_c2c_1d(fftw_complex *dat, fftw_complex *datft, int nrows, int ncols, int axis, int sign)
{
    int istride=1,ostride=1,idist=ncols,odist=ncols,ndata=ncols,ntrans=nrows;
    long int n = nrows*ncols;
    double nn;
    if(axis==0)
    {
        istride=ncols;
        ostride=ncols;
        idist=1;
        odist=1;
        ndata=nrows;
        ntrans=ncols;
    }
    fftw_plan plan=fftw_plan_many_dft(1, &ndata, ntrans, dat, &ndata, istride, idist, datft, &ndata, ostride, odist, sign, MKFFTW_FLAG);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    if(sign==1) //backward transform
    {
      nn=1.0/ndata;
      #pragma omp parallel for
      for(long int i=0; i < n; i++)
        {
          // printf("datft[%d] is %f + %fi\n", i, creal(datft[i]), cimag(datft[i]));
          datft[i]=datft[i]*nn;
        }
    }
    
}
/*--------------------------------------------------------------------------------*/
void read_wisdom(char *double_file, char *single_file)
{
  printf("files are: .%s. and .%s.\n",double_file,single_file);
  int dd=fftw_import_wisdom_from_filename(double_file);
  //int ss=fftwf_import_wisdom_from_filename(single_file);
  printf("return value is %d\n",dd);
}

/*--------------------------------------------------------------------------------*/
void write_wisdom(char *double_file, char *single_file)
{
  printf("files are: .%s. and .%s.\n",double_file,single_file);
  int dd=fftw_export_wisdom_to_filename(double_file);
  //int ss=fftwf_export_wisdom_to_filename(single_file);
  printf("return value is %d\n",dd);
}

void cleanup_threads()
{
  fftw_cleanup_threads();
}
