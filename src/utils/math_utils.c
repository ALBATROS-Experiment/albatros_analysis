#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>

// gcc -std=c99 -O3 -march=native -shared -o libmath.so -fPIC -fopenmp math_utils.c
void ctrans(complex * x, complex * xT, int nrows, int ncols)
{
    int nblocks = nrows/ncols;
    // printf("size of the temp arrays going to be %d\n", ncols*ncols*sizeof(complex));
    #pragma omp parallel
    {
        complex * temp1 = malloc(ncols*ncols*sizeof(complex));
        complex * temp2 = malloc(ncols*ncols*sizeof(complex));
        int ii=0;
        #pragma omp for nowait
        for(int ib=0;ib<nblocks;ib++){
            for(int a = 0;a<ncols;a++){
                ii=ib*ncols+a;
                for(int b = 0;b<ncols;b++){
                    temp1[a*ncols+b]=x[ii*ncols+b];
                }
            }
            //transpose sub-block
            for(int a = 0;a<ncols;a++){
                for(int b = 0;b<ncols;b++){
                    temp2[b*ncols+a]=temp1[a*ncols+b];
                }
            }
            //write sub-block
            for(int a = 0;a<ncols;a++){
                for(int b = 0;b<ncols;b++){
                    ii=ib*ncols+b;
                    xT[a*nrows+ii]=temp2[a*ncols+b];
                }
            }

        }
        free(temp1);
        free(temp2);
    }
    
}

void ctrans_zero(complex * x, complex * xT, int nrows, int ncols)
{   
    int nblocks = nrows/ncols;
    // int nrows2 = 2*nrows;
    long int nn = nrows*ncols*2; // xT has zeros at the end
    // printf("size of the temp arrays going to be %d\n", ncols*ncols*sizeof(complex));
    #pragma omp parallel for
    for(long int i=0; i<nn; i++){
        xT[i]=0;
    }
    #pragma omp parallel
    {
        complex * temp1 = malloc(ncols*ncols*sizeof(complex));
        complex * temp2 = malloc(ncols*ncols*sizeof(complex));
        int ii=0;
        #pragma omp for nowait
        for(int ib=0;ib<nblocks;ib++){
            for(int a = 0;a<ncols;a++){
                ii=ib*ncols+a;
                for(int b = 0;b<ncols;b++){
                    temp1[a*ncols+b]=x[ii*ncols+b];
                }
            }
            //transpose sub-block
            for(int a = 0;a<ncols;a++){
                for(int b = 0;b<ncols;b++){
                    temp2[b*ncols+a]=temp1[a*ncols+b];
                }
            }
            //write sub-block
            for(int a = 0;a<ncols;a++){
                for(int b = 0;b<ncols;b++){
                    ii=ib*ncols+b;
                    xT[2*a*nrows+ii]=temp2[a*ncols+b];
                }
            }

        }
        free(temp1);
        free(temp2);
    }
    // #pragma omp parallel for
    // for(int i=0; i<ncols; i++){
    //     for(int j=nrows;j<2*nrows;j++)
    //     {
    //         xT[2*i*nrows+j]=0;
    //     }
        
    // }
}