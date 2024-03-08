#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>
#define TILE_SIZE 32
void ctrans(complex * x, complex * xT, int nrows, int ncols)
{
    int nblocks = nrows/ncols;
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