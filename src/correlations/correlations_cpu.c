#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

void autocorr_4bit(uint8_t * data, uint8_t * corr, uint32_t nspec, uint32_t ncol)
{
    /*
        This is for fine control over what rows to correlate for some given data.
        Data is assumed to consist of only one pol (already sortpol'd).

        data: nspec * ncol array. ncol = nchan only for 4 bit

        corr: nspec * nchan array. Stores autocorr for each spec and each channel. 
        
        In 4 bit case input and output dimensions are same.
        However, for 1 bit, you'll be filling 4 channels for each input data point.
    */
    
    uint64_t nn = nspec * ncol;
    uint8_t imask=15;
      uint8_t rmask=255-15;

    #pragma omp parallel for default(none) firstprivate(imask,rmask,nn) shared(data,corr)
    for(int i = 0; i<nn; i++)
    {
        int8_t im=data[i]&imask;
        int8_t r=(data[i]&rmask)>>4;
        if (r > 8){r = r - 16;}
        if (im > 8){im = im - 16;}
        //printf("data %d, r %d, im %d\n", data[i], r, im);
        //printf("r %d, im %d", r, im);
        corr[i] = r*r + im*im;
        //printf("corr[i]=%d\n", corr[i]);
    }

}

void avg_autocorr_4bit(uint8_t * data, int64_t * corr, int nrows, int ncol)
{
    /*
        Returns an array of nchan elements. Sum over all spectra for each channel. 
        Division by appropriate spectra count will be taken care by python frontend.
        In 4 bit case start and stop idx correspond directly to spec_num
    */

    for(int i=0;i<ncol;i++)
    {
        corr[i]=0;
    }
    
    uint8_t imask=15;
      uint8_t rmask=255-15;
    // printf("\nfrom C autocorr: nrows %d, ncols %d\n", nrows, ncol);
    int64_t sum_pvt[ncol];
    
    // printf("\n First few data elements %d %d %d %d %d", data[0], data[1], data[2], data[64], data[65]);
    #pragma omp parallel private(sum_pvt)
    {
        for(int i =0;i<ncol;i++)
        {
            sum_pvt[i] = 0 ;
        }

        #pragma omp for nowait
        for(int i = 0; i<nrows; i++) // be careful. for loop over int start_idx is uint32. should be ok
        {
            for(int j=0; j<ncol; j++)
            {	
                
                int8_t im=data[i*ncol+j]&imask;
                int8_t r=(data[i*ncol+j]&rmask)>>4;
                if (r > 8){r = r - 16;}
                if (im > 8){im = im - 16;}
                // printf("int8r %d, r*r = %d\n", (int8_t)r, prod);
                sum_pvt[j] = sum_pvt[j] + r*r + im*im;
            }
        }
        #pragma omp critical
        {
            for(int i =0; i<ncol; i++)
            {
                corr[i] = corr[i] + sum_pvt[i];
            }
        }

    }
    // printf("\nValue of corr being returned from C\n");
    // for(int i=0;i<ncol;i++)
    // {
    // 	printf("%d ",corr[i]);
    // }
}

void avg_autocorr_4bit_raw()
{
    // something that can average whole files, both pols from raw data without any unpacking/sortpol
}

void xcorr_4bit(uint8_t * data0, uint8_t * data1, float * xcorr, int nrows, int ncol)
{

    int nn = nrows * ncol;
    uint8_t imask=15;
      uint8_t rmask=255-15;

    #pragma omp parallel for
    for(int i = 0; i<nn; i++)
    {
        int8_t im0=data0[i]&imask;
        int8_t r0=(data0[i]&rmask)>>4;
        if (r0 > 8){r0 = r0 - 16;}
        if (im0 > 8){im0 = im0 - 16;}

        int8_t im1=data1[i]&imask;
        int8_t r1=(data1[i]&rmask)>>4;
        if (r1 > 8){r1 = r1 - 16;}
        if (im1 > 8){im1 = im1 - 16;}

        //printf("data %d, r %d, im %d\n", data[i], r, im);
        //printf("r %d, im %d", r, im);
        xcorr[2*i] = r0*r1 + im0*im1;
        xcorr[2*i+1] = r1*im0 - r0*im1;
        //printf("corr[i]=%d\n", corr[i]);
    }
}

void avg_xcorr_4bit(uint8_t * data0, uint8_t * data1, float * xcorr, int nrows, int ncol)
{
    /*
        Returns an array of nchan elements. Sum over all spectra for each channel. 
        Division by appropriate spectra count will be taken care by python frontend.
    */

    uint8_t imask=15;
      uint8_t rmask=255-15;
    //+2.1bil to -2.4bil, should be enough, and compatible with float32
    int32_t sum_r_pvt[ncol], sum_im_pvt[ncol];

    for(int i=0; i<ncol; i++)
    {
        xcorr[2*i]=0;
        xcorr[2*i+1]=0;
    }

    #pragma omp parallel private(sum_r_pvt,sum_im_pvt)
    {
        //init
        for(int i=0;i<ncol;i++)
        {
            sum_r_pvt[i]=0;
            sum_im_pvt[i]=0;
        }

        #pragma omp for nowait
        for(int i=0; i<nrows; i++)
        {
            for(int j=0; j<ncol; j++)
            {
                int8_t im0=data0[i*ncol+j]&imask;
                int8_t r0=(data0[i*ncol+j]&rmask)>>4;
                if (r0 > 8){r0 = r0 - 16;}
                if (im0 > 8){im0 = im0 - 16;}
                

                int8_t im1=data1[i*ncol+j]&imask;
                int8_t r1=(data1[i*ncol+j]&rmask)>>4;
                if (r1 > 8){r1 = r1 - 16;}
                if (im1 > 8){im1 = im1 - 16;}
                // printf("%d J%d ... %d J%d\n",r0,im0, r1,im1);

                sum_r_pvt[j] = sum_r_pvt[j] + r0*r1 + im0*im1;
                sum_im_pvt[j] = sum_im_pvt[j] + r1*im0 - r0*im1;
            }
        }
        #pragma omp critical
        {
            for(int k=0; k<ncol; k++)
            {
                // printf("setting real xcorr of k=%d as %d\n",k, sum_r_pvt[k]);
                xcorr[2*k] = xcorr[2*k] + sum_r_pvt[k];
                xcorr[2*k+1] = xcorr[2*k+1] + sum_im_pvt[k];
            }
        }
    }
}

int avg_xcorr_4bit_2ant(uint8_t * data0, uint8_t * data1, float * xcorr, int64_t* specnum0, int64_t* specnum1, int64_t idxstart0, int64_t idxstart1, int nrows0, int nrows1, int ncol)
{
    int rownums0[nrows0], rownums1[nrows1], row_count=0, i=0,j=0;
    uint8_t imask=15;
    uint8_t rmask=255-15;
    //+2.1bil to -2.4bil, should be enough, and compatible with float32
    int32_t sum_r_pvt[ncol], sum_im_pvt[ncol];

    // printf("\n***Variables passed****\n");
    // printf("idx: %d %d %d %d\n", idxstart0, idxstart1, nrows0, nrows1);
    // printf("From C: %d %d",specnum0[0],specnum1[0]);
    // printf("ncol: %d\n", ncol);

    while((i<nrows0)&&(j<nrows1))
    {
        
        if((specnum0[i]-idxstart0)==(specnum1[j]-idxstart1))
        {
            rownums0[row_count]=i;
            rownums1[row_count]=j;
            row_count=row_count+1;
            i=i+1;
            j=j+1;
        }
        else if((specnum0[i]-idxstart0)>(specnum1[j]-idxstart1)) {j=j+1;}
        else {i=i+1;}
    }
    // printf("FROM C: rownums selected are\n\n");
    // for(int i =0;i<row_count;i++)
    // {
    // 	printf("%d    %d\n", rownums0[i], rownums1[i]);
    // }
    // printf("row count is: %d\n", row_count);

    for(int i=0; i<ncol; i++)
    {
        xcorr[2*i]=0;
        xcorr[2*i+1]=0;
    }

    #pragma omp parallel private(sum_r_pvt,sum_im_pvt)
    {
        //init
        for(int i=0;i<ncol;i++)
        {
            sum_r_pvt[i]=0;
            sum_im_pvt[i]=0;
        }

        #pragma omp for nowait
        for(int i=0; i<row_count; i++)
        {
            // printf("inside the loop...processing");
            for(int j=0; j<ncol; j++)
            {
                
                int8_t im0=data0[rownums0[i]*ncol+j]&imask;
                int8_t r0=(data0[rownums0[i]*ncol+j]&rmask)>>4;
                if (r0 > 8){r0 = r0 - 16;}
                if (im0 > 8){im0 = im0 - 16;}
                

                int8_t im1=data1[rownums1[i]*ncol+j]&imask;
                int8_t r1=(data1[rownums1[i]*ncol+j]&rmask)>>4;
                if (r1 > 8){r1 = r1 - 16;}
                if (im1 > 8){im1 = im1 - 16;}
                // printf("%d J%d ... %d J%d\n",r0,im0, r1,im1);

                sum_r_pvt[j] = sum_r_pvt[j] + r0*r1 + im0*im1;
                sum_im_pvt[j] = sum_im_pvt[j] + r1*im0 - r0*im1;
            }
        }
        #pragma omp critical
        {
            for(int k=0; k<ncol; k++)
            {
                // printf("setting real xcorr of k=%d as %d\n",k, sum_r_pvt[k]);
                xcorr[2*k] = xcorr[2*k] + sum_r_pvt[k];
                xcorr[2*k+1] = xcorr[2*k+1] + sum_im_pvt[k];
            }
        }
    }
    return row_count;
}

void avg_xcorr_1bit(uint8_t * data0, uint8_t * data1, float * xcorr, int nchan, const uint32_t nspec, const uint32_t ncol)
{
    // printf("entered corr func\n");
    //+2.1bil to -2.4bil, should be enough, and compatible with float32
    for(int i=0; i<nchan; i++)
    {
        xcorr[2*i]=0;
        xcorr[2*i+1]=0;
    }
    // printf("nspec is %d\n", nspec);
    #pragma omp parallel
    {
        // printf("INSIDE BLOCK\n");
        //init
        int32_t sum_r_pvt[nchan], sum_im_pvt[nchan];

        for(int i=0;i<nchan;i++)
        {
            sum_r_pvt[i]=0;
            sum_im_pvt[i]=0;
        }

        // printf("About to enter OUTER FOR\n");
        // printf("nspec is %d\n", nspec);
        #pragma omp for nowait
        for(int i=0; i<nspec; i++)
        {
            // printf("HELL\n");
            // fflush(stdout);
            int8_t c0r0,c0i0,c1r0,c1i0,c2r0,c2i0,c3r0,c3i0,c0r1,c0i1,c1r1,c1i1,c2r1,c2i1,c3r1,c3i1;
            int idx, colidx;
            // printf("enter outer for\n");
            // deal with the very last byte later. see switch-case below.
            for(int j=0; j<ncol-1; j++)
            {
                idx = i*ncol + j;
                c0r0 = (data0[idx]>>7)&1;
                c0i0 = (data0[idx]>>6)&1;
                c1r0 = (data0[idx]>>5)&1;
                c1i0 = (data0[idx]>>4)&1;
                c2r0 = (data0[idx]>>3)&1;
                c2i0 = (data0[idx]>>2)&1;
                c3r0 = (data0[idx]>>1)&1;
                c3i0 = (data0[idx])&1;

                c0r1 = (data1[idx]>>7)&1;  
                c0i1 = (data1[idx]>>6)&1;
                c1r1 = (data1[idx]>>5)&1;
                c1i1 = (data1[idx]>>4)&1;
                c2r1 = (data1[idx]>>3)&1;
                c2i1 = (data1[idx]>>2)&1;
                c3r1 = (data1[idx]>>1)&1;
                c3i1 = (data1[idx])&1;		
                colidx = 4*j;
                sum_r_pvt[colidx] = sum_r_pvt[colidx]       - 2*((c0r0^c0r1) + (c0i0^c0i1))+2;
                sum_im_pvt[colidx] = sum_im_pvt[colidx]     - 2*((c0r1^c0i0) - (c0r0^c0i1));
                sum_r_pvt[colidx+1] = sum_r_pvt[colidx+1]   - 2*((c1r0^c1r1) + (c1i0^c1i1))+2;
                sum_im_pvt[colidx+1] = sum_im_pvt[colidx+1] - 2*((c1r1^c1i0) - (c1r0^c1i1));
                sum_r_pvt[colidx+2] = sum_r_pvt[colidx+2]   - 2*((c2r0^c2r1) + (c2i0^c2i1))+2;
                sum_im_pvt[colidx+2] = sum_im_pvt[colidx+2] - 2*((c2r1^c2i0) - (c2r0^c2i1));
                sum_r_pvt[colidx+3] = sum_r_pvt[colidx+3]   - 2*((c3r0^c3r1) + (c3i0^c3i1))+2;
                sum_im_pvt[colidx+3] = sum_im_pvt[colidx+3] - 2*((c3r1^c3i0) - (c3r0^c3i1));
            }
            int j = ncol-1;
            idx = i*ncol + j;
            colidx = 4*j;
            // printf("about to hit switch when j = %d and colid = %d\n",j,colidx);
            switch(nchan%4)
            {	
                case 0:
                    // printf("Inside switch, j = %d, colidx = %d\n", j, colidx);
                    c0r0 = (data0[idx]>>7)&1;
                    c0i0 = (data0[idx]>>6)&1;
                    c1r0 = (data0[idx]>>5)&1;
                    c1i0 = (data0[idx]>>4)&1;
                    c2r0 = (data0[idx]>>3)&1;
                    c2i0 = (data0[idx]>>2)&1;
                    c3r0 = (data0[idx]>>1)&1;
                    c3i0 = (data0[idx])&1;
                    // printf("Byte passed for pol0 is %d\n", data0[idx]);
                    // printf("Bit signature is %d%d%d%d%d%d%d%d\n",c0r0,c0i0,c1r0,c1i0,c2r0,c2i0,c3r0,c3i0);

                    c0r1 = (data1[idx]>>7)&1;  
                    c0i1 = (data1[idx]>>6)&1;
                    c1r1 = (data1[idx]>>5)&1;
                    c1i1 = (data1[idx]>>4)&1;
                    c2r1 = (data1[idx]>>3)&1;
                    c2i1 = (data1[idx]>>2)&1;
                    c3r1 = (data1[idx]>>1)&1;
                    c3i1 = (data1[idx])&1;		
                    
                    sum_r_pvt[colidx] = sum_r_pvt[colidx]       - 2*((c0r0^c0r1) + (c0i0^c0i1))+2;
                    sum_im_pvt[colidx] = sum_im_pvt[colidx]     - 2*((c0r1^c0i0) - (c0r0^c0i1));
                    sum_r_pvt[colidx+1] = sum_r_pvt[colidx+1]   - 2*((c1r0^c1r1) + (c1i0^c1i1))+2;
                    sum_im_pvt[colidx+1] = sum_im_pvt[colidx+1] - 2*((c1r1^c1i0) - (c1r0^c1i1));
                    sum_r_pvt[colidx+2] = sum_r_pvt[colidx+2]   - 2*((c2r0^c2r1) + (c2i0^c2i1))+2;
                    sum_im_pvt[colidx+2] = sum_im_pvt[colidx+2] - 2*((c2r1^c2i0) - (c2r0^c2i1));
                    sum_r_pvt[colidx+3] = sum_r_pvt[colidx+3]   - 2*((c3r0^c3r1) + (c3i0^c3i1))+2;
                    sum_im_pvt[colidx+3] = sum_im_pvt[colidx+3] - 2*((c3r1^c3i0) - (c3r0^c3i1));
                    break;
                case 3:
                    c0r0 = (data0[idx]>>7)&1;
                    c0i0 = (data0[idx]>>6)&1;
                    c1r0 = (data0[idx]>>5)&1;
                    c1i0 = (data0[idx]>>4)&1;
                    c2r0 = (data0[idx]>>3)&1;
                    c2i0 = (data0[idx]>>2)&1;

                    c0r1 = (data1[idx]>>7)&1;  
                    c0i1 = (data1[idx]>>6)&1;
                    c1r1 = (data1[idx]>>5)&1;
                    c1i1 = (data1[idx]>>4)&1;
                    c2r1 = (data1[idx]>>3)&1;
                    c2i1 = (data1[idx]>>2)&1;
                    sum_r_pvt[colidx] = sum_r_pvt[colidx]       - 2*((c0r0^c0r1) + (c0i0^c0i1))+2;
                    sum_im_pvt[colidx] = sum_im_pvt[colidx]     - 2*((c0r1^c0i0) - (c0r0^c0i1));
                    sum_r_pvt[colidx+1] = sum_r_pvt[colidx+1]   - 2*((c1r0^c1r1) + (c1i0^c1i1))+2;
                    sum_im_pvt[colidx+1] = sum_im_pvt[colidx+1] - 2*((c1r1^c1i0) - (c1r0^c1i1));
                    sum_r_pvt[colidx+2] = sum_r_pvt[colidx+2]   - 2*((c2r0^c2r1) + (c2i0^c2i1))+2;
                    sum_im_pvt[colidx+2] = sum_im_pvt[colidx+2] - 2*((c2r1^c2i0) - (c2r0^c2i1));
                    break;
                case 2:
                    c0r0 = (data0[idx]>>7)&1;
                    c0i0 = (data0[idx]>>6)&1;
                    c1r0 = (data0[idx]>>5)&1;
                    c1i0 = (data0[idx]>>4)&1;

                    c0r1 = (data1[idx]>>7)&1;  
                    c0i1 = (data1[idx]>>6)&1;
                    c1r1 = (data1[idx]>>5)&1;
                    c1i1 = (data1[idx]>>4)&1;

                    sum_r_pvt[colidx] = sum_r_pvt[colidx]       - 2*((c0r0^c0r1) + (c0i0^c0i1))+2;
                    sum_im_pvt[colidx] = sum_im_pvt[colidx]     - 2*((c0r1^c0i0) - (c0r0^c0i1));
                    sum_r_pvt[colidx+1] = sum_r_pvt[colidx+1]   - 2*((c1r0^c1r1) + (c1i0^c1i1))+2;
                    sum_im_pvt[colidx+1] = sum_im_pvt[colidx+1] - 2*((c1r1^c1i0) - (c1r0^c1i1));
                    break;
                case 1:
                    c0r0 = (data0[idx]>>7)&1;
                    c0i0 = (data0[idx]>>6)&1;

                    c0r1 = (data1[idx]>>7)&1;  
                    c0i1 = (data1[idx]>>6)&1;

                    sum_r_pvt[colidx] = sum_r_pvt[colidx]       - 2*((c0r0^c0r1) + (c0i0^c0i1))+2;
                    sum_im_pvt[colidx] = sum_im_pvt[colidx]     - 2*((c0r1^c0i0) - (c0r0^c0i1));
                    break;
            }

        }
        #pragma omp critical
        {
            for(int k=0; k<nchan; k++)
            {
                // printf("setting real xcorr of k=%d as %d\n",k, sum_r_pvt[k]);
                xcorr[2*k] = xcorr[2*k] + sum_r_pvt[k];
                xcorr[2*k+1] = xcorr[2*k+1] + sum_im_pvt[k];
            }
        }
    }
}

void avg_xcorr_1bit_vanvleck(uint8_t * data0, uint8_t * data1, float * R0, float * R1, float * IM0, float * IM1, 
    int nchan, const uint32_t nrows, const uint32_t ncol)
{
    printf("entered corr func\n");
    //+2.1bil to -2.4bil, should be enough, and compatible with float32
    for(int i=0; i<nchan; i++)
    {
        R0[i]=0; R1[i]=0; IM0[i]=0; IM1[i]=0;
    }
    printf("nrows is %d\n", nrows);
    #pragma omp parallel
    {
        // printf("INSIDE BLOCK\n");
        //init
        int32_t sum_R0_pvt[nchan], sum_R1_pvt[nchan], sum_IM0_pvt[nchan], sum_IM1_pvt[nchan];

        for(int i=0;i<nchan;i++)
        {
            sum_R0_pvt[i]=0;
            sum_R1_pvt[i]=0;
            sum_IM0_pvt[i]=0;
            sum_IM1_pvt[i]=0;
        }

        // printf("About to enter OUTER FOR\n");
        // printf("nspec is %d\n", nspec);
        #pragma omp for nowait
        for(int i=0; i<nrows; i++)
        {
            // printf("HELL\n");
            // fflush(stdout);
            int8_t c0r0,c0i0,c1r0,c1i0,c2r0,c2i0,c3r0,c3i0,c0r1,c0i1,c1r1,c1i1,c2r1,c2i1,c3r1,c3i1;
            int idx, colidx;
            // printf("enter outer for\n");
            for(int j=0; j<ncol; j++)
            {
                idx = i*ncol + j;
                c0r0 = (data0[idx]>>7)&1;
                c0i0 = (data0[idx]>>6)&1;
                c1r0 = (data0[idx]>>5)&1;
                c1i0 = (data0[idx]>>4)&1;
                c2r0 = (data0[idx]>>3)&1;
                c2i0 = (data0[idx]>>2)&1;
                c3r0 = (data0[idx]>>1)&1;
                c3i0 = (data0[idx])&1;

                c0r1 = (data1[idx]>>7)&1;  
                c0i1 = (data1[idx]>>6)&1;
                c1r1 = (data1[idx]>>5)&1;
                c1i1 = (data1[idx]>>4)&1;
                c2r1 = (data1[idx]>>3)&1;
                c2i1 = (data1[idx]>>2)&1;
                c3r1 = (data1[idx]>>1)&1;
                c3i1 = (data1[idx])&1;		
                colidx = 4*j;
                sum_R0_pvt[colidx] = sum_R0_pvt[colidx] - 2*(c0r0^c0r1) + 1;
                sum_R1_pvt[colidx] = sum_R1_pvt[colidx] - 2*(c0i0^c0i1) + 1;
                sum_IM0_pvt[colidx] = sum_IM0_pvt[colidx] -2*(c0r0^c0i1) + 1;
                sum_IM1_pvt[colidx] = sum_IM1_pvt[colidx] -2*(c0r1^c0i0) + 1;

                sum_R0_pvt[colidx+1] = sum_R0_pvt[colidx+1] - 2*(c1r0^c1r1) + 1;
                sum_R1_pvt[colidx+1] = sum_R1_pvt[colidx+1] - 2*(c1i0^c1i1) + 1;
                sum_IM0_pvt[colidx+1] = sum_IM0_pvt[colidx+1] -2*(c1r0^c1i1) + 1;
                sum_IM1_pvt[colidx+1] = sum_IM1_pvt[colidx+1] -2*(c1r1^c1i0) + 1;

                sum_R0_pvt[colidx+2] = sum_R0_pvt[colidx+2] - 2*(c2r0^c2r1) + 1;
                sum_R1_pvt[colidx+2] = sum_R1_pvt[colidx+2] - 2*(c2i0^c2i1) + 1;
                sum_IM0_pvt[colidx+2] = sum_IM0_pvt[colidx+2] -2*(c2r0^c2i1) + 1;
                sum_IM1_pvt[colidx+2] = sum_IM1_pvt[colidx+2] -2*(c2r1^c2i0) + 1;

                sum_R0_pvt[colidx+3] = sum_R0_pvt[colidx+3] - 2*(c3r0^c3r1) + 1;
                sum_R1_pvt[colidx+3] = sum_R1_pvt[colidx+3] - 2*(c3i0^c3i1) + 1;
                sum_IM0_pvt[colidx+3] = sum_IM0_pvt[colidx+3] -2*(c3r0^c3i1) + 1;
                sum_IM1_pvt[colidx+3] = sum_IM1_pvt[colidx+3] -2*(c3r1^c3i0) + 1;

            }

        }
        #pragma omp critical
        {
            for(int k=0; k<nchan; k++)
            {
                R0[k] = R0[k] + sum_R0_pvt[k];
                R1[k] = R1[k] + sum_R1_pvt[k];
                IM0[k] = IM0[k] + sum_IM0_pvt[k];
                IM1[k] = IM1[k] + sum_IM1_pvt[k];
            }
        }
    }
}


int avg_xcorr_1bit_vanvleck_2ant(uint8_t * data0, uint8_t * data1, float * R0, float * R1, float * IM0, float * IM1, 
    int64_t* specnum0, int64_t* specnum1, const int64_t idxstart0, const int64_t idxstart1, const uint32_t nrows0, const uint32_t nrows1, const uint32_t ncol, const int nchan)
{
    int rownums0[nrows0], rownums1[nrows1], row_count=0, i=0,j=0;
    uint8_t imask=15;
    uint8_t rmask=255-15;

    while((i<nrows0)&&(j<nrows1))
    {
        
        if((specnum0[i]-idxstart0)==(specnum1[j]-idxstart1))
        {
            rownums0[row_count]=i;
            rownums1[row_count]=j;
            row_count=row_count+1;
            i=i+1;
            j=j+1;
        }
        else if((specnum0[i]-idxstart0)>(specnum1[j]-idxstart1)) {j=j+1;}
        else {i=i+1;}
    }
    // printf("FROM C: rownums selected are\n\n");
    // for(int i =0;i<row_count;i++)
    // {
    // 	printf("%d    %d\n", rownums0[i], rownums1[i]);
    // }
    // printf("row count is: %d\n", row_count);


    for(int i=0; i<nchan; i++)
    {
        R0[i]=0; R1[i]=0; IM0[i]=0; IM1[i]=0;
    }

    #pragma omp parallel
    {
        int32_t sum_R0_pvt[nchan], sum_R1_pvt[nchan], sum_IM0_pvt[nchan], sum_IM1_pvt[nchan];
        //init
        //+2.1bil to -2.4bil, should be enough, and compatible with float32
        for(int i=0;i<nchan;i++)
        {
            sum_R0_pvt[i]=0;
            sum_R1_pvt[i]=0;
            sum_IM0_pvt[i]=0;
            sum_IM1_pvt[i]=0;
        }

        #pragma omp for nowait
        for(int i=0; i<row_count; i++)
        {
            // printf("HELL\n");
            // fflush(stdout);
            int8_t c0r0,c0i0,c1r0,c1i0,c2r0,c2i0,c3r0,c3i0,c0r1,c0i1,c1r1,c1i1,c2r1,c2i1,c3r1,c3i1;
            int idx0, idx1, colidx;
            // printf("enter outer for\n");
            for(int j=0; j<ncol; j++)
            {
                idx0 = rownums0[i]*ncol + j;
                idx1 = rownums1[i]*ncol + j;
                c0r0 = (data0[idx0]>>7)&1;
                c0i0 = (data0[idx0]>>6)&1;
                c1r0 = (data0[idx0]>>5)&1;
                c1i0 = (data0[idx0]>>4)&1;
                c2r0 = (data0[idx0]>>3)&1;
                c2i0 = (data0[idx0]>>2)&1;
                c3r0 = (data0[idx0]>>1)&1;
                c3i0 = (data0[idx0])&1;

                c0r1 = (data1[idx1]>>7)&1;  
                c0i1 = (data1[idx1]>>6)&1;
                c1r1 = (data1[idx1]>>5)&1;
                c1i1 = (data1[idx1]>>4)&1;
                c2r1 = (data1[idx1]>>3)&1;
                c2i1 = (data1[idx1]>>2)&1;
                c3r1 = (data1[idx1]>>1)&1;
                c3i1 = (data1[idx1])&1;		
                colidx = 4*j;
                sum_R0_pvt[colidx] = sum_R0_pvt[colidx] - 2*(c0r0^c0r1) + 1;
                sum_R1_pvt[colidx] = sum_R1_pvt[colidx] - 2*(c0i0^c0i1) + 1;
                sum_IM0_pvt[colidx] = sum_IM0_pvt[colidx] -2*(c0r0^c0i1) + 1;
                sum_IM1_pvt[colidx] = sum_IM1_pvt[colidx] -2*(c0r1^c0i0) + 1;

                sum_R0_pvt[colidx+1] = sum_R0_pvt[colidx+1] - 2*(c1r0^c1r1) + 1;
                sum_R1_pvt[colidx+1] = sum_R1_pvt[colidx+1] - 2*(c1i0^c1i1) + 1;
                sum_IM0_pvt[colidx+1] = sum_IM0_pvt[colidx+1] -2*(c1r0^c1i1) + 1;
                sum_IM1_pvt[colidx+1] = sum_IM1_pvt[colidx+1] -2*(c1r1^c1i0) + 1;

                sum_R0_pvt[colidx+2] = sum_R0_pvt[colidx+2] - 2*(c2r0^c2r1) + 1;
                sum_R1_pvt[colidx+2] = sum_R1_pvt[colidx+2] - 2*(c2i0^c2i1) + 1;
                sum_IM0_pvt[colidx+2] = sum_IM0_pvt[colidx+2] -2*(c2r0^c2i1) + 1;
                sum_IM1_pvt[colidx+2] = sum_IM1_pvt[colidx+2] -2*(c2r1^c2i0) + 1;

                sum_R0_pvt[colidx+3] = sum_R0_pvt[colidx+3] - 2*(c3r0^c3r1) + 1;
                sum_R1_pvt[colidx+3] = sum_R1_pvt[colidx+3] - 2*(c3i0^c3i1) + 1;
                sum_IM0_pvt[colidx+3] = sum_IM0_pvt[colidx+3] -2*(c3r0^c3i1) + 1;
                sum_IM1_pvt[colidx+3] = sum_IM1_pvt[colidx+3] -2*(c3r1^c3i0) + 1;

            }

        }
        #pragma omp critical
        {
            for(int k=0; k<nchan; k++)
            {
                R0[k] = R0[k] + sum_R0_pvt[k];
                R1[k] = R1[k] + sum_R1_pvt[k];
                IM0[k] = IM0[k] + sum_IM0_pvt[k];
                IM1[k] = IM1[k] + sum_IM1_pvt[k];
            }
        }
    }
    return row_count;
}