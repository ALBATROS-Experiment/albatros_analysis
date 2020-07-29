import numpy as np

import pfb_helper as pfb
import time
import ctypes

import sys, os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from matplotlib import pyplot as plt
##compile with 
## gcc -O3 -o libpfb_helper.so -fPIC --shared pfb_helper.c -fopenmp
mylib=ctypes.cdll.LoadLibrary("libpfb_helper.so")
pfb_from_mat_c=mylib.pfb_from_mat
pfb_from_mat_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]

accumulate_pfb_transpose_complex_c=mylib.accumulate_pfb_transpose_complex
accumulate_pfb_transpose_complex_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]


plt.ion()


def quantize(mypfb,nlevel,dlevel):
    myscale=np.std(mypfb)
    mypfb=np.round(mypfb/(myscale*dlevel))
    thresh=(nlevel-1)//2
    mypfb[mypfb<-thresh]=-thresh
    mypfb[mypfb>thresh]=thresh
    return mypfb*(myscale*dlevel)

def bin_spec(spec,bin):
    nn=len(spec)//bin
    tmp=spec[:nn*bin]
    tmp=np.reshape(tmp,[nn,bin])
    return np.sum(tmp,axis=1)

def pfb_forward_c(x,ntap,nchan,window=pfb.sinc_hamming):
    x=np.real(x).copy()
    mat=(pfb.get_pfb_mat_sinc(nchan,ntap,window=window))
    #print(mat.shape)
    ii=2*(nchan-1)
    ll=len(x)//ii-(ntap-1)
    out=np.zeros([ll,nchan],dtype='complex')
    pfb_from_mat_c(out.ctypes.data,x.ctypes.data,mat.ctypes.data,nchan,mat.shape[1],ll)
    return out

def pfb_forward(x,ntap,nchan,window=pfb.sinc_hamming):
    x=np.real(x)
    ii=2*(nchan-1)
    ll=len(x)//ii-(ntap-1)
    mat=(pfb.get_pfb_mat_sinc(nchan,ntap,window=window))
    out=np.zeros([ll,nchan],dtype='complex')
    for i in range(ll):
        out[i,:]=np.dot(mat,x[i*ii:((i+ntap)*ii)])
    return out

def pfb_transpose(mypfb,ntap,window=pfb.sinc_hamming,do_c=False):
    nchan=mypfb.shape[1]
    mat=(pfb.get_pfb_mat_sinc(nchan,ntap,window=window))
    tmp=np.dot(mypfb,np.conj(mat))
    #ii=crap.shape[1]//ntap
    ii=2*(nchan-1)
    #print(ii)
    #assert(1==0)
    vec=np.zeros(ii*(tmp.shape[0]+ntap-1),dtype='complex')
    #print('tmp shape is ',tmp.shape)
    if do_c:
        accumulate_pfb_transpose_complex_c(vec.ctypes.data,tmp.ctypes.data,ntap,tmp.shape[0],nchan)
    else:
        for i in range(tmp.shape[0]):
            vec[(i*ii):(i*ii+tmp.shape[1])]+=tmp[i,:]
    return np.real(vec)
    #return vec

def filter_pfb(mypfb,mat=None,frac=0.85,dchunk=1,filt_mat=None):
    pfb_out=0*mypfb
    if filt_mat is None:
        u,s,v=np.linalg.svd(mat)
        ss=s[np.int(frac*len(s))]
        s_filt=(s/ss)**2
        s_filt[s_filt>1]=1
        filt_mat=np.dot(np.conj(u),np.dot(np.diag(s_filt),u.T))

    nchan=mypfb.shape[1]
    nblock=filt_mat.shape[0]//nchan    
    t1=time.time()
    nrep=mypfb.shape[0]-nblock
    for i in range(0,nrep,dchunk):
        vec=np.ravel(mypfb[i:i+nblock,:])
        vec_filt=np.dot(filt_mat,vec)
        pfb_out[i:i+nblock,:]=pfb_out[i:i+nblock,:]+np.reshape(vec_filt,[nblock,nchan])
    t2=time.time()
    print(t2-t1)
    return pfb_out*dchunk/nblock

def mymult(x,ntap,nchan,window=pfb.sinc_hamming):
    #y=pfb_forward(x,ntap,nchan,window)
    #xx=pfb_transpose(y,ntap,window)
    y=pfb_forward_c(x,ntap,nchan,window)
    xx=pfb_transpose(y,ntap,window,True)
    return xx

def mymult2(y,ntap,nchan,window=pfb.sinc_hamming):
    x=pfb_transpose(y,ntap,window)
    yy=pfb_forward(x,ntap,nchan,window)

    return yy


def myconj_grad(mypfb,ntap,window=pfb.sinc_hamming,niter=100):

    nchan=mypfb.shape[1]
    b=pfb_transpose(mypfb,ntap,window)
    #b=mypfb.copy()

    x=0.0*b
    r=b-mymult(x,ntap,nchan,window)
    p=r.copy()
    rtr=np.sum(np.conj(r)*r)
    for i in range(niter):
        #rtr=np.dot(np.conj(r),r)
        Ap=mymult(p,ntap,nchan,window)
        #pAp=np.dot(np.conj(p),Ap)
        pAp=pAp=np.sum(np.conj(p)*Ap)
        alpha=rtr/pAp
        x=x+alpha*p
        r=r-alpha*Ap
        #rtr_new=np.dot(np.conj(r),r)
        rtr_new=np.sum(np.conj(r)*r)
        beta=rtr_new/rtr
        #print(i,rtr_new,pAp)
        rtr=rtr_new
        p=r+beta*p

    return x
    

window=pfb.sinc_hamming
#window=pfb.sinc_window
nchan=16
ntap=4
nblock=ntap*8*2
nslice=1024*4*4

x=np.random.randn(nslice*2*(nchan-1))
t1=time.time()
mypfb=pfb_forward(x,ntap,nchan,window)
t2=time.time()
mypfb2=pfb_forward_c(x,ntap,nchan,window)
t3=time.time()
print('forward timings: ',t2-t1,t3-t2)

t1=time.time()
xx=pfb_transpose(mypfb,ntap,window)
t2=time.time()
xx2=pfb_transpose(mypfb,ntap,window,do_c=True)
t3=time.time()
print('transpose timings: ',t2-t1,t3-t2,np.std(xx2-xx))

y=myconj_grad(mypfb,ntap,window)
mypfb_quant=quantize(mypfb,15,0.4)
t1=time.time()
yquant=myconj_grad(mypfb_quant,ntap,window,niter=30)
t2=time.time()
print('conjgrad time is ',t2-t1)
ncut=20000
yy=yquant[ncut:-ncut]
yyft=np.fft.fft(yy)
spec=bin_spec(np.real(np.conj(yyft)*yyft),8)

plt.plot(spec)
plt.savefig('test.png')

t1=time.time();
y2=pfb.inverse_pfb(mypfb,window=window,ntap=ntap)
t2=time.time();
y2=np.ravel(y2)
print(t2-t1)

assert(1==0)



mypfb=np.zeros([nslice,nchan],dtype='complex')
mypfb[:,:]=np.random.randn(mypfb.shape[0],mypfb.shape[1])
mypfb[:,1:-1]=mypfb[:,1:-1]+1J*(np.random.randn(mypfb.shape[0],mypfb.shape[1]-2))

mypfb_filt,patches=pfb.filter_pfb_patches(mypfb,ntap=ntap,nblock=nblock,window=window,return_patches=True)

filt_mat=pfb.get_pfb_filter_mat(nchan,ntap,nblock,window=window)
filt_mat2=filt_mat*nblock  #npatch multiply is to account for the fact we're about to zero some stuff
filt_mat2[:(nblock*nchan)//2,:]=0
filt_mat2[(nblock//2+1)*nchan:,:]=0
mat=pfb.make_large_pfb_mat(nchan,ntap,nblock,window=window)
mypfb_filt2=filter_pfb(mypfb,filt_mat=filt_mat2)

#assert(1==0)
#if you have more data to filt, you could just call pfb.filter_pfb_patches(new_pfb,patches) now, without regenerating the filter
xx1=pfb.inverse_pfb(mypfb,window=window,ntap=ntap)
xx2=pfb.inverse_pfb(mypfb_filt,window=window,ntap=ntap)
xx3=pfb.inverse_pfb(mypfb_filt2,window=window,ntap=ntap)


x1=np.ravel(xx1)
x2=np.ravel(xx2)
x3=np.ravel(xx3)

ncut=20000
x1ft=np.fft.fft(x1[ncut:-ncut])
x2ft=np.fft.fft(x2[ncut:-ncut])
x3ft=np.fft.fft(x3[ncut:-ncut])
spec1=bin_spec(np.real(x1ft*np.conj(x1ft)),8)
spec2=bin_spec(np.real(x2ft*np.conj(x2ft)),8)
spec3=bin_spec(np.real(x3ft*np.conj(x3ft)),8)

