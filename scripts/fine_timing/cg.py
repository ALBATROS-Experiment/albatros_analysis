import numpy as np
from matplotlib import pyplot as plt

def _get_adev_prior(m, ddsigma):
    # m = slopes only
    # neumann bd with f'[0] = f'[-1] = 0
    y = np.empty(len(m),dtype='float64')
    y[0] = 5*m[0]-6*m[1]+m[2]
    y[1] = -6*m[0]+9*m[1]-4*m[2]+m[3]
    y[2:-2] = m[:-4] - 4*m[1:-3] + 6*m[2:-2] - 4*m[3:-1] + m[4:]
    y[-2] = -6*m[-1]+9*m[-2]-4*m[-3]+m[-4]
    y[-1] = 5*m[-1]-6*m[-2]+m[-3]
    return y/ddsigma**2

def _get_forward(params,x):
    p = np.empty(len(params),dtype='float64')
    n = len(params)//2 #half slopes, half intercepts
    y1 = params[:n, None] * x + params[n:, None]
    p[:n] = (y1)@x
    p[n:] = np.sum(y1,axis=1)
    return p

def _get_r(params,x,y):
    n = len(params)//2 #half slopes, half intercepts
    yd = y.reshape(n,-1) 
    print("x", x.shape)
    print("yd", yd.shape)
    r = np.empty(len(params),dtype='float64') # r = b - Ax in CG
    y1 = params[:n, None] * x + params[n:, None] # shape [n,  len(x)]
    print("y1", y1.shape)
    r[:n] = (yd-y1)@x
    r[n:] = np.sum(yd,axis=1) - np.sum(y1,axis=1)
    return r

def _get_forward_adev(params,x, ddsigma, noise):
    p = np.empty(len(params),dtype='float64')
    n = len(params)//2 #half slopes, half intercepts
    y1 = params[:n, None] * x + params[n:, None]
    y1/=noise**2
    p[:n] = (y1)@x + _get_adev_prior(params[:n], ddsigma)
    p[n:] = np.sum(y1,axis=1)
    return p

def _get_r_adev(params,x,y, ddsigma, noise):
    n = len(params)//2 #half slopes, half intercepts
    yd = y.reshape(n,-1) / noise**2
    # print("x", x.shape)
    # print("yd", yd.shape)
    # print("y1", y1.shape)
    r = np.empty(len(params),dtype='float64') # r = b - Ax in CG
    y1 = params[:n, None] * x + params[n:, None] # shape [n,  len(x)]
    y1 /= noise**2
    r[:n] = (yd-y1)@x - _get_adev_prior(params[:n], ddsigma)
    r[n:] = np.sum(yd,axis=1) - np.sum(y1,axis=1)
    return r


def cg_linear_model(params, x, y, eps=1e-6, niter=20):
    i=0
    r = _get_r(params, x, y)
    d = r
    delta_new = r.T@r
    delta0 = delta_new
    while i < niter and delta_new > eps**2 * delta0:
        q = _get_forward(d,x)
        alpha = delta_new/(d.T@q)
        params += alpha*d
        if i % 10 == 0:
            r = _get_r(params, x, y)
        else:
            r = r - alpha*q
        delta_old = delta_new
        delta_new = r.T@r
        beta = delta_new/delta_old
        d = r + beta*d
        i+=1
    print(f"final residual after {i} iterations is", np.sqrt(delta_new))

def cg_linear_model_adev(params, x, y, ddsigma, noise, eps=1e-6, niter=50):
    i=0
    r = _get_r_adev(params, x, y, ddsigma, noise)
    d = r
    delta_new = r.T@r
    delta0 = delta_new
    while i < niter and delta_new > eps**2 * delta0:
        q = _get_forward_adev(d,x, ddsigma, noise)
        alpha = delta_new/(d.T@q)
        params += alpha*d
        if i % 10 == 0:
            r = _get_r_adev(params, x, y, ddsigma, noise)
        else:
            r = r - alpha*q
        delta_old = delta_new
        delta_new = r.T@r
        beta = delta_new/delta_old
        d = r + beta*d
        i+=1
    print(f"final residual after {i} iterations is", np.sqrt(delta_new))

# generate data
if __name__=="__main__":

    m = 3
    c = 1
    x = np.linspace(-1,1,100)
    y1 = m*x + c + np.random.randn(len(x)) #data 1
    y2 = (m+2)*x + c + np.random.randn(len(x)) #data 2
    m1,c1 = np.polyfit(x,y1,1)
    m2,c2 = np.polyfit(x,y2,1)
    y = np.hstack([y1,y2]) #observed data vector
    yt = np.hstack([m*x + c,(m+2)*x + c]) #truth
    print('y shape', y.shape)
    params = np.asarray([m,m+2,c,0], dtype='float64')
    res =_get_r(params,x,yt)
    assert np.all(res)==0
    print('res', res)
    params = np.asarray([1,1,0,0], dtype='float64')
    cg_linear_model(params, x, y, eps=1e-6)
    print('params', params)
    print(m1,m2,c1,c2)

