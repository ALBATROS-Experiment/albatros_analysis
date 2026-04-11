import numpy as np
import numba as nb

@nb.njit(parallel=True)
def apply_delay(arr, out, delay, freqs):
    # apply delay to an array of complex electric field or their correlation
    # does exp( j 2 pi nu tau) sign of tau is user dependent
    # freqs should correspond to the columns of the nspec x nchan array
    nspec = arr.shape[0]
    nchan = arr.shape[1]
    for i in nb.prange(nspec):
        for j in range(nchan):
            out[i, j] = arr[i, j] * np.exp(2j * np.pi * freqs[j] * delay[i])
    return out

@nb.njit(parallel=True)
def xcorr_avg(arr1,arr2,acclen):
    #helper function
    nblocks = arr1.shape[0]//acclen
    nchan = arr1.shape[1]
    out = np.zeros((nblocks,nchan),dtype=arr1.dtype)
    for i in nb.prange(nblocks):
        for j in range(acclen):
            for k in range(nchan):
                out[i,k] += arr1[i*acclen + j,k]*np.conj(arr2[i*acclen + j,k])
        out[i,:]/=acclen
    return out

@nb.njit()
def func(tau_t, data, freq, weights):
    ''' 
    Get residuals for given tau at single time sample, for all blines

    Notes:
    - we set the reference antenna to zero delay: all other ants are with respect to it
    - therefore tau_t has length nant-1
    - residuals is 1D, and splits real and imag residuals up for fitting
    - convention is that timing difference is ai-aj
    '''
    # for one time sample
    nant=6
    nbl = nant*(nant-1)//2
    nfreq = len(freq)
    n_eval = nfreq * nbl
    residuals = np.zeros((2 * n_eval,), dtype="float64")
    for j in range(nfreq):
        blnum = 0
        two_pi_nu = 2 * np.pi * freq[j]
        for ai in range(nant):
            for aj in range(ai + 1, nant):
                rowidx = j * nbl + blnum
                w = weights[blnum]
                if ai == 0:
                    pred = (
                        -two_pi_nu * tau_t[aj - 1]
                    )  # ant 0 is refant
                else:
                    pred = (
                        two_pi_nu * (tau_t[ai - 1] - tau_t[aj - 1])
                    )
                y = np.exp(1j * pred) - np.exp(1j*data[rowidx])
                residuals[rowidx] = w * np.real(y)
                residuals[rowidx + n_eval] = w * np.imag(y)
                blnum += 1
    return residuals

@nb.njit()
def jac(tau_t, data, freq, weights):
    '''
    Get the Jacobian for given tau at single time sample, for all blines

    Notes:
    - Antenna 0 is the reference so its delay is zero
    - therefore tau_t has length nant-1
    '''
    
    nant=6
    nbl = nant*(nant-1)//2
    nfreq = len(freq)
    n_eval = nfreq * nbl
    J = np.zeros((2 * n_eval, nant-1), dtype="float64")
    for j in range(nfreq):
        blnum = 0
        two_pi_nu = 2 * np.pi * freq[j]
        for ai in range(nant):
            for aj in range(ai + 1, nant):
                rowidx = j * nbl + blnum
                w = weights[blnum]
                if ai == 0:
                    pred = (
                        -two_pi_nu * tau_t[aj - 1]
                    )  # ant 0 is refant
                    ym = np.exp(1j * pred)
                    dtheta = 1j * ym * w
                    re = two_pi_nu * np.real(dtheta)
                    im = two_pi_nu * np.imag(dtheta)
                    J[rowidx, aj - 1] = -re
                    J[rowidx + n_eval, aj - 1] = -im
                else:
                    pred = (
                        two_pi_nu * (tau_t[ai - 1] - tau_t[aj - 1])
                    )
                    ym = np.exp(1j * pred)
                    dtheta = 1j * ym * w
                    re = two_pi_nu * np.real(dtheta)
                    im = two_pi_nu * np.imag(dtheta)
                    J[rowidx, ai - 1] = re
                    J[rowidx, aj - 1] = -re
                    J[rowidx + n_eval, ai - 1] = im
                    J[rowidx + n_eval, aj - 1] = -im
                blnum += 1
    return J

@nb.njit()
def symmetrize(A):
    nr, nc = A.shape
    for i in range(nr):
        for j in range(i, nc):
            A[j, i] = A[i, j]

def get_grammian(nant):
    nbl = (nant - 1) * nant // 2
    Ag = np.zeros((nbl, nant - 1), dtype="float64")
    b = 0
    for i in range(nant):
        for j in range(i + 1, nant):
            # print(i,j)
            if i == 0:
                Ag[b, j - 1] = -1
            else:
                Ag[b, i - 1] = 1
                Ag[b, j - 1] = -1
            b += 1
    return Ag

def get_AtA_Atd(data, Ag, noise_var, nu, nant, nfreq, ntime, fit_constant=True):
    # data is ntime, nfreq, nbl, C-major
    nparam = (nant - 1) * ntime
    if fit_constant:
        nparam += nant - 1
    nbl = (nant - 1) * nant // 2
    npertime = nfreq * nbl
    myAtA = np.empty((nparam, nparam), dtype="float64")
    myAtd = np.empty((nparam,), dtype="float64")
    myAtA[:] = 0.0
    myAtd[:] = 0.0
    two_pi = 2 * np.pi
    Snu2 = np.sum(nu**2) * two_pi**2
    Snu = np.sum(nu) * two_pi
    print("Snu", Snu, "Snu2", Snu2)
    bs = nant - 1  # block size
    # print("nant", nant, "nbl", nbl, "ntime", ntime, "nfreq", nfreq, "bs", bs)
    # print("d shape", d.shape)
    # print("n pertime", npertime)
    for i in range(ntime):
        Agw = Ag / noise_var[i, :][:, None]  # whitened A
        myAtA[i * bs : (i + 1) * bs, i * bs : (i + 1) * bs] = Ag.T @ Agw * Snu2
        if fit_constant:
            myAtA[i * bs : (i + 1) * bs, ntime * bs :] = Ag.T @ Agw * Snu
            myAtA[ntime * bs :, ntime * bs :] += Ag.T @ Agw * nfreq
        for j in range(nfreq):
            idx = i * npertime + j * nbl
            vec = Agw.T @ data[i, j, :]
            myAtd[i * bs : (i + 1) * bs] += two_pi * nu[j] * vec
            if fit_constant:
                myAtd[-bs:] += vec
    symmetrize(myAtA)
    return myAtA, myAtd

def get_AtA_Atd_independent(data, Ag, noise_var, nu, nant, nfreq, ntime):
    # nparam is now 2 * (nant-1) * ntime
    # Packed as: [tau_t0, tau_t1, ..., tau_tn, phi_t0, phi_t1, ..., phi_tn]
    bs = nant - 1  
    nparam = 2 * bs * ntime
    nbl = (nant - 1) * nant // 2
    npertime = nfreq * nbl
    
    myAtA = np.zeros((nparam, nparam), dtype="float64")
    myAtd = np.zeros((nparam,), dtype="float64")
    
    two_pi = 2 * np.pi
    Snu2 = np.sum(nu**2) * (two_pi**2)
    Snu = np.sum(nu) * two_pi
    
    # Offsets for the two parameter blocks
    tau_start = 0
    phi_start = ntime * bs

    for i in range(ntime):
        # 1. Compute the antenna-based Gramian for this time slice
        # noise_var[i, :] is the variance per baseline
        Agw = Ag / noise_var[i, :][:, None]
        G = Ag.T @ Agw  # Shape (bs, bs)

        # 2. Determine indices for this specific time block
        idx_tau = tau_start + i * bs
        idx_phi = phi_start + i * bs

        # 3. Fill the AtA blocks (Block Diagonal)
        # Top-left: tau-tau correlation
        myAtA[idx_tau : idx_tau + bs, idx_tau : idx_tau + bs] = G * Snu2
        # Bottom-right: phi-phi correlation
        myAtA[idx_phi : idx_phi + bs, idx_phi : idx_phi + bs] = G * nfreq
        # Off-diagonals: tau-phi correlation
        myAtA[idx_tau : idx_tau + bs, idx_phi : idx_phi + bs] = G * Snu
        myAtA[idx_phi : idx_phi + bs, idx_tau : idx_tau + bs] = G * Snu

        # 4. Fill the Atd (RHS)
        for j in range(nfreq):
            # Projection of baseline data onto antenna space
            vec = Agw.T @ data[i, j, :]
            
            # Accumulate into the tau slot (scaled by frequency)
            myAtd[idx_tau : idx_tau + bs] += (two_pi * nu[j]) * vec
            # Accumulate into the phi slot
            myAtd[idx_phi : idx_phi + bs] += vec

    # myAtA is naturally symmetric in this construction, 
    # but we can call symmetrize for safety if needed
    # symmetrize(myAtA) 
    
    return myAtA, myAtd

# @nb.njit(parallel=True)
# def chisq_lm(params,data,B,nu,weights,tau_ref,scale,nant=6):
#     # data is now for each baseline
#     # params is now 2-D (nant-1) x n_params_per_ant
#     # weights is 2-D ntime x nbl
#     nbl = data.shape[2]
#     ntime = data.shape[0]
#     nfreq = data.shape[1]
#     params = params.reshape(nant-1,-1)
#     n_eval = ntime*nfreq*nbl
#     nk = B.shape[1] #spline basis matrix (ntime , nk)
#     residuals = np.empty((2*ntime*nfreq*nbl,),dtype='float64')
#     for i in nb.prange(2*n_eval):
#         residuals[i]=0
#     tau_t = np.zeros((nant-1, ntime),dtype='float64')
#     for i in range(0,nant-1):
#         ci=params[i,:-1] # basis coeffs
#         tau_t[i,:]= tau_ref[i]  + scale[i]*B@ci
#     phi0 = params[:,-1].copy()
#     tau_t = tau_t.T.copy() #transpose for faster access since nbl << ntime
#     # fill pred_phase and Jacobian
#     for i in nb.prange(ntime):
#         for j in range(nfreq):
#             blnum=0
#             two_pi_nu = 2*np.pi*nu[j]
#             for ai in range(nant):
#                 for aj in range(ai+1,nant):
#                     rowidx = i*nfreq*nbl + j*nbl + blnum
#                     if ai==0:
#                         pred = -two_pi_nu*tau_t[i,aj-1] - phi0[aj-1] #ant 0 is refant
#                     else:
#                         pred = two_pi_nu*(tau_t[i,ai-1]-tau_t[i,aj-1]) + phi0[ai-1]-phi0[aj-1]
#                     res = np.exp(1j*data[i,j,blnum])-np.exp(1j*pred)
#                     w=weights[i, blnum]
#                     residuals[rowidx]=w*np.real(res)
#                     residuals[rowidx+n_eval]=w*np.imag(res)
#                     blnum+=1
#     return residuals


@nb.njit(parallel=True)
def chisq_lm(params, data, B, nu, weights, tau_ref, scale, lamda, nant=6):
    # data is now for each baseline
    # params is now 2-D (nant-1) x n_params_per_ant
    # weights is 2-D ntime x nbl
    nbl = data.shape[2]
    ntime = data.shape[0]
    nfreq = data.shape[1]
    params = params.reshape(nant - 1, -1)
    nparam_per_ant = params.shape[1]
    n_eval = ntime * nfreq * nbl
    nk = B.shape[1]  # spline basis matrix (ntime , nk)
    prior_length = (nant - 1) * (
        params.shape[1] - 1
    )  # number of total spline coefficients
    residuals = np.empty((2 * ntime * nfreq * nbl + prior_length,), dtype="float64")
    for i in nb.prange(2 * n_eval + prior_length):
        residuals[i] = 0
    tau_t = np.zeros((nant - 1, ntime), dtype="float64")
    for i in range(0, nant - 1):
        ci = params[i, :-1]  # basis coeffs
        tau_t[i, :] = tau_ref[i] + scale[i] * B @ ci
    phi0 = params[:, -1].copy()
    tau_t = tau_t.T.copy()  # transpose for faster access since nbl << ntime
    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        for j in range(nfreq):
            blnum = 0
            two_pi_nu = 2 * np.pi * nu[j]
            for ai in range(nant):
                for aj in range(ai + 1, nant):
                    rowidx = i * nfreq * nbl + j * nbl + blnum
                    if ai == 0:
                        pred = (
                            -two_pi_nu * tau_t[i, aj - 1] - phi0[aj - 1]
                        )  # ant 0 is refant
                    else:
                        pred = (
                            two_pi_nu * (tau_t[i, ai - 1] - tau_t[i, aj - 1])
                            + phi0[ai - 1]
                            - phi0[aj - 1]
                        )
                    res = np.exp(1j * data[i, j, blnum]) - np.exp(1j * pred)
                    w = weights[i, blnum]
                    residuals[rowidx] = w * np.real(res)
                    residuals[rowidx + n_eval] = w * np.imag(res)
                    blnum += 1
    idx = 2 * n_eval
    for i in range(params.shape[0]):  # over nant-1
        for j in range(params.shape[1] - 1):  # over num of spline coeffs per antenna
            residuals[idx + i * (nparam_per_ant) + j] = (
                np.sqrt(lamda) * params[i, j]
            )  # skip the constant phi0, no prior on that
    return residuals


@nb.njit(parallel=True)
def jac_lm(params, data, B, nu, weights, tau_ref, scale, lamda, nant=6):
    # data is now for each baseline
    # params is now 2-D (nant-1) x n_params_per_ant
    # weights is 2-D ntime x nbl
    nbl = data.shape[2]
    ntime = data.shape[0]
    nfreq = data.shape[1]
    params = params.reshape(nant - 1, -1)
    nparam = np.size(params)  # coeffs and constant phi0 for nant-1
    nparam_per_ant = params.shape[1]
    n_eval = ntime * nfreq * nbl
    prior_length = (nant - 1) * (
        params.shape[1] - 1
    )  # number of total spline coefficients
    nk = B.shape[1]  # spline basis matrix (ntime , nk)
    J = np.empty((2 * n_eval + prior_length, nparam), dtype="float64")
    Jflat = np.ravel(J)
    nn = J.shape[0] * J.shape[1]
    for m in nb.prange(nn):
        Jflat[m] = 0
    tau_t = np.zeros((nant - 1, ntime), dtype="float64")
    phi0 = params[:, -1].copy()
    blnum = 0
    for i in range(0, nant - 1):
        ci = params[i, :-1]  # basis coeffs
        tau_t[i, :] = tau_ref[i] + scale[i] * B @ ci
    tau_t = tau_t.T.copy()  # transpose for faster access since nbl << ntime
    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        for j in range(nfreq):
            blnum = 0
            two_pi_nu = 2 * np.pi * nu[j]
            for ai in range(nant):
                for aj in range(ai + 1, nant):
                    rowidx = i * nfreq * nbl + j * nbl + blnum
                    w = weights[i, blnum]
                    if ai == 0:
                        pred = (
                            -two_pi_nu * tau_t[i, aj - 1] - phi0[aj - 1]
                        )  # ant 0 is refant
                        ym = np.exp(1j * pred)
                        dtheta = -1j * ym * w
                        for k in range(nk):
                            temp = two_pi_nu * B[i, k]
                            colidx2 = (
                                aj - 1
                            ) * nparam_per_ant + k  # only gotta fill antenna i+1 and onwards for ant 0
                            re = temp * np.real(dtheta)
                            im = temp * np.imag(dtheta)
                            J[rowidx, colidx2] = -re
                            J[rowidx + n_eval, colidx2] = -im
                        re = np.real(dtheta)
                        im = np.imag(dtheta)
                        colidx2 = (aj - 1) * nparam_per_ant + nk
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx2] = -im
                    else:
                        pred = (
                            two_pi_nu * (tau_t[i, ai - 1] - tau_t[i, aj - 1])
                            + phi0[ai - 1]
                            - phi0[aj - 1]
                        )
                        ym = np.exp(1j * pred)
                        dtheta = -1j * ym * w
                        for k in range(nk):
                            temp = two_pi_nu * B[i, k]
                            colidx1 = (ai - 1) * nparam_per_ant + k
                            colidx2 = (aj - 1) * nparam_per_ant + k
                            re = temp * np.real(dtheta)
                            im = temp * np.imag(dtheta)
                            J[rowidx, colidx1] = re
                            J[rowidx, colidx2] = -re
                            J[rowidx + n_eval, colidx1] = im
                            J[rowidx + n_eval, colidx2] = -im
                        re = np.real(dtheta)
                        im = np.imag(dtheta)
                        colidx1 = (ai - 1) * nparam_per_ant + nk
                        colidx2 = (aj - 1) * nparam_per_ant + nk
                        J[rowidx, colidx1] = re
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx1] = im
                        J[rowidx + n_eval, colidx2] = -im
                    blnum += 1
    idx = 2 * n_eval
    for i in range(nant - 1):
        for j in range(nparam_per_ant - 1):
            k = i * nparam_per_ant + j
            J[idx + k, k] = np.sqrt(lamda)
    return J

@nb.njit(parallel=True)
def chisq_lm_full_noconst(params, data, nu, weights, nant=6):
    # data is now for each baseline
    # no constant phi0
    # params is now 2-D (nant-1) x n_params_per_ant
    # weights is 2-D ntime x nbl
    nbl = data.shape[2]
    ntime = data.shape[0]
    nfreq = data.shape[1]

    n_eval = ntime * nfreq * nbl
    residuals = np.empty((2 * n_eval,), dtype="float64")
    for i in nb.prange(2 * n_eval):
        residuals[i] = 0

    tau_t = params[: ntime * (nant - 1)].reshape(-1, nant - 1)

    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        for j in range(nfreq):
            blnum = 0
            two_pi_nu = 2 * np.pi * nu[j]
            for ai in range(nant):
                for aj in range(ai + 1, nant):
                    rowidx = i * nfreq * nbl + j * nbl + blnum
                    if ai == 0:
                        pred = (
                            -two_pi_nu * tau_t[i, aj - 1]
                        )  # ant 0 is refant
                    else:
                        pred = (
                            two_pi_nu * (tau_t[i, ai - 1] - tau_t[i, aj - 1])
                        )
                    res = np.exp(1j * data[i, j, blnum]) - np.exp(1j * pred)
                    w = weights[i, blnum]
                    residuals[rowidx] = w * np.real(res)
                    residuals[rowidx + n_eval] = w * np.imag(res)
                    blnum += 1
    return residuals

@nb.njit(parallel=True)
def chisq_lm_full(params, data, nu, weights, lamda, nant=6):
    # data is now for each baseline
    # params is now 2-D (nant-1) x n_params_per_ant
    # weights is 2-D ntime x nbl
    nbl = data.shape[2]
    ntime = data.shape[0]
    nfreq = data.shape[1]

    n_eval = ntime * nfreq * nbl
    prior_length = ntime * (nant - 1)
    residuals = np.empty((2 * ntime * nfreq * nbl + prior_length,), dtype="float64")
    for i in nb.prange(2 * n_eval + prior_length):
        residuals[i] = 0

    tau_t = params[: ntime * (nant - 1)].reshape(-1, nant - 1)
    phi0 = params[ntime * (nant - 1) :]
    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        for j in range(nfreq):
            blnum = 0
            two_pi_nu = 2 * np.pi * nu[j]
            for ai in range(nant):
                for aj in range(ai + 1, nant):
                    rowidx = i * nfreq * nbl + j * nbl + blnum
                    if ai == 0:
                        pred = (
                            -two_pi_nu * tau_t[i, aj - 1] - phi0[aj - 1]
                        )  # ant 0 is refant
                    else:
                        pred = (
                            two_pi_nu * (tau_t[i, ai - 1] - tau_t[i, aj - 1])
                            + phi0[ai - 1]
                            - phi0[aj - 1]
                        )
                    res = np.exp(1j * data[i, j, blnum]) - np.exp(1j * pred)
                    w = weights[i, blnum]
                    residuals[rowidx] = w * np.real(res)
                    residuals[rowidx + n_eval] = w * np.imag(res)
                    blnum += 1
    idx = 2 * n_eval
    residuals[idx : idx + (nant - 1)] = np.sqrt(lamda) * (
        -2 * tau_t[0, :] + 2 * tau_t[1, :]
    )
    for i in range(1, ntime - 1):  # over nant-1
        for j in range(nant - 1):  # over num of spline coeffs per antenna
            residuals[idx + i * (nant - 1) + j] = np.sqrt(lamda) * (
                tau_t[i - 1, j] - 2 * tau_t[i, j] + tau_t[i + 1, j]
            )
    i = ntime - 1
    residuals[idx + i * (nant - 1) : idx + i * (nant - 1) + (nant - 1)] = np.sqrt(
        lamda
    ) * (-2 * tau_t[i, :] + 2 * tau_t[i - 1, :])
    return residuals

@nb.njit(parallel=True)
def jac_lm_full_noconst(params, data, nu, weights, nant=6):
    # this function is to fit one delay per time sample.
    # data is now for each baseline
    # params is now 2-D (nant-1) x ntime + 1
    # weights is 2-D ntime x nbl
    nbl = data.shape[2]
    ntime = data.shape[0]
    nfreq = data.shape[1]

    n_eval = ntime * nfreq * nbl
    J = np.empty(
        (2 * n_eval, (nant - 1) * ntime), dtype="float64" #all ant first, then next time
    )
    Jflat = np.ravel(J)
    nn = J.shape[0] * J.shape[1]
    for m in nb.prange(nn):
        Jflat[m] = 0.0
    
    tau_t = params[: ntime * (nant - 1)].reshape(-1, nant - 1)
    blnum = 0
    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        for j in range(nfreq):
            blnum = 0
            two_pi_nu = 2 * np.pi * nu[j]
            for ai in range(nant):
                for aj in range(ai + 1, nant):
                    rowidx = i * nfreq * nbl + j * nbl + blnum
                    w = weights[i, blnum]
                    if ai == 0:
                        pred = (
                            -two_pi_nu * tau_t[i, aj - 1]
                        )  # ant 0 is refant
                        ym = np.exp(1j * pred)
                        dtheta = -1j * ym * w
                        colidx2 = i * (nant - 1) + aj - 1
                        re = two_pi_nu * np.real(dtheta)
                        im = two_pi_nu * np.imag(dtheta)
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx2] = -im
                    else:
                        pred = (
                            two_pi_nu * (tau_t[i, ai - 1] - tau_t[i, aj - 1])
                        )
                        ym = np.exp(1j * pred)
                        dtheta = -1j * ym * w
                        colidx1 = i * (nant - 1) + ai - 1
                        colidx2 = i * (nant - 1) + aj - 1
                        re = two_pi_nu * np.real(dtheta)
                        im = two_pi_nu * np.imag(dtheta)
                        J[rowidx, colidx1] = re
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx1] = im
                        J[rowidx + n_eval, colidx2] = -im
                    blnum += 1
    return J

@nb.njit(parallel=True)
def jac_lm_full(params, data, nu, weights, lamda, nant=6):
    # this function is to fit one delay per time sample.
    # data is now for each baseline
    # params is now 2-D (nant-1) x ntime + 1
    # weights is 2-D ntime x nbl
    nbl = data.shape[2]
    ntime = data.shape[0]
    nfreq = data.shape[1]

    n_eval = ntime * nfreq * nbl
    prior_length = (nant - 1) * ntime
    J = np.empty(
        (2 * n_eval + prior_length, (nant - 1) * ntime + nant - 1), dtype="float64"
    )
    Jflat = np.ravel(J)
    nn = J.shape[0] * J.shape[1]
    for m in nb.prange(nn):
        Jflat[m] = 0.0
    tau_t = params[: ntime * (nant - 1)].reshape(-1, nant - 1)
    phi0 = params[ntime * (nant - 1) :]
    blnum = 0
    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        for j in range(nfreq):
            blnum = 0
            two_pi_nu = 2 * np.pi * nu[j]
            for ai in range(nant):
                for aj in range(ai + 1, nant):
                    rowidx = i * nfreq * nbl + j * nbl + blnum
                    w = weights[i, blnum]
                    if ai == 0:
                        pred = (
                            -two_pi_nu * tau_t[i, aj - 1] - phi0[aj - 1]
                        )  # ant 0 is refant
                        ym = np.exp(1j * pred)
                        dtheta = -1j * ym * w
                        colidx2 = i * (nant - 1) + aj - 1
                        re = two_pi_nu * np.real(dtheta)
                        im = two_pi_nu * np.imag(dtheta)
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx2] = -im
                        re = np.real(dtheta)
                        im = np.imag(dtheta)
                        colidx2 = ntime * (nant - 1) + aj - 1  # constant phi0
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx2] = -im
                    else:
                        pred = (
                            two_pi_nu * (tau_t[i, ai - 1] - tau_t[i, aj - 1])
                            + phi0[ai - 1]
                            - phi0[aj - 1]
                        )
                        ym = np.exp(1j * pred)
                        dtheta = -1j * ym * w
                        colidx1 = i * (nant - 1) + ai - 1
                        colidx2 = i * (nant - 1) + aj - 1
                        re = two_pi_nu * np.real(dtheta)
                        im = two_pi_nu * np.imag(dtheta)
                        J[rowidx, colidx1] = re
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx1] = im
                        J[rowidx + n_eval, colidx2] = -im
                        re = np.real(dtheta)
                        im = np.imag(dtheta)
                        colidx1 = ntime * (nant - 1) + ai - 1
                        colidx2 = ntime * (nant - 1) + aj - 1
                        J[rowidx, colidx1] = re
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx1] = im
                        J[rowidx + n_eval, colidx2] = -im
                    blnum += 1
    idx = 2 * n_eval
    rowidx = idx
    for j in range(nant - 1):
        colidx = 0
        J[rowidx + j, colidx + j] = -2 * np.sqrt(lamda)
        colidx = nant - 1
        J[rowidx + j, colidx + j] = 2 * np.sqrt(lamda)
    for i in range(1, ntime - 1):
        rowidx = idx + i * (nant - 1)
        for j in range(nant - 1):
            colidx = (i - 1) * (nant - 1)
            J[rowidx + j, colidx + j] = np.sqrt(lamda)
            colidx = i * (nant - 1)
            J[rowidx + j, colidx + j] = -2 * np.sqrt(lamda)
            colidx = (i + 1) * (nant - 1)
            J[rowidx + j, colidx + j] = np.sqrt(lamda)
    i = ntime - 1
    rowidx = idx + i * (nant - 1)
    for j in range(nant - 1):
        colidx = (i - 1) * (nant - 1)
        J[rowidx + j, colidx + j] = 2 * np.sqrt(lamda)
        colidx = i * (nant - 1)
        J[rowidx + j, colidx + j] = -2 * np.sqrt(lamda)
    return J


# @nb.njit(parallel=True)
# def jac_lm(params,data,B,nu,weights,tau_ref,scale,nant=6):
#     # data is now for each baseline
#     # params is now 2-D (nant-1) x n_params_per_ant
#     # weights is 2-D ntime x nbl
#     nbl = data.shape[2]
#     ntime = data.shape[0]
#     nfreq = data.shape[1]
#     params = params.reshape(nant-1,-1)
#     nparam = np.size(params) #coeffs and constant phi0 for nant-1
#     nparam_per_ant = params.shape[1]
#     n_eval = ntime*nfreq*nbl
#     nk = B.shape[1] #spline basis matrix (ntime , nk)
#     J = np.empty((2*ntime*nfreq*nbl,nparam),dtype='float64')
#     Jflat=np.ravel(J)
#     nn=2*ntime*nfreq*nbl*nparam
#     for i in nb.prange(nn):
#         Jflat[i]=0
#     tau_t = np.zeros((nant-1, ntime),dtype='float64')
#     phi0 = params[:,-1].copy()
#     blnum=0
#     for i in range(0,nant-1):
#         ci=params[i,:-1] # basis coeffs
#         tau_t[i,:]= tau_ref[i]  + scale[i]*B@ci
#     tau_t = tau_t.T.copy() #transpose for faster access since nbl << ntime
#     # fill pred_phase and Jacobian
#     for i in nb.prange(ntime):
#         for j in range(nfreq):
#             blnum=0
#             two_pi_nu = 2*np.pi*nu[j]
#             for ai in range(nant):
#                 for aj in range(ai+1,nant):
#                     rowidx = i*nfreq*nbl + j*nbl + blnum
#                     w = weights[i, blnum]
#                     if ai==0:
#                         pred = -two_pi_nu*tau_t[i,aj-1] - phi0[aj-1] #ant 0 is refant
#                         ym = np.exp(1j*pred)
#                         dtheta = -1j*ym*w
#                         for k in range(nk):
#                             temp = two_pi_nu*B[i,k]
#                             colidx2 = (aj-1)*nparam_per_ant + k #only gotta fill antenna i+1 and onwards for ant 0
#                             re = temp*np.real(dtheta)
#                             im = temp*np.imag(dtheta)
#                             J[rowidx, colidx2] = -re
#                             J[rowidx+n_eval,colidx2] = -im
#                         re = np.real(dtheta)
#                         im = np.imag(dtheta)
#                         colidx2 = (aj-1)*nparam_per_ant + nk
#                         J[rowidx, colidx2] = -re
#                         J[rowidx+n_eval,colidx2] = -im
#                     else:
#                         pred = two_pi_nu*(tau_t[i,ai-1]-tau_t[i,aj-1]) + phi0[ai-1]-phi0[aj-1]
#                         ym = np.exp(1j*pred)
#                         dtheta = -1j*ym*w
#                         for k in range(nk):
#                             temp = two_pi_nu*B[i,k]
#                             colidx1 = (ai-1)*nparam_per_ant + k
#                             colidx2 = (aj-1)*nparam_per_ant + k
#                             re = temp*np.real(dtheta)
#                             im = temp*np.imag(dtheta)
#                             J[rowidx, colidx1] = re
#                             J[rowidx, colidx2] = -re
#                             J[rowidx+n_eval,colidx1] = im
#                             J[rowidx+n_eval,colidx2] = -im
#                         re = np.real(dtheta)
#                         im = np.imag(dtheta)
#                         colidx1 = (ai-1)*nparam_per_ant + nk
#                         colidx2 = (aj-1)*nparam_per_ant + nk
#                         J[rowidx, colidx1] = re
#                         J[rowidx, colidx2] = -re
#                         J[rowidx+n_eval,colidx1] = im
#                         J[rowidx+n_eval,colidx2] = -im
#                     blnum+=1
#     return J

nb.njit(parallel=True)


def chisq_admm(tau_phi0, tau_smooth, u, z, rho, data, nu, weights, tau_ref, scale=1):
    ntime = data.shape[0]
    nfreq = data.shape[1]
    n_eval = data.shape[0] * data.shape[1]
    phi0 = tau_phi0[-1]
    tau = tau_phi0[:-1]
    res = np.zeros((2 * n_eval + ntime,), dtype="float64")
    for i in nb.prange(ntime):
        w = weights[i]
        for j in range(nfreq):
            two_pi_nu = 2 * np.pi * nu[j]
            pred = two_pi_nu * (tau_ref + scale * tau[i]) + phi0
            dtheta = np.exp(1j * data[i, j]) - np.exp(1j * pred)
            res[i * nfreq + j] = w * np.real(dtheta)
            res[i * nfreq + j + n_eval] = w * np.imag(dtheta)
        res[i + 2 * n_eval] = np.sqrt(rho) * (tau[i] - tau_smooth[i] - z[i] + u[i])
    return res


@nb.njit(parallel=True)
def jac_admm(tau_phi0, tau_smooth, u, z, rho, data, nu, weights, tau_ref, scale=1):
    n_eval = data.shape[0] * data.shape[1]
    ntime = data.shape[0]
    nfreq = data.shape[1]
    phi0 = tau_phi0[-1]
    tau = tau_phi0[:-1]
    J = np.zeros((2 * n_eval + ntime, ntime + 1), dtype="float64")
    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        w = weights[i]
        for j in range(nfreq):
            two_pi_nu = 2 * np.pi * nu[j]
            pred = two_pi_nu * (tau_ref + scale * tau[i]) + phi0
            ym = np.exp(1j * pred)
            dtheta = -1j * two_pi_nu * ym
            J[i * nfreq + j, i] = w * np.real(dtheta)
            J[i * nfreq + j + n_eval, i] = w * np.imag(dtheta)
            dtheta = -1j * ym
            J[i * nfreq + j, -1] = w * np.real(dtheta)
            J[i * nfreq + j + n_eval, -1] = w * np.imag(dtheta)
        J[i + 2 * n_eval, i] = np.sqrt(rho)
    return J


def lmsolver(
    xc, alpha, chan, scale=1, lamda=16, xtol=1e-6, ftol=1e-6, niter=10, debug=False
):
    c = 2 * np.pi * chan
    N = xc.shape[0]
    n = np.arange(N)
    # print("len chan", len(chan), "len n", len(n))
    lamda = lamda
    for ii in range(niter):
        if debug:
            print(
                f"------------------------------ LM iter {ii} -------------------------------"
            )
        xc_phased = (
            1e-3 * xc * np.exp(1j * c[np.newaxis, :] * n[:, np.newaxis] * alpha)
        )  # n is specnum
        f = -np.mean(np.abs(np.mean(xc_phased, axis=0)) ** 2)
        S0conj = np.conj(np.mean(xc_phased, axis=0))
        S1 = np.mean(xc_phased * n[:, np.newaxis] * c[np.newaxis, :], axis=0)
        S2 = np.mean(xc_phased * n[:, np.newaxis] ** 2 * c[np.newaxis, :] ** 2, axis=0)
        # print(np.abs(S0conj), np.abs(S1), np.abs(S2))
        df = np.imag(np.mean(S0conj * S1))
        ddf = np.real(np.mean(S0conj * S2)) - np.mean(np.abs(S1) ** 2)
        step = -df / (ddf + lamda * np.abs(ddf))
        alpha2 = alpha + step
        xc_phased = (
            1e-3 * xc * np.exp(1j * c[np.newaxis, :] * n[:, np.newaxis] * alpha2)
        )  # n is specnum
        f2 = -np.mean(np.abs(np.mean(xc_phased, axis=0)) ** 2)
        if debug:
            print(
                f"lamda: {lamda:5.3e} alpha: {alpha:5.3e} f: {f:5.3e}, df: {df:5.3e}, ddf: {ddf:5.3e}, step: {step:5.3e}, alpha2: {alpha2:5.3e} f2: {f2:5.3e}"
            )
        if f2 < f:  # proceeding towards minimization
            # accept
            if debug:
                print("accepting...")
            rel_alpha = (alpha2 - alpha) / alpha
            rel_f = (f2 - f) / f
            if debug:
                print(
                    f"rel alpha: {np.abs(rel_alpha):5.3e} rel f: {np.abs(rel_f):5.3e}"
                )
            alpha = alpha2
            lamda /= 2
            if np.abs(rel_alpha) < xtol or np.abs(rel_f) < ftol:
                if debug:
                    print("converged.")
                return alpha
        else:
            lamda *= 2
    raise Exception(f"Failed to converge in {niter} iterations.")
    return alpha


def objective_func(xc, alpha, chan):
    c = 2 * np.pi * chan
    N = xc.shape[0]
    n = np.arange(N)
    # print("len chan", len(chan), "len n", len(n), "xc shape", xc.shape)
    xc_phased = 1e-3 * xc * np.exp(1j * c[np.newaxis, :] * n[:, np.newaxis] * alpha)
    # print("xc phased shape", xc_phased.shape)
    f = -np.mean(np.abs(np.mean(xc_phased, axis=0)) ** 2)
    S0conj = np.conj(np.mean(xc_phased, axis=0))
    # print("S0conj shape", S0conj.shape)
    S1 = np.mean(xc_phased * n[:, np.newaxis] * c[np.newaxis, :], axis=0)
    # print("S1 shape", S1.shape)
    S2 = np.mean(xc_phased * n[:, np.newaxis] ** 2 * c[np.newaxis, :] ** 2, axis=0)
    # print("S2 shape", S2.shape)
    df = np.imag(np.mean(S0conj * S1))
    ddf = np.real(np.mean(S0conj * S2)) - np.mean(np.abs(S1) ** 2)
    return f, df, ddf
