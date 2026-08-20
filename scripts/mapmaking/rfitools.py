import numpy as np
import numba as nb
import time as time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import os
import glob
import re

def load_all_parts(dir_path):
    """
    Finds all visibility part files in a directory, sorts them numerically,
    and loads them into a single pre-allocated big array.
    """
    # Find all part files
    pattern = os.path.join(dir_path, "*_part*.npy")
    part_files = glob.glob(pattern)
    
    if not part_files:
        print(f"No part files found in {dir_path}")
        return None

    # Extract part number and sort numerically
    def get_part_num(f):
        match = re.search(r'_part(\d+)', f)
        return int(match.group(1)) if match else -1
        
    part_files.sort(key=get_part_num)
    num_files = len(part_files)
    
    print(f"Found {num_files} files. Pre-allocating and loading...")

    # Load first file to get dimensions and dtype
    first_arr = np.load(part_files[0])
    print("shape of first file", first_arr.shape)
    nbl, chunk_time, nchan, npol2 = first_arr.shape
    dtype = first_arr.dtype

    last_arr = np.load(part_files[-1])
    total_time = (num_files-1)*chunk_time + last_arr.shape[1]

    # Pre-allocate the big array
    full_array = np.empty((total_time, nchan, nbl), dtype=dtype, order='F')
    full_array[:chunk_time, :, :] = 0.5*(first_arr[:,:,:,0] + first_arr[:,:,:,3]).transpose(1,2,0)
    full_array[-last_arr.shape[1]:, :, :] = 0.5*(last_arr[:,:,:,0] + last_arr[:,:,:,3]).transpose(1,2,0)
    del first_arr, last_arr
    # Fill the array

    start=chunk_time
    for i in range(1, num_files-1):
        f = part_files[i]
        arr = np.load(f)
        end = start + arr.shape[1]
        I = 0.5*(arr[:,:,:,0] + arr[:,:,:,3])
        full_array[start:end, :, :] = I.transpose(1,2,0)
        start = end
        print(f"  Processed {os.path.basename(f)} into slice [{start}:{start+arr.shape[1]}]")
    
    print(f"\nLoading complete.")
    print(f"Resulting shape: {full_array.shape}")
    print(f"Total size: {full_array.nbytes / 1024**3:.2f} GB")

    return full_array

@nb.njit(parallel=True)
def tilled_transpose(x,bsc=32,bsr=128): 
    nr, nc = x.shape
    # preserve input layout
    if x.flags.f_contiguous and not x.flags.c_contiguous:
        # make an F-contiguous (nc, nr) array
        buf = np.empty((nr, nc), dtype=x.dtype)
        y = buf.T
    else:
        # default to C-order
        y = np.empty((nc, nr), dtype=x.dtype)
        
    br = (nr + bsr - 1)//bsr
    bc = (nc + bsc - 1)//bsc
    totblocks = br*bc
    # print("totblocks", totblocks, br, bc)
    for i in nb.prange(totblocks):
        bi=i//bc
        bj=i%bc
        # print("bi, bj", bi, bj)
        imax=min(bi*bsr+bsr,nr)
        jmax=min(bj*bsc+bsc,nc)
        # print("imax, jmax",imax, jmax)
        for ii in range(bi*bsr, imax):
            for jj in range(bj*bsc, jmax):
                # print("putting ii,jj", ii, jj)
                y[jj,ii] = x[ii,jj]
    return y

# @nb.njit(parallel=True)
# def fast_median(vis_amp):
#     #need to paralellize along time. time is faster moving, in Fortran type
#     ntime,nfreq = vis_amp.shape
#     score = np.empty(nfreq,dtype=vis_amp.dtype)
#     for i in nb.prange(nfreq):
#         score[i] = np.median(vis_amp[:,i])
#     return score

@nb.njit(parallel=True)
def mad_1d(x):
    m = np.median(x)
    d = np.empty(x.size, np.float64)
    for i in nb.prange(x.size):
        d[i] = np.abs(x[i] - m)
    return 1.4826 * np.median(d)

@nb.njit(parallel=True)
def time_mask(vis_amp, thresh):
    #need to paralellize along time. time is faster moving, in Fortran type
    ntime,nfreq = vis_amp.shape
    mask = np.empty(ntime,dtype='bool')
    score = np.empty(ntime,dtype=vis_amp.dtype)
    vis_ampT = tilled_transpose(vis_amp) 
    for i in nb.prange(ntime):
        score[i] = np.median(vis_ampT[:,i])
    mad = mad_1d(score)
    med = np.median(score)
    for i in nb.prange(ntime):
        mask[i] = np.abs(score[i]-med) > thresh * mad
    return mask, score
    
@nb.njit(parallel=True)
def freq_mask(vis_amp, thresh):
    #need to paralellize along time. time is faster moving, in Fortran type
    ntime,nfreq = vis_amp.shape
    mask = np.empty(nfreq,dtype='bool')
    score = np.empty(nfreq,dtype=vis_amp.dtype)
    for i in nb.prange(nfreq):
        score[i] = np.median(vis_amp[:,i])
    mad = mad_1d(score)
    med = np.median(score)
    for i in nb.prange(nfreq):
        mask[i] = np.abs(score[i]-med) > thresh * mad
    return mask, score


    
@nb.njit(parallel=True)
def pixel_mask(vis_amp,thresh):
    y = vis_amp.ravel()
    n = len(y)
    mymad = mad_1d(y)
    mymed = np.median(y)
    bad_pix = np.zeros(y.shape,dtype='bool')
    for i in nb.prange(n):
            bad_pix[i]= np.abs(y[i]-mymed) >  thresh*mymad
    return bad_pix.reshape(vis_amp.shape)

def read_file(path,numparts,chunklen,nchan,nant,acclen,osamp=64):
    ntime = numparts*chunklen
    acctime=acclen*16e-6*osamp
    nbl = len(np.triu_indices(nant)[0])
    arr = np.zeros((ntime,nchan,nbl),dtype='complex64',order='F')
    print("loading",ntime*acctime, "seconds")
    for part_num in range(numparts):
        t1=time.time()
        f=np.load(path+f".part{part_num:05d}.npy")
        t2=time.time()
        read=t2-t1
        t1=time.time()
        # arr[50*part_num:50*(part_num+1), :] = f[20,:,:,3]
        I = 0.5*(f[:,:,:,0] + f[:,:,:,3])
        arr[chunklen*part_num:chunklen*(part_num+1), :, :] = I.transpose(1,2,0) #bl, time, chan, pol
        t2=time.time()
        fill=t2-t1
        if part_num%10==0:
            print("read", part_num)
            print("read time", read, "fill time", fill)
    return arr

@nb.njit(parallel=True)
def _median_of_columns(x):
    """
    Returns median along axis=0, i.e. one median per column.
    Best when each column access x[:, i] is contiguous in memory.
    """
    nr, nc = x.shape
    out = np.empty(nc, dtype=x.dtype)
    for i in nb.prange(nc):
        out[i] = np.median(x[:, i])
    return out


@nb.njit(parallel=True)
def _median_of_rows(x):
    """
    Returns median along axis=1, i.e. one median per row.
    Best when each row access x[i, :] is contiguous in memory.
    """
    nr, nc = x.shape
    out = np.empty(nr, dtype=x.dtype)
    for i in nb.prange(nr):
        out[i] = np.median(x[i, :])
    return out


def fast_median(x, axis=0, bsc=32, bsr=128):
    """
    Median along axis 0 or 1, using tiled transpose when that makes
    thread access more contiguous.

    axis=0 -> output shape (nc,)
    axis=1 -> output shape (nr,)
    """
    if axis == 0:
        # Want medians of columns.
        # Good directly if columns are contiguous => Fortran layout.
        if x.flags.f_contiguous and not x.flags.c_contiguous:
            return _median_of_columns(x)
        else:
            # Transpose so original columns become rows/cols in a friendlier layout
            xt = tilled_transpose(x, bsc=bsc, bsr=bsr)
            # median of rows of xt == median of columns of x
            return _median_of_rows(xt)

    elif axis == 1:
        # Want medians of rows.
        # Good directly if rows are contiguous => C layout.
        if x.flags.c_contiguous:
            return _median_of_rows(x)
        else:
            xt = tilled_transpose(x, bsc=bsc, bsr=bsr)
            # median of columns of xt == median of rows of x
            return _median_of_columns(xt)

    else:
        raise ValueError("axis must be 0 or 1")

@nb.njit(parallel=True)
def block_average_masked(vis, mask, block_time, block_freq):
    """
    vis  : (ntime, nfreq) real or complex array
    mask : (ntime, nfreq) bool array, True means flagged/bad
    returns:
        out   : block-averaged array
        cnt   : number of unmasked samples per output bin
    """
    ntime, nfreq = vis.shape
    out_time = (ntime + block_time - 1) // block_time
    out_freq = (nfreq + block_freq - 1) // block_freq

    out = np.empty((out_time, out_freq), dtype=vis.dtype)
    cnt = np.zeros((out_time, out_freq), dtype=np.int64)

    nanval = np.nan + 1j * np.nan


    for i in nb.prange(out_time * out_freq):
        bt = i // out_freq
        bf = i % out_freq

        t0 = bt * block_time
        t1 = min(t0 + block_time, ntime)
        f0 = bf * block_freq
        f1 = min(f0 + block_freq, nfreq)

        s = 0.0 + 0.0j
        c = 0
        for t in range(t0, t1):
            for f in range(f0, f1):
                if not mask[t, f]:
                    s += vis[t, f]
                    c += 1
        cnt[bt, bf] = c
        if c > 0:
            out[bt, bf] = s / c
        else:
            out[bt, bf] = nanval

    return out, cnt


@nb.njit(parallel=True)
def block_average_masked_delay(vis, mask, freqs, delay, block_time, block_freq):
    """
    vis        : (ntime, nfreq) real or complex array
    mask       : (ntime, nfreq) bool array, True means flagged/bad
    freqs      : (nfreq, ) real array of channel frequencies in units of original channels (FFT size 4096)
    delay      : (ntime, ) or scalar delay for this baseline (for clock correction/beamforming)
    block_time : int number of time samples to average
    block_freq : int number of freq samples to average
    
    returns:
        out   : block-averaged array
        cnt   : number of unmasked samples per output bin
    """
    ntime, nfreq = vis.shape
    out_time = (ntime + block_time - 1) // block_time
    out_freq = (nfreq + block_freq - 1) // block_freq

    out = np.empty((out_time, out_freq), dtype=vis.dtype)
    cnt = np.zeros((out_time, out_freq), dtype=np.int64)

    nanval = np.nan + 1j * np.nan

    # if isinstance(delay, np.ndarray): doesnt work inside numba
    for i in nb.prange(out_time * out_freq):
        bt = i // out_freq
        bf = i % out_freq

        t0 = bt * block_time
        t1 = min(t0 + block_time, ntime)
        f0 = bf * block_freq
        f1 = min(f0 + block_freq, nfreq)

        s = 0.0 + 0.0j
        c = 0
        for f in range(f0, f1):
            for t in range(t0, t1):
                msk = 1 - mask[t,f]
                s += msk * vis[t, f] * np.exp(2j*np.pi*freqs[f]*delay[t]/4096)
                c += msk
        cnt[bt, bf] = c
        if c > 0:
            out[bt, bf] = s / c
        else:
            out[bt, bf] = nanval
    # else: #float delay
    #     for i in nb.prange(out_time * out_freq):
    #         bt = i // out_freq
    #         bf = i % out_freq
    
    #         t0 = bt * block_time
    #         t1 = min(t0 + block_time, ntime)
    #         f0 = bf * block_freq
    #         f1 = min(f0 + block_freq, nfreq)
    
    #         s = 0.0 + 0.0j
    #         c = 0
    #         for f in range(f0, f1):
    #             exp =  np.exp(2j*np.pi*freqs[f]*delay/4096)
    #             for t in range(t0, t1):
    #                 msk = 1 - mask[t,f]
    #                 s += msk * vis[t, f] * exp
    #                 c += msk
    #         cnt[bt, bf] = c
    #         if c > 0:
    #             out[bt, bf] = s / c
    #         else:
    #             out[bt, bf] = nanval
    return out, cnt

def itrs_to_enu(xyz, ref_lat, ref_lon):
    if len(xyz.shape)==1:
        xyz = xyz.reshape(-1,3)
    enu=np.zeros(xyz.shape,dtype=xyz.dtype)
    nn = xyz.shape[0]
    _lat=np.deg2rad(ref_lat)
    _lon=np.deg2rad(ref_lon)
    sin_lat = np.sin(_lat)
    cos_lat = np.cos(_lat)
    sin_lon = np.sin(_lon)
    cos_lon = np.cos(_lon)
    for i in range(nn):
        xyz_use = xyz[i]
        enu[i, 0] = -sin_lon * xyz_use[0] + cos_lon * xyz_use[1]
        enu[i, 1] = (
          - sin_lat * cos_lon * xyz_use[0]
          - sin_lat * sin_lon * xyz_use[1]
          + cos_lat * xyz_use[2]
        )
        enu[i, 2] = (
          cos_lat * cos_lon * xyz_use[0]
          + cos_lat * sin_lon * xyz_use[1]
          + sin_lat * xyz_use[2]
        )
    return enu

def get_all_bls(coords, ants = None, refonly=False, enu=True):
    #ant0 is always the refant for ENU coord sys
    if ants is None:
        ants = np.arange(len(coords.keys()))
    nant=len(ants)
    nbl=0
    bls=[]
    idx=0
    if refonly:
        for j in range(0,nant):
            coords1 = coords[0]
            coords2 = coords[ants[j]]
            xyz1=EarthLocation.from_geodetic(lat=coords1[0], lon=coords1[1], height=coords1[2]).itrs.cartesian.xyz.to_value()
            xyz2=EarthLocation.from_geodetic(lat=coords2[0], lon=coords2[1], height=coords2[2]).itrs.cartesian.xyz.to_value()
            bls.append(xyz2-xyz1)
            idx+=1
    else:
        for i in range(nant):
            for j in range(i+1,nant):
                coords1 = coords[ants[i]]
                coords2 = coords[ants[j]]
                xyz1=EarthLocation.from_geodetic(lat=coords1[0], lon=coords1[1], height=coords1[2]).itrs.cartesian.xyz.to_value()
                xyz2=EarthLocation.from_geodetic(lat=coords2[0], lon=coords2[1], height=coords2[2]).itrs.cartesian.xyz.to_value()
                bls.append(xyz2-xyz1)
                idx+=1
    bls=np.asarray(bls)
    print(bls.shape)
    if enu:
        bls=itrs_to_enu(bls,coords[0][0],coords[0][1])
    return bls

# def get_geo_delay(bl,alt,az):
#     x = 0
#     c=299792458
#     x = np.sin(az)*np.cos(alt) * bl[0] #east
#     x += np.cos(az)*np.cos(alt) * bl[1] #north
#     x += np.sin(alt) * bl[2] #up
#     # print(x)
#     return x/c

@nb.njit(parallel=True)
def geo_delay_from_enu(bls,az,alt):
    # all angles in rad
    npix = alt.shape[0]
    nbl = bls.shape[0] #each row is x, y, z
    delays = np.empty((nbl,npix),dtype='float64')
    c=299792458
    for pix in nb.prange(npix):
        cos_alt = np.cos(alt[pix])
        sin_alt = np.sin(alt[pix])
        cos_az = np.cos(az[pix])
        sin_az = np.sin(az[pix])
        #unit vectors of source in ENU
        u0 = sin_az * cos_alt #east
        u1 = cos_az * cos_alt #north
        u2 = sin_alt          #up
        for bl in range(nbl):
            ea,no,up = bls[bl]
            p = u0 * ea + u1 * no + u2 * up #dot product of source dir. and baseline
            delays[bl, pix] = p/c
    return delays

@nb.njit(parallel=True)
def geo_delay_from_itrs(bls,ha,dec,lon):
    npix = ha.shape[0]
    nbl = bls.shape[0] #each row is x, y, z
    delays = np.empty((nbl,npix),dtype='float64')
    c=299792458
    for pix in nb.prange(npix):
        gha = ha[pix] - lon
        cos_dec = np.cos(dec[pix])
        sin_dec = np.sin(dec[pix])
        cos_gha = np.cos(gha)
        sin_gha = np.sin(gha)
        # Unit vectors of source in ITRS (ECEF)
        u0 = cos_dec * cos_gha  # greenwich (Lat=0, Lon=0)
        u1 = -cos_dec * sin_gha # ~malaysia (Lat=0, Lon=90E)
        u2 = sin_dec            # north pole
        for bl in range(nbl):
            x,y,z = bls[bl] #x greenwich, japan, north pole
            p = u0 * x + u1 * y + u2 * z
            delays[bl, pix] = p/c
    return delays

@nb.njit(parallel=True)
def hadec_to_azalt(ha, dec, lat):
    """
    Convert Hour Angle and Declination to Altitude and Azimuth.
    All inputs and outputs must be in radians.
    
    Parameters:
    -----------
    ha : float or array-like
        Hour Angle(s) in radians
    dec : float or array-like
        Declination(s) in radians
    lat : float
        Observer's latitude in radians
        
    Returns:
    --------
    az : float or array-like
        Azimuth angle(s)
    alt : float or array-like
        Altitude angle(s)
    """
    az = np.empty(ha.shape, ha.dtype)
    alt = np.empty(ha.shape, ha.dtype)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    npix = len(ha)
    for pix in nb.prange(npix):
        sinha = np.sin(ha[pix])
        cosha = np.cos(ha[pix])
        sindec = np.sin(dec[pix])
        cosdec = np.cos(dec[pix])
        
        aa = sinlat * sindec + coslat * cosdec * cosha
        aa = max(-1.0, min(1.0, aa)) #clip to -1, 1
        alt[pix] = np.arcsin(aa)
        sinaz = -cosdec * sinha
        cosaz = (coslat * sindec - sinlat * cosdec * cosha)
        aa = np.arctan2(sinaz, cosaz)
        az[pix] = (aa + 2 * np.pi) % (2 * np.pi)
    return az, alt

@nb.njit(parallel=True)
def azalt_to_hadec(az, alt, lat):
    """
    Convert Altitude and Azimuth to Hour Angle and Declination.
    All inputs and outputs must be in radians.
    
    Parameters:
    -----------
    az : float or array-like
        Azimuth angle(s) in radians
    alt : float or array-like
        Altitude angle(s) in radians
    lat : float
        Observer's latitude in radians
        
    Returns:
    --------
    ha : float or array-like
        Hour Angle(s) in radians [-pi, pi]
    dec : float or array-like
        Declination(s) in radians [-pi/2, pi/2]
    """
    ha = np.empty(az.shape, az.dtype)
    dec = np.empty(az.shape, az.dtype)
    npix = len(ha)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    
    for pix in nb.prange(npix):
        sinalt = np.sin(alt[pix])
        cosalt = np.cos(alt[pix])
        sinaz = np.sin(az[pix])
        cosaz = np.cos(az[pix])
        
        aa = sinalt * sinlat + cosalt * coslat * cosaz

        # Clip to strictly [-1, 1] to avoid NaN errors from floating point inaccuracies
        aa = max(-1.0, min(1.0, aa)) #clip to -1, 1
        dec[pix] = np.arcsin(aa)
    
        y = -sinaz * cosalt
        x = sinalt * coslat - cosalt * sinlat * cosaz
        aa = np.arctan2(y, x)
        ha[pix] = aa % (2 * np.pi)
    return ha, dec


@nb.njit(parallel=True)
def get_map(data,freqs,delays,npix):
    print(data.shape, freqs.shape, delays.shape)
    map1 = np.zeros(npix, dtype=np.float32) #float32
    nfreq, nbl = data.shape
    N_vis = nfreq * nbl
    for p in nb.prange(npix):
        pixel_sum = 0.
        for f in range(nfreq):
            nu = freqs[f]
            for b in range(nbl):
                tau = delays[p, b] #delay shape is npix, nbl for CPU

                # Calculate the fringe factor for this specific visibility
                fringe = np.exp(2j * np.pi * nu * tau)
                
                # Accumulate the dot product
                pixel_sum += fringe.real * data[f, b].real - fringe.imag * data[f, b].imag
        
        # Calculate the mean and assign to the pixel map
        map1[p] = pixel_sum / N_vis
    return map1