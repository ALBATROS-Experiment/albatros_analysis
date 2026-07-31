import numpy as np 
import numba as nb
from astropy.coordinates import EarthLocation, SkyCoord, AltAz

@nb.njit(parallel=True)
def get_map(data,freqs,delays,npix):
    '''
    Get dirty map on CPU for single time given some visibility data.

    Parameters
    ----------
    data : np.ndarray shape (nfreq, nbl)
        Visibility Data
    freqs : np.ndarry shape (nfreqs)
        Frequencies
    delays : np.ndarray shape (npix, nbl)
        Delays onto each pixel for each bline
    npix : int
        Number of pixels
    
    Returns
    ------
    np.ndarray shape (npix)
        Dirty map, intensity value for each pixel
    '''
    print(data.shape, freqs.shape, delays.shape)
    map = np.zeros(npix, dtype=np.complex128)
    nfreq, nbl = data.shape
    N_vis = nfreq * nbl
    for p in nb.prange(npix):
        pixel_sum = 0j
        for f in range(nfreq):
            nu = freqs[f]
            for b in range(nbl):
                tau = delays[p, b] #delay shape is npix, nbl for CPU

                # Calculate the fringe factor for this specific visibility
                fringe = np.exp(-2j * np.pi * nu * tau)
                
                # Accumulate the dot product
                pixel_sum += fringe * data[f, b]
        
        # Calculate the mean and assign to the pixel map
        map[p] = pixel_sum / N_vis
    return map


@nb.njit(parallel=True)
def geo_delay_from_enu(bls, az, alt, angle='deg'):
    """
    Calculate geometric delays for ENU baselines and sky positions.
    Sky positions in az/alt (local frame), accepts both radians and degrees.

    Parameters
    ----------
    bls : np.ndarray, shape (nbl, 3)
        Baseline vectors in ENU coordinates: east, north, up.
    az : np.ndarray, shape (npix,)
        Azimuth of each sky position.
    alt : np.ndarray, shape (npix,)
        Altitude of each sky position.
    angle : {'deg', 'rad'}, optional
        Units of ``az`` and ``alt``. Default is ``'deg'``.

    Returns
    -------
    np.ndarray, shape (nbl, npix)
        Geometric delays in seconds.
    """
    # check the angle type is valid
    assert angle in ['deg', 'rad']
    if angle == 'deg':
        alt = np.pi*alt/180
        az = np.pi*az/180
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
def geo_delay_from_itrs(bls, ha, dec, lon):
    """
    Calculate geometric delays for ITRS and sky position. 
    Sky positions in dec/ha (celestial frame), only accepts radians for now.
    """
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
    '''
    Construct all baselines in ENU/ITRS given coordinates in lat/lon/alt.
    '''
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