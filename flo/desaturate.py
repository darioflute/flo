def normalizePSF(x, y, dx, dy, psf, subimage):
    """
    Normalize the PSF to the saturated star in the image
    x,y image coords of the star
    dx,dy distance of star from psf center
    psf empirical PSF
    subimage part of the image containing the star
    """
    import numpy as np
    from astropy.nddata import block_reduce
    
    # Centers of the empirical PSFs
    xpsf = np.array([4.0, 2047.5, 4091.0, 4.0, 2047.5, 4091.0, 4.0, 2047.5, 4091.0])
    ypsf = np.array([4.0, 4.0, 4.0, 2047.5, 2047.5, 2047.5, 4091.0, 4091.0, 4091.0])
    # Regions and meshgrid to compute background and normalization
    r_bkg_int , r_bkg_ext = 12, 17
    xv, yv = np.meshgrid(np.arange(35)-17, np.arange(35)-17, indexing='ij')


    # Selection of PSF positions
    dpsf = np.hypot(x-xpsf, y-ypsf)
    idpsf = np.argmin(dpsf)
    data = psf[idpsf]
    # Compute the PSF down to L2 images' resolution
    br = block_reduce(data[5-dy:-6-dy,5-dx:-6-dx], 10, func=np.sum)
    # Compute number of NaN in subimage and choose correct r_int, r_ext
    nnan = np.sum(np.isnan(subimage))
    rnan = np.sqrt(nnan/3.1415)
    if rnan < 11:
        r_int = rnan+1
    else:
        r_int = 11
    r_ext = r_int + 2
    #print('rnan ', rnan)

    # Computation of normalization and substitution
    distance = np.hypot(xv-dx*0.1,yv-dy*0.1)
    idx_int = distance <= r_int
    idx_ext = (distance >= r_int) & (distance < r_ext)
    idx_bkg = (distance >= r_bkg_int) & (distance < r_bkg_ext)        
    psfext = np.median(br[idx_ext])
    psfbkg = np.median(br[idx_bkg])
    srcext = np.nanmedian(subimage[idx_ext])
    srcbkg = np.nanmedian(subimage[idx_bkg])
    normalization = (srcext - srcbkg) / (psfext - psfbkg)
    #print(i, normalization)
    subimage[idx_int] = br[idx_int] * normalization
    return subimage

    
def desaturatePsf(L2, stars, output=None, minarea=1):
    """
    The code applies the correct PSF to the list of stars in the Roman L2 image.

    Inputs
    ------
    L2 - name of the L2 file
    stars:  list of stars (or point-like objects) which have saturated of highly non-linear PSF center
            the list is in Skycoords from astropy (frame='icrs')
    output: optional name of output file with corrected PSF. If not given the output name is the same as the input name with a "_us" at the end of the name.
    """
    import roman_datamodels as rdm
    from roman_datamodels.dqflags import pixel
    import h5py
    import numpy as np
    import os

    # Input L2 file
    with rdm.open(L2) as dm:
        image = dm.data.copy()
        dq = dm.dq.copy()
        wfidetector = dm.meta.instrument.detector
        wfioptelement = dm.meta.instrument.optical_element
        wcs = dm.meta.wcs

    # Select the correct PSF
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..',"data") # data directory in the distribution
    with h5py.File(os.path.join(path,'epsf_'+wfioptelement+'.h5'), 'r') as hdf5_file:
        psf = hdf5_file[wfidetector][:]


    # Trasform the list of stars in x, y coordinates and select stars inside image
    ra, dec = stars.ra.deg, stars.dec.deg
    x, y = wcs.backward_transform(ra, dec)
    idx = (x >= 17) & (x < 4088-17) & (y >= 17) & (y < 4088-17)
    ngaia = np.sum(idx)
    if ngaia == 0:
        print('No GAIA stars in this image')
    else:
        print(f'There are {ngaia} bright stars which will be corrected')

    if ngaia > 0:
        # Central pixel of subimage and distance of star from central pixel in 1/10 of pixel
        x, y = x[idx], y[idx]
        xx = np.round(x).astype(int)
        yy = np.round(y).astype(int)
        dx = np.floor((x - xx)*10).astype(int)
        dy = np.floor((y - yy)*10).astype(int)

        for i in range(len(x)):
            subimage = image[yy[i]-17:yy[i]+18, xx[i]-17:xx[i]+18]
            image[yy[i]-17:yy[i]+18, xx[i]-17:xx[i]+18] = normalizePSF(x[i], y[i], dx[i], dy[i], psf, subimage)

    # Now consider the clusters of uncorrected saturated pixels which are possible stars not in GAIA
    from scipy.ndimage import label, find_objects
    nan_mask = np.isnan(image) & ((dq & pixel.SATURATED.value) != 0)
    labeled_mask, num_features = label(nan_mask)
    slices = find_objects(labeled_mask)
    xx, yy = np.arange(4088), np.arange(4088)
    X, Y = np.meshgrid(xx, yy)
    n = 0
    xb, yb = [], []
    for i, sl in enumerate(slices):
        if sl is not None:
            yi, ye = sl[0].start, sl[0].stop
            xi, xe = sl[1].start, sl[1].stop
            if (xe-xi)*(ye-yi) > minarea:
                # Find barycenter of the cluster
                idnan = np.where(nan_mask[yi:ye, xi:xe] == 1)
                bX, bY = X[yi:ye, xi:xe], Y[yi:ye, xi:xe]
                yb.append(np.mean(bY[idnan]))
                xb.append(np.mean(bX[idnan]))
                n += 1

    if n > 0:
        xb, yb = np.array(xb), np.array(yb)
        idx = (xb >= 17) & (xb < 4088-17) & (yb >= 17) & (yb < 4088-17)
        if np.sum(idx) > 0:
            xb, yb = xb[idx], yb[idx]
            n = len(xb)
        else:
            n = 0

    # Central pixel of subimage and distance of star from central pixel in 1/10 of pixel
    if n > 0:
        xx = np.round(xb).astype(int)
        yy = np.round(yb).astype(int)
        dx = np.floor((xb - xx)*10).astype(int)
        dy = np.floor((yb - yy)*10).astype(int)
        print(f'PSF substitution for {n} non GAIA saturated sources ')
        for i in range(len(xb)):
            subimage = image[yy[i]-17:yy[i]+18, xx[i]-17:xx[i]+18]
            image[yy[i]-17:yy[i]+18, xx[i]-17:xx[i]+18] = normalizePSF(xb[i], yb[i], dx[i], dy[i], psf, subimage)

    if (ngaia + n) == 0:
        print('No saturated sources in this image')
        return 0
    else:
        print(f'Corrected a total of {ngaia+n} saturated sources')
        
    # Save the corrected image
    if output is None:
        infile = os.path.abspath(L2)
        path, filename = os.path.split(infile)
        filename, ext = os.path.splitext(os.path.split(infile)[1])
        output = os.path.join(path, filename+'_us'+ext)    
    with rdm.open(L2) as dm:
        dm.data = image
        dm.save(output)
