def desaturatePsf(L2, stars, output=None):
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
    import h5py
    from astropy.nddata import block_reduce
    import numpy as np
    import os

    # Input L2 file
    with rdm.open(L2) as dm:
        image = dm.data.copy()
        wfidetector = dm.meta.instrument.detector
        wfioptelement = dm.meta.instrument.optical_element
        wcs = dm.meta.wcs

    # Select the correct PSF
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..',"data") # data directory in the distribution
    with h5py.File(os.path.join(path,'epsf_'+wfioptelement+'.h5'), 'r') as hdf5_file:
        psf = hdf5_file[wfidetector][:]

    # Centers of the empirical PSFs
    xpsf = np.array([4.0, 2047.5, 4091.0, 4.0, 2047.5, 4091.0, 4.0, 2047.5, 4091.0])
    ypsf = np.array([4.0, 4.0, 4.0, 2047.5, 2047.5, 2047.5, 4091.0, 4091.0, 4091.0])

    # Regions and meshgrid to compute background and normalization
    r_int, r_ext, r_bkg_int , r_bkg_ext = 3.5,5,12,17
    xv, yv = np.meshgrid(np.arange(35)-17, np.arange(35)-17, indexing='ij')

    # Trasform the list of stars in x, y coordinates and select stars inside image
    ra, dec = stars.ra.deg, stars.dec.deg
    x, y = wcs.backward_transform(ra, dec)
    idx = (x >= 17) & (x < 4088-17) & (y >= 17) & (y < 4088-17)  
    if np.sum(idx) == 0:
        print('No stars in this image')
        exit
    else:
        print('There are ',np.sum(idx),' stars')
    x, y = x[idx], y[idx]

    # Central pixel of subimage and distance of star from central pixel in 1/10 of pixel
    xx = np.round(x).astype(int)
    yy = np.round(y).astype(int)
    dx = np.floor((x - xx)*10).astype(int)
    dy = np.floor((y - yy)*10).astype(int)

    for i in range(len(x)):
        # Selection of PSF positions
        dpsf = np.hypot(x[i]-xpsf, y[i]-ypsf)
        idpsf = np.argmin(dpsf)
        data = psf[idpsf]
        # Compute the PSF down to L2 images' resolution
        br = block_reduce(data[5-dy[i]:-6-dy[i],5-dx[i]:-6-dx[i]], 10, func=np.sum)
        # Selection of subimage
        subimage = image[yy[i]-17:yy[i]+18, xx[i]-17:xx[i]+18]

        # Computation of normalization and substitution
        distance = np.hypot(xv-dx[i]*0.1,yv-dy[i]*0.1)
        idx_int = distance <= r_int
        idx_ext = (distance >= r_int) & (distance < r_ext)
        idx_bkg = (distance >= r_bkg_int) & (distance < r_bkg_ext)        
        psfext = np.median(br[idx_ext])
        psfbkg = np.median(br[idx_bkg])
        srcext = np.nanmedian(subimage[idx_ext])
        srcbkg = np.nanmedian(subimage[idx_bkg])
        normalization = (srcext - srcbkg) / (psfext - psfbkg)
        print(normalization)
        subimage[idx_int] = br[idx_int] * normalization
        image[yy[i]-17:yy[i]+18, xx[i]-17:xx[i]+18] = subimage

    # Save the corrected image
    if output is None:
        infile = os.path.abspath(L2)
        path, filename = os.path.split(infile)
        filename, ext = os.path.splitext(os.path.split(infile)[1])
        output = os.path.join(path, filename+'_us'+ext)    
    with rdm.open(L2) as dm:
        dm.data = image
        dm.save(output)
