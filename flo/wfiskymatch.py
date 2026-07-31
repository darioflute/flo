def pixmatch(files, outfile="offsets", verbose=False, offsets=None):
    """
    Input:

      files, list of files with images and WCS (can be compressed fits)

    Output:

      offsets, array of offsets
    
    """
    from scipy.sparse import csc_array
    from scipy.sparse.linalg import spsolve
    import h5py
    import numpy as np
    import os
    from astropy.stats import biweight_location
    
    row = []
    col = []
    data = []
    nfiles = len(files)
    B = np.zeros(nfiles)
    D = np.zeros(nfiles)

    # Rough overlap using the coverage extension
    coverage = []
    for file in files:
        with h5py.File(file, 'r') as hdf5_file:
            cov = hdf5_file['hpcoverage'][:]
            coverage.append(cov)
    xcorr = np.zeros((nfiles,nfiles))
    for i in range(nfiles-1):
        a = coverage[i]
        for j in range(i+1, nfiles):
            b = coverage[j]
            common, aidx, bidx = np.intersect1d(a, b, return_indices=True)
            if len(common) > 0:
                xcorr[i,j] = 1
                xcorr[j,i] = 1
    if verbose:
        print('There is a total of {0:n} overlaps'.format(int(np.sum(xcorr))))
    
    # Detailed computation for xcorr==1 only
    for i in range(nfiles-1):
        if verbose:
            print('.', end='')
        with h5py.File(files[i], 'r') as hdf5_file:
            hpimage = hdf5_file['hpimage'][:]
            apix = hpimage['pixel']
            aval = hpimage['value']
            aunc2 = hpimage['unc']**2
        if offsets is not None:
            aval += offsets[i]
        for j in range(i+1, nfiles):
            if xcorr[i,j] == 1:
                with h5py.File(files[j], 'r') as hdf5_file:
                    hpimage = hdf5_file['hpimage'][:]
                    bpix = hpimage['pixel']
                    bval = hpimage['value']
                    bunc2 = hpimage['unc']**2
                    common_elements, aidx, bidx = np.intersect1d(apix, bpix, return_indices=True)
                if offsets is not None:
                    bval += offsets[j]
                if len(common_elements) > 0:
                    a_ij = - np.sum(1/(aunc2[aidx] + bunc2[bidx]))
                    I_ij = - np.sum((aval[aidx]-bval[bidx])/(aunc2[aidx] + bunc2[bidx]))
                    col.extend([i,j])
                    row.extend([j,i])
                    data.extend([a_ij, a_ij])
                    B[i] += I_ij
                    B[j] -= I_ij
                    D[i] -= a_ij
                    D[j] -= a_ij
    
    ii = np.arange(nfiles)
    row.extend(ii)
    col.extend(ii)
    data.extend(D)
    A = csc_array((data, (row, col)), shape=(nfiles,nfiles))

    epsilon = spsolve(A, B)
    #delta = - np.nanmedian(epsilon)
    delta = - biweight_location(epsilon)
    epsilon += delta

    # Save offsets with file names
    import h5py
    if outfile.endswith('.h5'):
        pass
    else:
        outfile += '.h5'

    filenames = []
    for file in files:
        filepath, filebase = os.path.split(file)
        filename = filebase.split('.')[0]
        filenames.append(filename)
        
    with h5py.File(outfile, 'w') as hdf5_file:
        d = hdf5_file.create_dataset('offsets', (len(epsilon),), dtype='float32') 
        d[:] = epsilon
        ds = hdf5_file.create_dataset('files', shape=len(files), dtype=h5py.string_dtype(), compression="gzip")
        ds[:] = filenames
        
    return epsilon


def maskSources(data, err=None, dq=None, block=50, fwhm=1.0, areathreshold=1000, eccthreshold=0.8, nsigma=0.7):

    import numpy as np

    # Define background
    from photutils.background import Background2D, MedianBackground
    data0 = data.copy()
    bkg_estimator = MedianBackground()
    bkg = Background2D(data0, (block,block), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data0 -= bkg.background

    # Convolve with Gaussian kernel
    from astropy.convolution import convolve
    from photutils.segmentation import make_2dgaussian_kernel
    kernel = make_2dgaussian_kernel(fwhm, size=5)
    convolved_data = convolve(data0, kernel)

    # Detect sources and create segmentation map
    from photutils.segmentation import detect_sources, SourceCatalog
    if err is None:
        threshold = nsigma * bkg.background_rms
    else:
        print('Using error to compute threshold')
        threshold = nsigma * err

    print('Median bkg rms ', np.nanmedian(bkg.background_rms))
    if err is not None:
        print('Median err     ', np.nanmedian(err))
    segment_map = detect_sources(convolved_data, threshold, npixels=10)
    cat = SourceCatalog(data0, segment_map, convolved_data=convolved_data)

    # Select sources to be masked
    ecc = (cat.eccentricity.value < eccthreshold) | (cat.area.value < areathreshold)
    roundsources = cat.label[ecc]
    mask = np.isin(segment_map.data, roundsources)

    # Mask bad pixels if dq is available
    if dq is not None:
        from roman_datamodels.dqflags import pixel
        idx = dq == pixel.GOOD.value
        mask[~idx] = True

    # Return mask
    return mask


def asdf2healpix(infile, outdir,nsparse=17,ncoverage=11):
    """
    Reproject an ASDF WFI image to a Healpix tessellation with 3" pixels

    input:
       infile, 'name of asdf file'
       outdir, 'name of output directory'
       nsparse, 2**nsparse gives the resolution of the Healpix grid
       nsigma,  number of sigma to mask bright point sources

    The code creates files with list of pixels, values, and uncertainty in hdf5 format
    """
    import os
    import numpy as np
    import time
    # 0. Extract filename [assumed to be the part before the first dot from the left]
    filepath, filebase = os.path.split(infile)
    filename = filebase.split('.')[0]
    # 1. Read the ASDF file
    import roman_datamodels as rd
    from roman_datamodels.dqflags import pixel

    start_time = time.time()
    with rd.open(infile) as dm:
        data = dm.data.copy()
        err = dm.err.copy()
        wcs = dm.meta.wcs
        dq = dm.dq.copy()
    print('file ', filename,' read in ', time.time() - start_time)
        
    start_time = time.time()
    # 2. Mask point sources and bad pixels
    mask = maskSources(data, dq=dq, fwhm=1, areathreshold=1000)    
    # 3. Image and errors are block reduced by a factor nblock
    from astropy.nddata import block_reduce
    data[mask] = np.nan
    nblock = 7*2
    data = block_reduce(data, nblock, func=np.nanmean)
    err = block_reduce(err, nblock, func=np.nanmean)
    # Impainting
    from maskfill import maskfill
    idx = ~np.isfinite(data)
    if np.sum(idx) > 0:
        mask1 = np.zeros(np.shape(data))
        mask1[idx] = 1
        data,_ = maskfill(data, mask1, operator='median' )
    # 4. Healsparse map
    from healsparse import healSparseMap as hspm
    import healpy as hp
    nside_coverage = 2**ncoverage  # 1.7' x 1.7' tiles
    nside_sparse = 2**nsparse    # 1.6" x 1.6" pixels
    hsp_map = hspm.HealSparseMap.make_empty(nside_coverage, nside_sparse, dtype=np.float64)
    # List of Healpix pixels 
    ny, nx = np.shape(data)
    x, y = np.arange(nx)*nblock+(nblock-1)*0.5, np.arange(ny)*nblock+(nblock-1)*0.5
    xx, yy = np.meshgrid(x, y)
    ra, dec = wcs(xx, yy)
    px_num = hp.ang2pix(hsp_map._nside_sparse, ra, dec, nest=True, lonlat=True)
    upx = np.unique(px_num)
    # 5. Interpolation and update
    from scipy.ndimage import map_coordinates
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    coord_system_out = 'icrs'
    radec = hp.pix2ang(hsp_map._nside_sparse, upx, nest=True)
    lon_out = radec[1]*180/np.pi
    lat_out = 90-radec[0]*180/np.pi
    world_out = SkyCoord(lon_out*u.degree, lat_out*u.degree, frame=coord_system_out)
    # Look up pixels in input WCS
    xinds, yinds = wcs.world_to_pixel(world_out)
    xinds = xinds/nblock - (nblock-1) * 0.5
    yinds = yinds/nblock - (nblock-1) * 0.5
    coords = np.array([yinds, xinds])
    ## order of the spline interpolation: 1 is bilinear (does not create noticeable border effects)
    values = map_coordinates(data, coords, output=None, order = 1, cval=np.nan)
    errvalues = map_coordinates(err, coords, output=None, order = 1, cval=np.nan)
    # Update only possible with float64 !
    hsp_map.update_values_pix(upx, values.astype(np.float64))
    # 6. Save pixels with finite values
    import h5py
    idx = np.isfinite(values)
    upx = upx[idx]
    values = values[idx]
    sigmas = errvalues[idx]
    hpimage = np.rec.array([upx,values,sigmas],
                      formats='int64,float32,float32',
                      names='pixel, value, unc')
    idcoverage = np.where(hsp_map.coverage_map>0)[0]  # Select covered tiles
    print('saving in '+os.path.join(outdir,filename+'.h5')+' time: ', time.time() - start_time)
    with h5py.File(os.path.join(outdir,filename+'.h5'), 'w') as hdf5_file:
        hdf5_file.create_dataset('hpimage', data=hpimage, compression="gzip")
        d = hdf5_file.create_dataset('hpcoverage', (len(idcoverage),), dtype='int64') 
        d[:] = idcoverage


def computeOverlaps(files):
    """
    Given a list of h5 files computes the overlaps between files and save them in a sparse array matrix (CSC)
    """    
    import h5py
    import numpy as np
    from scipy.sparse import csc_array

    nfiles = len(files)
    coverage = []
    for file in files:
        with h5py.File(file, 'r') as hdf5_file:
            cov = hdf5_file['hpcoverage'][:]
            coverage.append(cov)

    row = []
    col = []
    data = []
    for i in range(nfiles-1):
        a = coverage[i]
        for j in range(i+1, nfiles):
            b = coverage[j]
            common, aidx, bidx = np.intersect1d(a, b, return_indices=True)
            if len(common) > 0:
                col.extend([i,j])
                row.extend([j,i])
                data.extend([1,1])

    overlap = csc_array((data, (row, col)), shape=(nfiles,nfiles))
    return overlap

def addOffset(L2, offset=0, output=None):
    """
    Add an offset to the data of a L2 WFI file
    """
    import roman_datamodels as rdm
    import os
    
    if output is None:
        infile = os.path.abspath(L2)
        path, filename = os.path.split(infile)
        filename, ext = os.path.splitext(os.path.split(infile)[1])
        output = os.path.join(path, filename[:-5]+'_off_cal'+ext)    

    with rdm.open(L2) as dm:
        dm.data += offset
        dm.save(output)

    return 1
