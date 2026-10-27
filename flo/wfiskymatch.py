def wfipixmatch(files, outfile="offsets", verbose=False, minoverlap=5):
    """
    Input:

         files, list of background files

    Output:

         offsets, list of offsets to apply to the original images
    """
    from scipy.sparse import csc_array
    from scipy.sparse.linalg import spsolve
    import h5py
    import numpy as np
    import os
    from astropy.stats import biweight_location, biweight_scale

    # Find groups of detectors (wfi01-wfi18)
    # A) strip name of file
    filenames = []
    for file in files:
        directory, filename = os.path.split(file)
        filenames.append(filename[:25])
    # B) find 18 detectors groups
    filegroups = []
    ufiles = np.unique(filenames)    
    for ufile in ufiles:
        matches = [item for item in files if ufile in item]
        filegroups.append(matches)
    
    row = []
    col = []
    data = []
    ngroups = len(filegroups)
    B = np.zeros(ngroups)
    D = np.zeros(ngroups)

    # Rough overlap using the coverage extension
    coverage = []
    for filegroup in filegroups:
        covgroup = []
        for file in filegroup:
            with h5py.File(file, 'r') as hdf5_file:
                cov = hdf5_file['hpcoverage'][:]
                covgroup.extend(cov)
        coverage.append(covgroup)
        
    xcorr = np.zeros((ngroups,ngroups))
    for i in range(ngroups-1):
        a = coverage[i]
        for j in range(i+1, ngroups):
            b = coverage[j]
            common, aidx, bidx = np.intersect1d(a, b, return_indices=True)
            if len(common) > 0:
                xcorr[i,j] = 1
                xcorr[j,i] = 1
    if verbose:
        print('There is a total of {0:n} overlaps'.format(int(np.sum(xcorr))))
     

    # Detailed computation for xcorr==1 only
    for i in range(ngroups-1):
        if verbose:
            print('.', end='')
        for file in filegroups[i]:
            apix = []
            aval = []
            aunc2 = []
            with h5py.File(file, 'r') as hdf5_file:
                hpimage = hdf5_file['hpimage'][:]
                apix.extend(hpimage['pixel'])
                aval.extend(hpimage['value'])
                aunc2.extend(hpimage['unc']**2)
            ncov = len(apix)
            apix = np.array(apix)
            aval = np.array(aval)
            aunc2 = np.array(aunc2)
            
        for j in range(i+1, ngroups):
            if xcorr[i,j] == 1:
                bpix = []
                bval = []
                bunc2 = []
                for file in filegroups[j]:
                    with h5py.File(file, 'r') as hdf5_file:
                        hpimage = hdf5_file['hpimage'][:]
                        bpix.extend(hpimage['pixel'])
                        bval.extend(hpimage['value'])
                        bunc2.extend(hpimage['unc']**2)
                bpix = np.array(bpix)
                bval = np.array(bval)
                bunc2 = np.array(bunc2)
                common_elements, aidx, bidx = np.intersect1d(apix, bpix, return_indices=True)
                if (len(common_elements) > 0):
                    if verbose:
                        print(i, j, len(common_elements))
                    a_ij = - np.mean(1/(aunc2[aidx] + bunc2[bidx]))
                    I_ij = - np.mean((aval[aidx]-bval[bidx])/(aunc2[aidx] + bunc2[bidx]))
                    col.extend([i,j])
                    row.extend([j,i])
                    data.extend([a_ij, a_ij])
                    B[i] += I_ij
                    B[j] -= I_ij
                    D[i] -= a_ij
                    D[j] -= a_ij
    
    ii = np.arange(ngroups)
    row.extend(ii)
    col.extend(ii)
    data.extend(D)
    A = csc_array((data, (row, col)), shape=(ngroups,ngroups))
    # Check if array is connected
    from scipy.sparse.csgraph import connected_components 
    n_components, labels = connected_components(csgraph=A, directed=True, return_labels=True)
    if n_components > 1:
        print('The images are separated in ', n_components,' components')
        print('Separate the images in different groups before matching them')
        return 0

    else:
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

        # Associate epsilon to the input files

        ufiles = list(ufiles)
        fnames = []
        fepsilon = []
        for filename, file in zip(filenames, files):
            idx = ufiles.index(filename)
            filepath, filebase = os.path.split(file)
            fnames.append(filebase.split('.')[0])
            fepsilon.append(epsilon[idx])

        #fepsilon = np.array(fepsilon)
        #fnames = np.array(fnames)
        
        with h5py.File(outfile, 'w') as hdf5_file:
            d = hdf5_file.create_dataset('offsets', shape=len(fepsilon), dtype='float32') 
            d[:] = fepsilon
            ds = hdf5_file.create_dataset('files', shape=len(files), dtype=h5py.string_dtype(), compression="gzip")
            ds[:] = fnames
        
        return np.array(fepsilon)


        
def pixmatch(files, outfile="offsets", verbose=False, offsets=None, minoverlap=5):
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
    from astropy.stats import biweight_location, biweight_scale
    
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
            #vloc, vscale = biweight_location(aval), biweight_scale(aval)
            #sloc, sscale = biweight_location(aunc2), biweight_scale(aunc2)
            #idx = (aval > vloc - 3*vscale) & (aval < vloc + 10*vscale) & (aunc2 > sloc - 3*sscale) & (aunc2 < sloc+5*sscale)
            #apix = apix[idx]
            #aval = aval[idx]
            #aunc2 = aunc2[idx]
            ncov = len(apix)
            
        if offsets is not None:
            aval += offsets[i]
        for j in range(i+1, nfiles):
            if xcorr[i,j] == 1:
                with h5py.File(files[j], 'r') as hdf5_file:
                    hpimage = hdf5_file['hpimage'][:]
                    bpix = hpimage['pixel']
                    bval = hpimage['value']
                    bunc2 = hpimage['unc']**2
                    #vloc, vscale = biweight_location(bval), biweight_scale(bval)
                    #sloc, sscale = biweight_location(bunc2), biweight_scale(bunc2)
                    #idx = (bval > vloc - 3*vscale) & (bval < vloc + 10*vscale) & (bunc2 > sloc - 3*sscale) & (bunc2 < sloc+5*sscale)
                    #bpix = bpix[idx]
                    #bval = bval[idx]
                    #bunc2 = bunc2[idx]
                    common_elements, aidx, bidx = np.intersect1d(apix, bpix, return_indices=True)
                if offsets is not None:
                    bval += offsets[j]
                if (len(common_elements) > 0):
                    if verbose:
                        print(i, j, len(common_elements))
                    #a_ij = - np.sum(1/(aunc2[aidx] + bunc2[bidx]))
                    #I_ij = - np.sum((aval[aidx]-bval[bidx])/(aunc2[aidx] + bunc2[bidx]))
                    a_ij = - np.mean(1/(aunc2[aidx] + bunc2[bidx]))
                    I_ij = - np.mean((aval[aidx]-bval[bidx])/(aunc2[aidx] + bunc2[bidx]))
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

    # Check if array is connected
    from scipy.sparse.csgraph import connected_components 
    n_components, labels = connected_components(csgraph=A, directed=True, return_labels=True)
    if n_components > 1:
        print('The images are separated in ', n_components,' components')
        print('Separate the images in different groups before matching them')
        return 0

    else:
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
            d = hdf5_file.create_dataset('offsets', shape=len(epsilon), dtype='float32') 
            d[:] = epsilon
            ds = hdf5_file.create_dataset('files', shape=len(files), dtype=h5py.string_dtype(), compression="gzip")
            ds[:] = filenames
        
        return epsilon


def pixmatch2(files, outfile="offsets", verbose=False, offsets=None, minoverlap=5):
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
    from astropy.stats import biweight_location, biweight_scale
    
    nfiles = len(files)
    B = np.zeros(nfiles)
    D = np.zeros(nfiles)
    a_ij = np.zeros((nfiles, nfiles))
    b_ij = np.zeros((nfiles, nfiles))

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
            vloc, vscale = biweight_location(aval), biweight_scale(aval)
            sloc, sscale = biweight_location(aunc2), biweight_scale(aunc2)
            idx = (aval > vloc - 3*vscale) & (aval < vloc + 5*vscale) & (aunc2 > sloc - 3*sscale) & (aunc2 < sloc+5*sscale)
            idx = np.isfinite(aval) & np.isfinite(aunc2)
            apix = apix[idx]
            aval = aval[idx]
            aunc2 = aunc2[idx]
            
        if offsets is not None:
            aval += offsets[i]
        for j in range(i+1, nfiles):
            if xcorr[i,j] == 1:
                with h5py.File(files[j], 'r') as hdf5_file:
                    hpimage = hdf5_file['hpimage'][:]
                    bpix = hpimage['pixel']
                    bval = hpimage['value']
                    bunc2 = hpimage['unc']**2
                    vloc, vscale = biweight_location(bval), biweight_scale(bval)
                    sloc, sscale = biweight_location(bunc2), biweight_scale(bunc2)
                    idx = (bval > vloc - 3*vscale) & (bval < vloc + 5*vscale) & (bunc2 > sloc - 3*sscale) & (bunc2 < sloc+5*sscale)
                    idx = np.isfinite(bval) & np.isfinite(bunc2)
                    bpix = bpix[idx]
                    bval = bval[idx]
                    bunc2 = bunc2[idx]
                    common_elements, aidx, bidx = np.intersect1d(apix, bpix, return_indices=True)
                if offsets is not None:
                    bval += offsets[j]
                if len(common_elements) > minoverlap:
                    if verbose:
                        print(i, j, len(common_elements))
                    a_ij[i,j] = np.sum(1/(aunc2[aidx] + bunc2[bidx]))
                    b_ij[i,j] = np.sum((aval[aidx]-bval[bidx])/(aunc2[aidx] + bunc2[bidx]))
                    a_ij[j,i] = a_ij[i,j]
                    b_ij[j,i] = - b_ij[i,j]

    row = []
    col = []
    data = []
    for i in range(nfiles):
        B[i] = -np.sum(b_ij[i,:])
        row.append(i)
        col.append(i)
        data.append(np.sum(a_ij[i,:]))
        for j in range(nfiles):
            if a_ij[i,j] > 0:
                data.append(-a_ij[i,j])
                row.append(i)
                col.append(j)

    A = csc_array((data, (row, col)), shape=(nfiles,nfiles))

    idx = np.isfinite(data)
    print('A ', np.sum(~idx))
    idx = np.isfinite(B)
    print('B ', np.sum(~idx))

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


def maskSources(data, err=None, dq=None, block=50, fwhm=1.0, areathreshold=1000, eccthreshold=0.5, nsigma=0.7):

    import numpy as np

    # Define background
    from photutils.background import Background2D, MedianBackground
    data0 = data.copy()

    # Inpaint saturated pixels
    #from roman_datamodels.dqflags import pixel    
    #from maskfill import maskfill
    #idx = (dq & pixel.DO_NOT_USE.value) != 0
    #mask1 = np.zeros(np.shape(data))
    #mask1[idx] = 1
    #data0,_ = maskfill(data0, mask1, operator='median' )
    
    # Mask pixels with "do not use" or "no lin correction" if dq available
    if dq is not None:
        from roman_datamodels.dqflags import pixel
        idx = (dq & pixel.DO_NOT_USE.value) != 0
        data0[idx] = np.nan
        idx = (dq & pixel.NO_LIN_CORR.value) != 0
        data0[idx] = np.nan
    
    
    bkg_estimator = MedianBackground()
    bkg = Background2D(data0, (block,block), filter_size=(3,3), bkg_estimator=bkg_estimator,exclude_percentile=12.0)
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
    segment_map = detect_sources(convolved_data, threshold, n_pixels=10)
    cat = SourceCatalog(data0, segment_map, convolved_data=convolved_data)

    # Select sources to be masked
    ecc = (cat.eccentricity.value < eccthreshold) | (cat.area.value < areathreshold)
    roundsources = cat.label[ecc]
    mask = np.isin(segment_map.data, roundsources)

    # Mask negative values (below 3*sigma under the background)
    idx = data0 < -3 * bkg.background_rms
    mask[idx] = True
    
    # Mask bad pixels marked as "Do Not Use" or "No Lin Corr" if dq is available
    if dq is not None:
        from roman_datamodels.dqflags import pixel
        idx = (dq & pixel.DO_NOT_USE.value) != 0
        mask[idx] = True
        idx = (dq & pixel.NO_LIN_CORR.value) != 0
        mask[idx] = True

    # Search for clusters of uncorrected saturated pixels which are possible stars not in GAIA bright stars
    from scipy.ndimage import label, find_objects
    nan_mask = np.isnan(data) & ((dq & pixel.SATURATED.value) != 0)
    labeled_mask, num_features = label(nan_mask)
    slices = find_objects(labeled_mask)
    Y, X = np.ogrid[:4088, :4088]
    nb = 14*3
    for sl in slices:
        if sl is not None:
            yi, ye = sl[0].start, sl[0].stop
            xi, xe = sl[1].start, sl[1].stop
            if (xe-xi)*(ye-yi) > 1: # minimal area considered including 
                # Find barycenter of the cluster
                condition = nan_mask[yi:ye, xi:xe] == 1
                nnan = np.sum(np.where(condition))
                if nnan > 1:
                    idnan = np.argwhere(condition)
                    centroid = np.mean(idnan, axis=0)
                    Yc, Xc = yi+centroid[0], xi+centroid[1]
                    #print(np.sqrt(nnan/3.1415))
                    #rb  = nnan/3.1415 * nb**2 # nb x radius
                    rb = nb**2
                    cmask = (X - Xc)**2 + (Y - Yc)**2 <= rb
                    mask[cmask] = True
        
    # Return mask
    return mask


def asdf2healpix(infile, outdir,nsparse=17,ncoverage=11,inpainting=False,savefits=False):
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
    import crds
    
    # 0. Extract filename [assumed to be the part before the first dot from the left]
    filepath, filebase = os.path.split(infile)
    filename = filebase.split('.')[0]
    # 1. Read the ASDF file
    import roman_datamodels as rdm
    from roman_datamodels.dqflags import pixel

    start_time = time.time()
    with rdm.open(infile) as dm:
        data = dm.data.copy()
        err = dm.err.copy()
        wcs = dm.meta.wcs
        dq = dm.dq.copy()
        meta = dm.meta.copy()
        
    dn2mjy = meta.photometry.conversion_megajanskys
    print('file ', filename,' read in ', time.time() - start_time)
    
    #crds_result = crds.getreferences({'ROMAN.META.INSTRUMENT.DETECTOR': meta.instrument.detector,
    #                              'ROMAN.META.INSTRUMENT.NAME': meta.instrument.name,
    #                              'ROMAN.META.EXPOSURE.START_TIME': meta.exposure.start_time.isot}, 
    #                             observatory='roman', reftypes=['area'], ignore_cache=False)
    # Consider the different area of pixels on the sky
    # See: https://github.com/spacetelescope/roman_notebooks/blob/main/notebooks/aperture_photometry/aperture_photometry.ipynb
    #with rdm.open(crds_result['area']) as pam:
    #    pamdata = pam.data.copy()
    #data /= pamdata
    #err /= pamdata

    # Convert data to MJy/sr
    data *= dn2mjy
    err *= dn2mjy
    start_time = time.time()
    # 2. Mask point sources and bad pixels
    mask = maskSources(data, dq=dq, fwhm=1, areathreshold=1000)
    # 3. Image and errors are block reduced by a factor nblock
    from astropy.nddata import block_reduce
    data[mask] = np.nan
    nblock = 7*2
    data = block_reduce(data, nblock, func=np.nanmedian) # nanmedian is more robust than nanmean ...
    err = block_reduce(err, nblock, func=np.nanmedian)
    # Impainting
    if inpainting:
        from maskfill import maskfill
        idx = ~np.isfinite(data)
        if np.sum(idx) > 0:
            mask1 = np.zeros(np.shape(data))
            mask1[idx] = 1
            data,_ = maskfill(data, mask1, operator='median' )
    # Save background in fits file
    if savefits:
        from astropy.io import fits
        outfits =  os.path.join(outdir,filename+'.fits')
        hdu = fits.PrimaryHDU(data=data)
        hdu.writeto(outfits, overwrite=True)
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
    values = map_coordinates(data.astype(np.float64), coords, output=None, order = 1, cval=np.nan)
    errvalues = map_coordinates(err.astype(np.float64), coords, output=None, order = 1, cval=np.nan)
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

def addOffset(L2, offset=0, output=None, outdir=None, verbose=False):
    """
    Add an offset to the data of a L2 WFI file

    L2:      original L2 product
    output:  name of output file
    outdir:  name of output directory
    """
    import roman_datamodels as rdm
    import os
    import crds
    
    if output is None:
        infile = os.path.abspath(L2)
        path, filename = os.path.split(infile)
        filename, ext = os.path.splitext(os.path.split(infile)[1])
        if outdir is None:
            output = os.path.join(path, filename[:-4]+'_off_cal'+ext)
        else:
            output = os.path.join(outdir, filename[:-4]+'_off_cal'+ext)
            if verbose:
                print('output ', output)

    with rdm.open(L2) as dm:
        meta = dm.meta
        dn2mjy = meta.photometry.conversion_megajanskys
        
        #crds_result = crds.getreferences({'ROMAN.META.INSTRUMENT.DETECTOR': meta.instrument.detector,
        #                              'ROMAN.META.INSTRUMENT.NAME': meta.instrument.name,
        #                              'ROMAN.META.EXPOSURE.START_TIME': meta.exposure.start_time.isot}, 
        #                             observatory='roman', reftypes=['area'], ignore_cache=False)
        # Consider the different area of pixels on the sky
        # See: https://github.com/spacetelescope/roman_notebooks/blob/main/notebooks/aperture_photometry/aperture_photometry.ipynb
        #with rdm.open(crds_result['area']) as pam:
        #    pamdata = pam.data.copy()
        offsets = offset / dn2mjy #* pamdata
        dm.data += offsets  # Add offsets in DN/s
        #dm.data /= pamdata
        #dm.err /= pamdata
        dm.save(output)

    return 1
