#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:08:06 2026

@author: dfadda
"""

def maskAllSources(data, sca='01', err=None, dq=None, block=50, fwhm=1.0, areathreshold=1000, 
                   eccthreshold=0.5, nsigma=0.7, nexpand=30, nellipse=2, nsources=2):

    import numpy as np

    # Define background
    from photutils.background import Background2D, MedianBackground
    data0 = data.copy()

    
    # Mask pixels with "do not use" or "no lin correction" if dq available
    if dq is not None:
        from roman_datamodels.dqflags import pixel
        idx = (dq & pixel.DO_NOT_USE.value) != 0
        data0[idx] = np.nan
    
    # Add circles of bad pixels
    Y, X = np.ogrid[:4088, :4088]
    circles = []
    if sca == '03':
        circles = [(270,1663,23), (864, 2047,35), (3404,2369,19)]
    elif sca == '05':
        circles = [(2678,2896,23), (2771,4033,35), (2051,3771,66)]
    elif sca == '08':
        #circles = [(353,1766,16), (2134, 4027, 145)]
        circles = [(353,1766,16), (2134, 4027, 16)]
    elif sca == '10':
            circles = [(2954,334,13)]
    elif sca == '13':
        circles = [(3420,2255, 46)]
    elif sca == '14':
        circles = [(963, 1526, 36)]
    elif sca == '18':
        circles = [(1291,3231,21)]
    
    if len(circles) > 0:
        for circle in circles:
            Yc, Xc, r = circle[0], circle[1], circle[2]
            cmask = (X - Xc)**2 + (Y - Yc)**2 <= r**2
            data0[cmask] = np.nan

    
    bkg_estimator = MedianBackground()
    bkg = Background2D(data0, (block,block), filter_size=(3,3), 
                       bkg_estimator=bkg_estimator,exclude_percentile=12.0)
    data0 -= bkg.background

    # Convolve with Gaussian kernel
    from astropy.convolution import convolve
    from photutils.segmentation import make_2dgaussian_kernel
    
    
    kernel = make_2dgaussian_kernel(fwhm, size=5)
    convolved_data = convolve(data0, kernel)

    # Detect sources and create segmentation map
    from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
    if err is None:
        threshold = nsigma * bkg.background_rms
    else:
        print('Using error to compute threshold')
        threshold = nsigma * err

    segment_map = detect_sources(convolved_data, threshold, n_pixels=30)
    segm_deblended = deblend_sources(
    convolved_data,
    segment_map,
    npixels=10,
    n_levels=32,
    contrast=0.001,
    mode="exponential",
    )   
    
    
    cat = SourceCatalog(data0, segm_deblended, convolved_data=convolved_data)

    # Select sources to be masked
    ecc = (cat.eccentricity.value < eccthreshold) | (cat.area.value < areathreshold)
    roundsources = cat.label[ecc]
    mask = np.isin(segment_map.data, roundsources)

    # Expand brighter sources
    from scipy.ndimage import binary_dilation
    idbig = cat.area.value > 1000
    bigsources = cat.label[idbig]
    maskbig = np.isin(segment_map.data, bigsources)
    structure = np.ones((3,3), dtype=bool)
    dilated_mask = binary_dilation(maskbig, structure=structure, iterations=3)
    mask |= dilated_mask
    # Sort areas to mask biggest sources
    areas = cat.area.value
    sareas = np.sort(areas)
    idbig = cat.area.value > sareas[-100]
    bigsources = cat.label[idbig]
    maskbig = np.isin(segment_map.data, bigsources)
    structure = np.ones((3,3), dtype=bool)
    dilated_mask = binary_dilation(maskbig, structure=structure, iterations=10)
    mask |= dilated_mask
    
    idbig = cat.area.value > sareas[-30]
    bigsources = cat.label[idbig]
    maskbig = np.isin(segment_map.data, bigsources)
    dilated_mask = binary_dilation(maskbig, structure=structure, iterations=nexpand)
    mask |= dilated_mask
    
    #print('Areas :', sareas[-100], sareas[-30], sareas[-2])
    from photutils.aperture import EllipticalAperture
   
    idbig = (cat.area.value > sareas[-30]) & (cat.area.value < 20000)
    catbig = cat[idbig]
    for c in catbig:
        e = c.elongation.value
        if e > 2:
            e = 2
        o = c.orientation.value
        r = c.equivalent_radius.value
        a = 3 * np.sqrt(e) * r
        b = 3 * r / np.sqrt(e)
        #print(e,o,r,a,b)
        # Define center (x, y), semi-major axis (a), semi-minor axis (b), and angle (radians)
        aperture = EllipticalAperture((c.x_centroid, c.y_centroid), a=a, b=b, theta=np.radians(o))
        # Generate boolean or sub-pixel mask for a given image shape
        emask = aperture.to_mask(method='center').to_image((4088,4088))
        mask[emask > 0] = True
    
    
    maxarea = max([50000,sareas[-nsources]])
    idbig = (cat.area.value >= 20000) & (cat.area.value < maxarea)
    catbig = cat[idbig]
    # Expand elliptical masks
    for c in catbig:
        e = c.elongation.value
        if e > 2:
            e = 2
        o = c.orientation.value
        r = c.equivalent_radius.value
        a = 3 * np.sqrt(e) * r
        b = 3 * r / np.sqrt(e)
        #print(e,o,r,a,b)
        # Define center (x, y), semi-major axis (a), semi-minor axis (b), and angle (radians)
        aperture = EllipticalAperture((c.x_centroid, c.y_centroid), a=a, b=b, theta=np.radians(o))
        # Generate boolean or sub-pixel mask for a given image shape
        emask = aperture.to_mask(method='center').to_image((4088,4088))
        mask[emask > 0] = True

   
    idbig = cat.area.value >= maxarea
    catbig = cat[idbig]
    # Expand elliptical masks
    for c in catbig:
        e = c.elongation.value
        if e > 2:
            e = 2
        o = c.orientation.value
        r = c.equivalent_radius.value
        a = nellipse * np.sqrt(e) * r
        b = nellipse * r / np.sqrt(e)
        #print(e,o,r,a,b)
        # Define center (x, y), semi-major axis (a), semi-minor axis (b), and angle (radians)
        aperture = EllipticalAperture((c.x_centroid, c.y_centroid), a=a, b=b, theta=np.radians(o))
        # Generate boolean or sub-pixel mask for a given image shape
        emask = aperture.to_mask(method='center').to_image((4088,4088))
        mask[emask > 0] = True

    # Mask negative values (below 5*sigma under the background)
    idx = data0 < -5 * bkg.background_rms
    mask[idx] = True
    
    # Mask bad pixels marked as "Do Not Use"  if dq is available
    if dq is not None:
        from roman_datamodels.dqflags import pixel
        idx = (dq & pixel.DO_NOT_USE.value) != 0
        mask[idx] = True

    # Return mask
    return mask


def computeFlat(sca, files, nellipse=4, nexpand=20, nsources=2, targets=None):
    """
    Compute the flat by coadding ditherings

    Parameters
    ----------
    isca : TYPE
        Detector number
        
    files: TYPE
        List of files

    Returns
    -------
    None.

    """

    import roman_datamodels as rdm
    import numpy as np
    from flo.desaturate import guidewindow
    from astropy.io import fits
    import warnings
    from astropy.stats import biweight_location

    # Ignore warnings
    warnings.filterwarnings('ignore')
 
    print('SCA '+sca+': ', end='')
    cube = np.zeros((len(files), 4088,4088))
    for i,file in enumerate(files):
        print('.', end=" ")
        with rdm.open(file) as dm:
            image = dm.data.copy()
            dq = dm.dq.copy()
            wfidetector = dm.meta.instrument.detector
            gs = dm.meta.guide_star
            mask = np.zeros((4088, 4088), dtype=bool)
            # NaN guide window
            x1,x2,y1,y2 = guidewindow(gs, wfidetector)
            mask[y1:y2, x1:x2] = True
            # Ghost of guide window 
            if wfidetector[-2:] == '12':
                xgh1, xgh2 = 2647, 2667
                ygh1, ygh2 = y1, y2-1            
                # NaN the elements
                mask[ygh1:ygh2,xgh1:xgh2] = True
            # Bad pixel in 1
            if wfidetector[-2:] == '01':
                mask[4060:,2070:2070+1200]=True                
            image[mask] = np.nan
            # Check if specific has been defined
            if targets is None:
                pass
            else:
                for target in targets:
                    center, radius = target
                    Xc,Yc = dm.meta.wcs.backward_transform(center.ra.deg, center.dec.deg)
                    Y, X = np.ogrid[:4088, :4088]
                    distance = np.hypot(X-Xc, Y-Yc)
                    image[distance < radius] = np.nan
            # NaN all sources
            mask = maskAllSources(image, sca=sca, dq =dq, eccthreshold=0, 
                                  areathreshold=100000, nexpand=nexpand, 
                                  nellipse=nellipse, nsources=nsources)
            image[mask] = np.nan
            cube[i] = image
    print()
    # Flattening
    medcube = np.zeros(len(files))
    for i in range(len(files)):
        medcube[i] = np.nanmedian(cube[i].ravel())
        cube[i] /= medcube[i]
    #sortedcube = np.sort(cube, axis=0)
    #flat = np.nanmean(sortedcube[:5],axis=0)
    flat = biweight_location(cube, axis=0, ignore_nan=True)
    #flat = np.nanmin(cube, axis=0)
    flat /= np.nanmedian(flat)
    # Saving
    hdu = fits.PrimaryHDU(data=flat)
    hdu.writeto('flat_'+sca+'.fits', overwrite=True)
    return cube