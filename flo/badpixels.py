#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:05:57 2026

@author: dfadda
"""

def maskNegativePixels(file):
    """
    

    Parameters
    ----------
    L2 : TYPE
        L2 file 

    Returns
    -------
    None.

    """
    import roman_datamodels as rdm
    from roman_datamodels.dqflags import pixel   
    import numpy as np

    
    #outfile = file[:-5]+'_ok.asdf'
    with rdm.open(file) as dm:
        data = dm.data
        dq = dm.dq
        idx = ((data < 0) | (data > 50000)) & ((dq & pixel.DO_NOT_USE.value) == 0)
        med = np.nanmedian(data.ravel())
        print(file + f': {np.sum(idx)/4088**2 * 100:.2f}% negative non-masked pixels, median {med:.1f}')
        # Put data to NaN and flag pixel as do_not_use
        data[idx] = np.nan
        dq[idx] |= pixel.DO_NOT_USE.value
        
        # Classify as DO NOT USE all pixels with NaNs
        rows, cols = np.where(np.isnan(dm.data))
        dm.dq[rows,cols] |= pixel.DO_NOT_USE.value
        dm.save(file)   
        
        
def makeBadPixelMask(nsca,vmin=-0.05,vmax=0.05,npix=100):
    
    from glob import glob as gb
    import roman_datamodels as rdm
    import numpy as np
    from astropy.io import fits
    from flo.desaturate import guidewindow
    
    sca = '{0:02d}'.format(nsca)
    
    files = gb('./Cal/*wfi'+sca+'*_cal.asdf')
    nfiles = len(files)
    
    cube = np.empty((nfiles, 4088, 4088))
    for i,file in enumerate(files):
        with rdm.open(file) as dm:
            data = dm.data
            stripes = np.nanmedian(data, axis=1,keepdims=True)
            data -= stripes
            stripes = np.nanmedian(data, axis=0, keepdims=True)
            data -= stripes
            # Blank out guide windows
            x1, x2, y1, y2 = guidewindow(dm.meta.guide_star, dm.meta.instrument.detector)            
            data[y1:y2,x1:x2] = np.nan
            cube[i] = data #- np.nanmedian(data.ravel())
    
    minimage = np.nanmin(cube, axis=0)
    minimage -= np.nanmedian(minimage.ravel())
        
    idx = (minimage > vmax) | (minimage < vmin)
    mask = np.zeros((4088, 4088))
    mask[idx] = 1
    
    # Consider the clusters of masked pixels with at least npix elements
    print('remove isolated clusters at two scales')
    from scipy.ndimage import label, sum_labels
    mask = np.zeros((4088, 4088),dtype=bool)
    idx = (minimage > 0.04) | (minimage < -0.035)
    mask[idx] = True
    labeled_mask, num_features = label(mask, structure=np.ones((3,3)))
    cluster_sizes = sum_labels(mask, labeled_mask, range(1, num_features + 1))
    min_size = 1000
    large_cluster_labels = np.where(cluster_sizes > min_size)[0] + 1
    mask1000 = np.isin(labeled_mask, large_cluster_labels)
    mask = np.zeros((4088, 4088),dtype=bool)
    idx = (minimage > vmax) | (minimage < vmin)
    mask[idx] = True
    labeled_mask, num_features = label(mask, structure=np.ones((3,3)))
    cluster_sizes = sum_labels(mask, labeled_mask, range(1, num_features + 1))
    min_size = 100
    large_cluster_labels = np.where(cluster_sizes > min_size)[0] + 1
    mask100 = np.isin(labeled_mask, large_cluster_labels)
    mask = mask100 | mask1000
                
    # Add circles of bad pixels
    Y, X = np.ogrid[:4088, :4088]
    circles = []
    if sca == '03':
        circles = [(270,1663,23), (864, 2047,35), (3404,2369,19)]
    elif sca == '05':
        circles = [(2678,2896,23), (2771,4033,35), (2051,3771,66)]
    elif sca == '08':
        circles = [(353,1766,16), (2134, 4027, 145)]
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
            mask[cmask] = True
            
    # Write to a FITS file
    mask = mask.astype(np.uint8)
    hdu = fits.PrimaryHDU(data=mask)
    hdu.writeto('badpixels_'+sca+'.fits', overwrite=True)

    return mask

def maskBadPixels(calfile, indir='./Postcal', outdir='./Postcal'):
    
    from astropy.io import fits        
    import roman_datamodels as rdm
    from roman_datamodels.dqflags import pixel
    import numpy as np
    import os
    
    # Create the output directory if not existent
    os.makedirs(outdir, exist_ok=True)
    
    #path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..',"data") 
    with rdm.open(os.path.join(indir, calfile)) as dm:
        sca = dm.meta.filename[-16:-14]
        isca = int(sca)
        #  Mask the guide window
        gs = dm.meta.guide_star
        # correct window (the dq does not work perfectly ...)
        if isca in [3,6,9,12,15,18]:
            x1 = gs['window_xstart']-5
            x2 = gs['window_xstop']-4
            y1 = gs['window_ystart']-3
            y2 = gs['window_ystop']-2
        else:
            x1 = gs['window_xstart']-3
            x2 = gs['window_xstop']-2
            y1 = gs['window_ystart']-5
            y2 = gs['window_ystop']-4
        # NaN the elements
        dm.data[y1:y2,x1:x2] = np.nan
        # Change the dq to 'do not use'
        dm.dq[y1:y2,x1:x2] |= pixel.DO_NOT_USE.value
        
        # Ghost of guide window 
        if sca == '12':
            xgh1, xgh2 = 2647, 2667
            ygh1, ygh2 = y1, y2-1            
            # NaN the elements
            dm.data[ygh1:ygh2,xgh1:xgh2] = np.nan
            # Change the dq to 'do not use'
            dm.dq[ygh1:ygh2,xgh1:xgh2] |= pixel.DO_NOT_USE.value

        
        # Add circles of bad pixels
        Y, X = np.ogrid[:4088, :4088]
        circles = []
        if sca == '03':
            circles = [(270,1663,23), (864, 2047,35), (3404,2369,19)]
        elif sca == '05':
            circles = [(2678,2896,23), (2771,4033,35), (2051,3771,25)]
        elif sca == '08':
            circles = [(353,1766,16), (2134, 4027, 16)]
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
                dm.data[cmask] = np.nan
                dm.dq[cmask] |= pixel.DO_NOT_USE.value
 
                
        # Classify as DO NOT USE all pixels with NaNs
        rows, cols = np.where(np.isnan(dm.data))
        dm.dq[rows,cols] |= pixel.DO_NOT_USE.value
                
        # Open improved bad pixel mask
        #badpixfile = 'badpixels_'+sca+'.fits.gz'
        #with fits.open(os.path.join(path, badpixfile)) as f:
        #    mask = f[0].data
        #    idx = np.where(mask == 1)
        #    dm.data[idx] = np.nan
        #    dm.dq[idx] |= pixel.DO_NOT_USE.value        
        # Save file
        dm.save(os.path.join(outdir, calfile))
        