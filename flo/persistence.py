#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:35:19 2026

@author: dfadda
"""

def maskBadRegion(dm, sca):

    import numpy as np
    from roman_datamodels.dqflags import pixel
        
    # Add circles of bad pixels
    Y, X = np.ogrid[:4088, :4088]
    circles = []
    if sca == '03':
        circles = [(270,1663,23), (864, 2047,35), (3404,2369,19)]
    elif sca == '05':
        circles = [(2678,2896,23), (2771,4033,35), (2051,3772,65)]
    elif sca == '08':
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
            dm.data[cmask] = np.nan
            dm.dq[cmask] &= ~pixel.SATURATED.value
 

def killPersistence(xc, yc, r, image, kill=False, train=False):
    """
    Remove a persistence from an image by computing a median circular profile

    Parameters
    ----------
    xc : TYPE
        barycenter of saturation
    yc : TYPE
        barycenter of saturation
    n : TYPE
        number of saturated pixels
    image : TYPE
        original image

    Returns
    -------
    image : TYPE
        image without persistence

    """
    
    import numpy as np
    #r = int(11.656543612388592 +  0.0299272208042616 * n) + 5
    if train:
        dd = 70
    else:
        dd = r + 5

    
    x1, x2 = max([0,int(xc)-dd]), min([4087,int(xc)+dd])
    y1, y2 = max([0,int(yc)-dd]), min([4087,int(yc)+dd])  
    x0, y0 = min([xc, dd+xc-int(xc)]), min([yc, dd+yc-int(yc)])
    subimage = image[y1:y2, x1:x2].copy()
    
    Y, X = np.ogrid[:(y2-y1), :(x2-x1)]
    Yc, Xc = x0, y0
    distance = np.hypot(X - Xc,Y - Yc)
    if r > 200:
        line = np.abs(np.tan(47*np.pi/180)*(X-Xc) - (Y-Yc))
        spike = (line < 12) 
        line = np.abs(np.tan(-47*np.pi/180)*(X-Xc) - (Y-Yc))
        spike |= (line < 4) 
        line = np.abs(np.tan(-12*np.pi/180)*(X-Xc) - (Y-Yc))
        spike |= (line < 4) 
        line = np.abs(np.tan(12*np.pi/180)*(X-Xc) - (Y-Yc))
        spike |= (line < 4) 
        line = np.abs(np.tan(69*np.pi/180)*(X-Xc) - (Y-Yc))
        spike |= (line < 8) 
        line = np.abs(np.tan(-69*np.pi/180)*(X-Xc) - (Y-Yc))
        spike |= (line < 8) 
        
    back = np.nanmedian(subimage[(distance > dd-5) & (distance <= dd)])
    profile = np.nanmedian(subimage[distance < 2])
    subimage[distance < 2] -= (profile - back)
    verbose = True
    if train:
        for j in range(3, dd+1):
            cmaskj =  (distance > j-1) & (distance <= j)
            if (j>r-2) & (r>200) :
                profile = np.nanmedian(subimage[cmaskj & ~spike])
            else:
                profile = np.nanmedian(subimage[cmaskj])
            if profile > back:
                subimage[cmaskj] -= (profile - back)
            else:
                rout = j
                if verbose:
                    print('max radius is: ', j, profile, back)
                verbose = False
                break
    else:
        for j in range(3, r+1):
            cmaskj =  (distance > j-1) & (distance <= j)
            if (j>r-2) & (r>200) :
                profile = np.nanmedian(subimage[cmaskj & ~spike])
            else:
                profile = np.nanmedian(subimage[cmaskj])
            subimage[cmaskj] -= (profile - back)
    if kill:
        image[y1:y2, x1:x2]=subimage 
    if train:
        return rout
    else:
        return r


def findPersistence(files, pfiles, min_size=5, filter=None, kill=False , save=False, train=False):
    """
    The code finds persistences and modify up to three subsequent files

    Parameters
    ----------
    files : TYPE
        Original calibrated files
    pfiles : TYPE
        Postcalibrated files
    min_size : TYPE, optional
        Minimum number of saturated pixels. The default is 5.

    Returns
    -------
    None.

    """
    
    
    from scipy.ndimage import label, sum_labels
    import numpy as np
    import roman_datamodels as rdm
    from roman_datamodels.dqflags import pixel
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from astropy import visualization



    kernel = np.ones((3,3), dtype=int)  
    Y, X = np.ogrid[:4088, :4088]
    
    if filter == 'F146':
        intercept = 6.30
        slope =  2.10
    elif filter == 'F213':
        #intercept = 6.35
        #slope = 3.26        
        intercept = 7.22
        slope = 2.35
    elif filter == 'F106':
        intercept = 7.28
        slope = 2.18
    elif filter == 'F062':
        intercept = 6.11
        slope = 2.39
    else:
        print("Please, specify filter !")
    dw = 100 # size of half window

    for ifile in range(len(files)-1):
        file = files[ifile]
        with rdm.open(file) as dm:
            dq = dm.dq.copy()
            sca = dm.meta.instrument.detector[-2:]
            maskBadRegion(dm, sca)
            image = dm.data.copy()
        #with rdm.open(files[ifile+1]) as dm:
        #    nextimage = dm.data.copy()
        nextfile = pfiles[ifile+1]
        if ifile+2 < len(files):
            next2file = pfiles[ifile+2]
        if ifile+3 < len(files):
            next3file = pfiles[ifile+3]
        nan_mask = np.isnan(image) & ((dq & pixel.SATURATED.value) != 0) #& ((dq & pixel.DO_NOT_USE) != 1)
        if np.sum(nan_mask) > min_size:
            with rdm.open(nextfile) as dm:
                next1image = dm.data
                labeled_mask, num_features = label(nan_mask, structure=kernel)
                cluster_sizes = sum_labels(nan_mask, labeled_mask, range(1, num_features + 1))
                large_cluster_labels = np.where(cluster_sizes > min_size)[0] + 1
                # centroids
                if len(large_cluster_labels) > 0:
                    n = len(large_cluster_labels)
                    for i in range(n):
                        llabel = large_cluster_labels[i]
                        rows, cols = np.where(labeled_mask == llabel)
                        yc = np.nanmean(rows)
                        xc = np.nanmean(cols)
                        ns = np.sum(labeled_mask == llabel)
                        rc = np.sqrt(ns/np.pi)
                        distance = np.hypot(X - xc,Y - yc)
                        flux = np.nanmedian(image[(distance > rc)  & (distance < (rc+4))])
                        newflux = np.nanmedian(next1image[(distance > rc)  & (distance < (rc+4))])
                        back = np.nanmedian(image[(distance > (rc+10))  & (distance < (rc+12))])
                        frc = next1image[distance < rc]
                        fmean = np.nanmean(frc)
                        nnan = np.isnan(frc).sum()
                        # check if there is flux
                        if (flux > 2 * back) & (flux > 2 * newflux) & np.isfinite(fmean) & (nnan < 4):   
                            # Refine barycenter from persistence
                            rows, cols = np.where(distance < rc)
                            frc /= fmean
                            xc = np.nanmean(cols*frc)
                            yc = np.nanmean(rows*frc)
                            print('file, x, y, n ', ifile, xc, yc, ns)
                            xx, yy = int(xc), int(yc)
                            x1 = max([0,xx-dw])
                            x2 = min([4088,xx+dw])
                            y1 = max([0,yy-dw])
                            y2 = min([4088,yy+dw])            
                            subimage = image[y1:y2,x1:x2]
                            percentile = 95
                            interval = visualization.PercentileInterval(percentile)
                            vmin, vmax = interval.get_limits(subimage)
                            norm = visualization.ImageNormalize(
                                vmin=vmin, vmax=vmax, stretch=visualization.AsinhStretch(a=1)
                            )
                            percentile = 85
                            interval = visualization.PercentileInterval(percentile)
                            vmin, vmax = interval.get_limits(next1image[y1:y2,x1:x2])
                            norm2 = visualization.ImageNormalize(
                                vmin=vmin, vmax=vmax, stretch=visualization.AsinhStretch(a=1)
                            )
                            fig,(ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(12,3))
                            ax1.imshow(image, norm=norm, origin='lower', cmap='inferno')
                            ax1.set_xlim(xx-dw,xx+dw)
                            ax1.set_ylim(yy-dw,yy+dw)
                            circle = patches.Circle((xx,yy), rc, edgecolor='red', facecolor='none', linewidth=2)
                            ax1.add_patch(circle)
                            rtot = int(intercept + rc * slope)
                            rmin = killPersistence(xc,yc, rtot, next1image, kill=kill, train=train)
                            ax2.imshow(next1image, norm=norm2, origin='lower', cmap='inferno')
                            ax2.set_xlim(xx-dw,xx+dw)
                            ax2.set_ylim(yy-dw,yy+dw)
                            circle = patches.Circle((xx,yy), rmin, edgecolor='green', facecolor='none', linewidth=2)
                            ax2.add_patch(circle)
                            if save:
                                dm.save(nextfile)
                            if ifile+2 < len(files):
                                with rdm.open(next2file) as dm2:
                                    next2image = dm2.data
                                    percentile = 85
                                    interval = visualization.PercentileInterval(percentile)
                                    vmin, vmax = interval.get_limits(next2image[y1:y2,x1:x2])
                                    norm2 = visualization.ImageNormalize(
                                        vmin=vmin, vmax=vmax, stretch=visualization.AsinhStretch(a=1))
                                    rmin = killPersistence(xc,yc,rtot,next2image, kill=kill,train=train)
                                    ax3.imshow(next2image, norm=norm2, origin='lower', cmap='inferno')
                                    ax3.set_xlim(xx-dw,xx+dw)
                                    ax3.set_ylim(yy-dw,yy+dw) 
                                    circle = patches.Circle((xx,yy), rmin, edgecolor='green', facecolor='none', linewidth=2)
                                    ax3.add_patch(circle)
                                    if save:
                                        dm2.save(next2file)
                            if ifile+3 < len(files):
                                with rdm.open(next3file) as dm3:
                                    next3image = dm3.data
                                    rmin = killPersistence(xc,yc,rtot,next3image,kill=kill,train=train)
                                    percentile = 85
                                    interval = visualization.PercentileInterval(percentile)
                                    vmin, vmax = interval.get_limits(next3image[y1:y2,x1:x2])
                                    norm2 = visualization.ImageNormalize(
                                        vmin=vmin, vmax=vmax, stretch=visualization.AsinhStretch(a=1))
                                    ax4.imshow(next3image, norm=norm2, origin='lower', cmap='inferno')
                                    ax4.set_xlim(xx-dw,xx+dw)
                                    ax4.set_ylim(yy-dw,yy+dw) 
                                    circle = patches.Circle((xx,yy), rmin, edgecolor='green', facecolor='none', linewidth=2)
                                    ax4.add_patch(circle)
                                    if save:
                                        dm3.save(next3file)
                            plt.show()                
