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

    
    outfile = file[:-5]+'_ok.asdf'
    with rdm.open(file) as dm:
        data = dm.data.copy()
        dq = dm.dq.copy()
    idx = ((data < 0) | (data > 50000)) & ((dq & pixel.DO_NOT_USE.value) == 0)
    med = np.nanmedian(data.ravel())
    print(file + f': {np.sum(idx)/4088**2 * 100:.2f}% negative non-masked pixels, median {med:.1f}')
    # Put data to NaN and flag pixel as do_not_use
    data[idx] = np.nan
    dq[idx] |= pixel.DO_NOT_USE.value
    
    # Mask pixels without non-linearity correction
    idx = (dq & pixel.NO_LIN_CORR) != 0
    data[idx] = np.nan
    dq[idx] |= pixel.DO_NOT_USE.value
    
    
    # Overwrite file
    with rdm.open(file) as dm:
        dm.data = data
        dm.dq = dq
        dm.save(file)   