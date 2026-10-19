def create_cell_json(target, x0, y0, ra0, dec0, ra0c, dec0c, members, nx=5000, ny=5000, pixscale=0.055/3600):
    """
    Generate a JSON structure identical to skycells for romancal

    Parameters
    ----------
    target : TYPE
        DESCRIPTION.
    x0 : TYPE
        DESCRIPTION.
    y0 : TYPE
        DESCRIPTION.
    ra0 : TYPE
        DESCRIPTION.
    dec0 : TYPE
        DESCRIPTION.
    ra0c : TYPE
        DESCRIPTION.
    dec0c : TYPE
        DESCRIPTION.
    members : TYPE
        DESCRIPTION.
    nx : TYPE, optional
        DESCRIPTION. The default is 5000.
    ny : TYPE, optional
        DESCRIPTION. The default is 5000.
    pixscale : TYPE, optional
        DESCRIPTION. The default is 0.055/3600.

    Returns
    -------
    result : TYPE
        json structure

    """

    skycell_info = {
        "name": target,
        "pixel_scale": pixscale,
        "ra_projection_center": ra0,
        "dec_projection_center": dec0,
        "x0_projection": x0,
        "y0_projection": y0,
        "ra_center": ra0c,
        "dec_center": dec0c,
        "nx": nx,
        "ny": ny,
        "orientat": -1.3823617696762085,
        "orientat_projection_center": 0.0
    }

    product = {
        "name": "_stars_"+target, 
        "members": []
    }

    for member in members:
        product['members'].append({'expname': member, 'exptype': 'science'})
    
    result = {
        "asn_type": "image",
        "asn_rule": "DMS_ELPP_Base",
        "version_id": None,
        "code_version": "0.22.0",
        "target": target,
        "degraded_status": "No known degraded exposures in association.",
        "program": "StarF",
        "data_release_id": "stars",
        "constraints": "No constraints",
        "asn_id": "a3001",
        "asn_pool": "none",
        "skycell_wcs_info": skycell_info,
        "products":[product]
    }

    return result


def make_cell_asn(files, npix=5000, pixscale=0.11, outname='mosaic_'):
    """
    This code generates json files to reproject and coadd a list of files
    in a series of square tiles with the same center of projection.
    The association files to run romancal/mosaicpipeline are generated
    in json format using the outname as root for their names.  

    Parameters
    ----------
    files : TYPE
        a list of asdf single image files
    size : TYPE, optional
        the minimum size of the square tile. The default is 5000.
    pixscale : TYPE, optional
        the pixel scale. The default is 0.055.
    outname : TYPE, optional
        the root name of the json files. The default is 'mosaic_'.

    Returns
    -------
    int
        0 stands for failure

    """

    import roman_datamodels as rdm
    import numpy as np
    from astropy.modeling import models
    from astropy import coordinates as coord
    from astropy import units as u
    from gwcs import wcs
    from gwcs import coordinate_frames as cf
    from flo.polygons import convex_polygons_intersect
    import io
    import json



    # Get centers and ra min,max and dec min,max for each file

    ra0, dec0 = [], []
    ra, dec = [], []  # List of vertices
    ramin, ramax, decmin, decmax = [], [], [], []     
    for file in files:
        with rdm.open(file) as dm:
            wcs = dm.meta.wcs
            a0, d0 = wcs(2044.5,2044.5)
            a1, d1 = wcs(0,0)
            a2, d2 = wcs(4087,0)
            a3, d3 = wcs(4087,4087)
            a4, d4 = wcs(0,4087)
            ra0.append(a0)
            dec0.append(d0)
            ra.append([a1,a2,a3,a4])
            dec.append([d1,d2,d3,d4])
            ramin.append(min([a1,a2,a3,a4]))
            ramax.append(max([a1,a2,a3,a4]))
            decmin.append(min([d1,d2,d3,d4]))
            decmax.append(max([d1,d2,d3,d4]))

    # Compute the center of final mosaic as barycenter of the image centers
    
    crval1 = np.mean(ra0)
    crval2 = np.mean(dec0)
    
    # Find size of final mosaic (symmetric wrt the central coordinate)
    
    dmax = max(decmax)
    dmin = min(decmin)
    amax = max(ramax)
    amin = min(ramin)
    
    da = max([amax-a0, a0-amin])
    dd = max([dmax-d0, d0-dmin])
    
    nx = 2 * 15 * da * np.cos(crval2 * np.pi / 180) / pixscale
    ny = 2 * dd * 3600 / pixscale
    
    # Find size of the cell
    if (nx < npix) & (ny < npix):
        print('There is no need to do smaller mosaics')
        return 0
    
    # Divide the mosaic in cells and save the associations as json asn files
    ncol = np.ceil(np.ceil(nx)/npix)
    nrow = np.ceil(np.ceil(ny)/npix)
    # Make them odd numbers to center first cell at 0,0
    ncol = int(ncol // 2 * 2 + 1)
    nrow = int(nrow // 2 * 2 + 1)
    # Build cells
    col, row = np.indices((ncol, nrow))
    col -= ncol//2
    row -= nrow//2
    
    # Compute nx, ny
    nx = ncol * npix
    ny = nrow * npix
    
    # Astrometry of the tile
    x0t, y0t = (nx-1)/2 , (ny-1)/2
    pix = pixscale / 3600.
    pixelshift = models.Shift(-x0t) & models.Shift(-y0t)
    pixelscale = models.Scale(-pix) & models.Scale(pix) # 0.1 arcsec/pixel
    tangent_projection = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(crval1, crval2, 180.)
    det2sky = pixelshift | pixelscale | tangent_projection | celestial_rotation
    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs', unit=(u.deg, u.deg))
    wcsobj = wcs.WCS([(detector_frame, det2sky),(sky_frame, None)])

    
    # Cell center
    x0, y0 = col * npix + x0t, row * npix + y0t
    # Cell corners
    hcellsize = npix // 2 
    x1, y1 = x0 - hcellsize, y0 - hcellsize
    x2, y2 = x0 + hcellsize, y0 - hcellsize
    x3, y3 = x0 + hcellsize, y0 + hcellsize
    x4, y4 = x0 - hcellsize, y0 + hcellsize
    
    a0, d0 = wcsobj(x0, y0)
    a1, d1 = wcsobj(x1, y1)
    a2, d2 = wcsobj(x2, y2)
    a3, d3 = wcsobj(x3, y3)
    a4, d4 = wcsobj(x4, y4)

    for i in range(nrow):
        for j in range(ncol):

            pcell = [(a1[i,j],d1[i,j]),(a2[i,j],d2[i,j]),(a3[i,j],d3[i,j]),(a4[i,j],d4[i,j])]
            members = []
            # Find the overlapping images
            for ra_, dec_, file in zip(ra, dec, files):
                pimage = [(ra_[k],dec_[k]) for k in range(4)]
                if convex_polygons_intersect(pcell, pimage):
                    members.append(file)
                print(i,j,members)
    
    
            target = '160p57x61y60'
            cell_json = create_cell_json(target, x0[i,j], y0[i,j], ra0[i,j], dec0[i,j], a0[i,j], d0[i,j], members)

            filename = 'test.asn'
            with io.open(filename, mode='w') as f:
                str_= json.dumps(cell_json, indent=4, separators=(',', ': '), ensure_ascii=False)
                f.write(str_)