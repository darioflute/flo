def create_cell_json(target, x0, y0, ra0, dec0, ra0c, dec0c, orient, members, outname, nx=5000, ny=5000, pixscale=0.055/3600):
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
        "orientat": ra0-ra0c,
        "orientat_projection_center": orient
    }

    product = {
        "name": "_"+outname+"_"+target, 
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
        "program": outname,
        "data_release_id": outname,
        "constraints": "No constraints",
        "asn_id": "a0001",
        "asn_pool": "none",
        "skycell_wcs_info": skycell_info,
        "products":[product]
    }

    return result

def angle360(x):
    # Express angles inside 0<= x <= 360
    if x < 0:
        x += 360
    elif x > 360:
        x -= 360
    return x


def computeOrientation(wcs):
    """
    Compute the orientation of a L2 product from header

    Parameters
    ----------
    wcs : TYPE
        wcs from the meta

    Returns
    -------
    theta: 
        orientation angle

    """
    import numpy as np
    
    hdr = wcs.to_fits()[0]
    cd = np.zeros((2,2))
    try:
        cd[0,0] = hdr['CD1_1']
        cd[0,1] = hdr['CD1_2']
        cd[1,0] = hdr['CD2_1']
        cd[1,1] = hdr['CD2_2']
    except:
        cd[0,0] = hdr['PC1_1']
        cd[0,1] = hdr['PC1_2']
        cd[1,0] = hdr['PC2_1']
        cd[1,1] = hdr['PC2_2']
        

    scale_x = np.hypot(cd[0, 0], cd[1, 0])
    scale_y = np.hypot(cd[0, 1], cd[1, 1])

    # Normalize the CD matrix to get the rotation/parity matrix
    rot_matrix = cd / [[scale_x, scale_x], [scale_y, scale_y]]

    # Rotation angle
    theta_rad = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    theta_deg = np.degrees(theta_rad)

    return scale_x, theta_deg   


def make_cells_asn(files, npix=5000, pixscale=0.10876549, outname='mosaic', refdir='./'):
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
    orientation = []
    for file in files:
        with rdm.open(file) as dm:
            a0, d0 = dm.meta.wcs(2044.5,2044.5)
            a1, d1 = dm.meta.wcs(0,0)
            a2, d2 = dm.meta.wcs(4087,0)
            a3, d3 = dm.meta.wcs(4087,4087)
            a4, d4 = dm.meta.wcs(0,4087)
            ra0.append(a0)
            dec0.append(d0)
            ra.append([a1,a2,a3,a4])
            dec.append([d1,d2,d3,d4])
            ramin.append(min([a1,a2,a3,a4]))
            ramax.append(max([a1,a2,a3,a4]))
            decmin.append(min([d1,d2,d3,d4]))
            decmax.append(max([d1,d2,d3,d4]))
            cd, cr = computeOrientation(dm.meta.wcs)
            orientation.append(cr)

    # Compute the center of final mosaic as barycenter of the image centers
    
    crval1 = np.mean(ra0)
    crval2 = np.mean(dec0)
    
    # Compute the median orientation
    
    medorientation = np.nanmedian(np.array(orientation))
    print(f'Median orientation is {medorientation} degrees')
    
    # Find size of final mosaic (symmetric wrt the central coordinate)
    
    dmax = max(decmax)
    dmin = min(decmin)
    amax = max(ramax)
    amin = min(ramin)
    
    # Compute tentative wcs of tile
    print('orientation ', medorientation)
    pix = pixscale / 3600.
    x0t, y0t = 0, 0
    pix = pixscale / 3600.
    pixelshift = models.Shift(-x0t) & models.Shift(-y0t)
    pixelscale = models.Scale(-pix) & models.Scale(pix) # 0.1 arcsec/pixel
    tangent_projection = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(crval1, crval2, -medorientation)
    det2sky = pixelshift | pixelscale | tangent_projection | celestial_rotation
    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs', unit=(u.deg, u.deg))
    wcsobj = wcs.WCS([(detector_frame, det2sky),(sky_frame, None)])
    
    
    x, y = [], []
    for a, d in zip(ra, dec):
        x1, y1 = wcsobj.backward_transform(a[0], d[0])
        x2, y2 = wcsobj.backward_transform(a[1], d[1])
        x3, y3 = wcsobj.backward_transform(a[2], d[2])
        x4, y4 = wcsobj.backward_transform(a[3], d[3])
        x.extend([x1,x2,x3,x4])
        y.extend([y1,y2,y3,y4])
        
    nx = max(x) - min(x)
    ny = max(y) - min(y)
    
    
    #da = max([amax-crval1, crval1-amin])
    #dd = max([dmax-crval2, crval2-dmin])
    #nx = 2 * da * np.cos(crval2 * np.pi / 180)  * 3600 / pixscale
    #ny = 2 * dd * 3600 / pixscale
    
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
    pixelshift = models.Shift(-x0t) & models.Shift(-y0t)
    pixelscale = models.Scale(-pix) & models.Scale(pix) # 0.1 arcsec/pixel
    tangent_projection = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(crval1, crval2, -medorientation)
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

    # Part of the file name (tile center coords)
    dsign = 'm' if (crval1 < 0) else 'p'
    ra0_, dec0_ = round(crval1), round(np.abs(crval2))
    # Format of name
    namefmt = '{0:03d}{1:s}{2:02d}x{3:02d}y{4:02d}'
    npix2 = (npix-1)/2
    for i in range(nrow):
        for j in range(ncol):
            pcell = [(a1[i,j],d1[i,j]),(a2[i,j],d2[i,j]),(a3[i,j],d3[i,j]),(a4[i,j],d4[i,j])]
            members = []
            # Find the overlapping images
            for ra_, dec_, file in zip(ra, dec, files):
                pimage = [(ra_[k],dec_[k]) for k in range(4)]
                if convex_polygons_intersect(pcell, pimage):
                    members.append(file)
            if len(members) > 0:
                print(i,j,members)
                # Target definition
                x0_ = col[i,j] + ncol//2
                y0_ = row[i,j] + nrow//2
                xpix = x0t - x0[i,j] + npix2
                ypix = y0t - y0[i,j] + npix2
                target = namefmt.format(ra0_, dsign, dec0_, x0_, y0_)
                cell_json = create_cell_json(target, xpix, ypix, crval1, crval2, 
                                             a0[i,j], d0[i,j], medorientation, members, outname)
                filename = outname + '_' + target + '.asn'
                with io.open(filename, mode='w') as f:
                    str_= json.dumps(cell_json, indent=4, separators=(',', ': '), ensure_ascii=False)
                    f.write(str_)
                
    # Fake skycell reference file
    from astropy.time import Time
    from datetime import time,date,datetime
    import asdf
    import roman_datamodels._stnode as stnode
    import os

    d = date(2025, 1, 1)
    t = time(0, 0, 0)
    startdate = datetime.combine(d, t)

    # Structure of the tile metadata
    wcs_tile_dtype = [
        ('index', 'i4'),
        ('ra_tangent', 'f8'),      # RA tile center  [crval1]
        ('dec_tangent', 'f8'),     # Dec tile center [crval2]
        ('ra_min', 'f8'),         # Limits in RA of the tile
        ('ra_max', 'f8'),         # 
        ('dec_min', 'f8'),        # Limits in Dec of the tile
        ('dec_max', 'f8'),        #
        ('orientat', 'f4'),       # Orientation projection [crota2]
        ('x_tangent', 'f8'), # x coord of projection region center
        ('y_tangent', 'f8'), # y coord of projection region center
        ('nx', 'i4'), # x-size of projection region in pixels
        ('ny', 'i4'), # y-size of projection region in pixels
        ('skycell_start','i4'),           # First cell index
        ('skycell_end','i4'),             # Last cell index
    ]

    # Structure of the cell metadata
    wcs_cell_dtype = [
        ('name', 'U16'),
        ('ra_center', 'f8'),     # center of the cell
        ('dec_center', 'f8'),    # center of the cell
        ('orientat', 'f4'),      # orientation of the cell
        ('x_tangent', 'f8'), # x coordinate of projection center in pixels [crpix1]
        ('y_tangent', 'f8'), # y coordinate of projection center in pixels [crpix2]
        ('ra_corn1', 'f8'),      # RA corner 1
        ('dec_corn1', 'f8'),
        ('ra_corn2', 'f8'),
        ('dec_corn2', 'f8'),
        ('ra_corn3', 'f8'),
        ('dec_corn3', 'f8'),
        ('ra_corn4', 'f8'),
        ('dec_corn4', 'f8')
    ]

    ntiles = 1
    tiles = np.empty(ntiles, dtype=wcs_tile_dtype)
    cells = np.empty(0,  dtype=wcs_cell_dtype)
    
    tile = tiles[0]
    tile['index'] = 0
    tile['ra_tangent'] = '{0:.17g}'.format(crval1)
    tile['dec_tangent'] = '{0:.17g}'.format(crval2)
    tile['ra_min'] = '{0:.17g}'.format(amin)
    tile['ra_max'] = '{0:.17g}'.format(amax)
    tile['dec_min'] = '{0:.17g}'.format(dmin)
    tile['dec_max'] = '{0:.17g}'.format(dmax)       
    tile['skycell_start'] = len(cells)     # First cell index
    tile['x_tangent'] = x0t
    tile['y_tangent'] = y0t
    tile['nx'] = nx
    tile['ny'] = ny     
    tile['orientat'] = 180-medorientation # Orientation projection

    icell = 0
    cell = np.empty(ncol * nrow, dtype=wcs_cell_dtype)
    for i in range(nrow):
        for j in range(ncol):
            # Target definition
            x0_ = col[i,j] + ncol//2
            y0_ = row[i,j] + nrow//2
            xpix = x0t - x0[i,j] + npix2
            ypix = y0t - y0[i,j] + npix2
            target = namefmt.format(ra0_, dsign, dec0_, x0_, y0_)
            cell[icell]['name'] = namefmt.format(ra0_, dsign, dec0_, x0_, y0_)
            cell[icell]['ra_center'] = '{0:.17g}'.format(a0[i,j]) # cell center
            cell[icell]['dec_center'] = '{0:.17g}'.format(d0[i,j])
            cell[icell]['orientat'] = crval1 - a0[i,j] # orientation wrt tile
            cell[icell]['x_tangent'] = xpix # position of tile center
            cell[icell]['y_tangent'] = ypix
            cell[icell]['ra_corn1'] = '{0:.17g}'.format(a1[i,j])
            cell[icell]['dec_corn1'] = '{0:.17g}'.format(d1[i,j])
            cell[icell]['ra_corn2'] = '{0:.17g}'.format(a2[i,j])
            cell[icell]['dec_corn2'] = '{0:.17g}'.format(d2[i,j])
            cell[icell]['ra_corn3'] = '{0:.17g}'.format(a3[i,j])
            cell[icell]['dec_corn3'] = '{0:.17g}'.format(d3[i,j])
            cell[icell]['ra_corn4'] = '{0:.17g}'.format(a4[i,j])
            cell[icell]['dec_corn4'] = '{0:.17g}'.format(d4[i,j])
            icell += 1
    # Concatenate cells to the cells from previous tiles
    cells = np.concatenate([cells, cell])
    tile['skycell_end'] = len(cells)       # Last cell index
    meta = {
        "reftype": "SKYCELLS",
        "pedigree": "GROUND",
        "description": "Roman Telescope has adopted a tessellation scheme of the celestial sphere to obtain mosaics stored in the Roman archive. This tessellation partitions the sphere into 4058 sky tiles whose centers and limits in declination are defined by a Healpix double tessellation scheme with Nside = 13.",
        "author": "Dario Fadda",
        "useafter": Time(startdate),
        "telescope": "ROMAN",
        "origin": "STSCI",
        "instrument": {
            "name": "WFI"
        },
        "vparity": -1,
        "nxy_skycell": npix,
        "skycell_border_pixels": 0,
        "pixel_scale": pixscale,
    }
    skycellref = stnode.SkycellsRef()
    skycellref['meta'] = meta
    skycellref['projection_regions'] = tiles
    skycellref['skycells'] =  cells
    skycellref['datamodel_name'] = "RomanSkycellsRefModel"
    with asdf.AsdfFile() as afout:
        afout = asdf.AsdfFile()
        afout.tree['roman'] = skycellref
        afout.write_to(os.path.join(refdir, 'roman_wfi_skycells_0001.asdf')) 
        
def stitchCells(files, outname = 'mosaic'):
    """
    

    Parameters
    ----------
    files : list
        List of files to stitch together
    outname : string
        Name of output FITS file. The default is 'mosaic'.

    Returns
    -------
    None.

    """
    from astropy.io import fits
    import roman_datamodels as rdm
    import numpy as np
    
    nx = 0
    ny = 0
    for file in files:
        nx_ = int(file[-16:-14])
        ny_ = int(file[-13:-11])
        nx = nx_ if nx_ > nx else nx
        ny = ny_ if ny_ > ny else ny
    nx += 1
    ny += 1
    
    nxc = (nx-1)//2
    nyc = (ny-1)//2

    nx *= 5000
    ny *= 5000
    image = np.full((ny, nx), np.nan)
    for file in files:
        nx_ = int(file[-16:-14])
        ny_ = int(file[-13:-11])
        with rdm.open(file) as dm:
            image[ny_ * 5000: ny_ * 5000 + 5000, nx_ * 5000: nx_ * 5000 + 5000] = dm.data
            if (nx_ == nxc) & (ny_ == nyc):
                #meta = dm.meta
                #header = meta.wcs.to_fits()[0]
                #crval1 = header['crval1']
                #crval2 = header['crval2']
                #cdelt, crota2 = computeOrientation(meta.wcs)
                cdelt = dm.meta.wcsinfo['pixel_scale']
                crval1 = dm.meta.wcsinfo['ra_ref']
                crval2 = dm.meta.wcsinfo['dec_ref']
                crota2 = -dm.meta.wcsinfo['orientation_ref']

    #image = image[::-1, ::-1]
    #header = meta.wcs.to_fits()[0]
    ctheta = np.cos(crota2 * np.pi/180)
    stheta = np.sin(crota2 * np.pi/180)
    cdelt1 = -cdelt
    cdelt2 = cdelt
    
    # crop image NaN borders
    nans = np.isnan(image)
    # Find rows and columns that are NOT completely NaN
    valid_rows = ~np.all(nans, axis=1)
    valid_cols = ~np.all(nans, axis=0)
    if not np.any(valid_rows) or not np.any(valid_cols):
        print('No valid rows or columns')
        return 0
    # Get min and max indices for non-NaN rows and columns
    row_indices = np.where(valid_rows)[0]
    col_indices = np.where(valid_cols)[0]
    r_min, r_max = row_indices[0], row_indices[-1]
    c_min, c_max = col_indices[0], col_indices[-1]
    image = image[r_min : r_max + 1, c_min : c_max + 1]
    crpix1 = nx/2 - c_min
    crpix2 = ny/2 - r_min

    hdu = fits.PrimaryHDU(data=image)
    hdu.header['wcsaxes'] = 2
    hdu.header['wcsname'] = 'world'
    hdu.header['crpix1'] = crpix1
    hdu.header['crpix2'] = crpix2
    hdu.header['crval1'] = crval1
    hdu.header['crval2'] = crval2
    hdu.header['PC1_1'] =  cdelt1 * ctheta
    hdu.header['PC1_2'] = -cdelt2 * stheta
    hdu.header['PC2_1'] =  cdelt1 * stheta
    hdu.header['PC2_2'] =  cdelt2 * ctheta
    hdu.header['cdelt1'] = 1.0
    hdu.header['cdelt2'] = 1.0
    hdu.header['ctype1'] = "RA---TAN"
    hdu.header['ctype2'] = "DEC--TAN"
    hdu.header['cunit1'] = "deg"
    hdu.header['cunit2'] = "deg"
    hdu.header['LONPOLE'] = 180
    hdu.header['LATPOLE'] = crval2
    hdu.header['RADESYS'] = 'ICRS'
    hdu.header['MJDREF'] = 0.0
    del hdu.header['EXTEND'] 
    hdu.writeto(outname+'.fits', overwrite='True')   

    print('Rotation angle is: ', crota2)
    print('Scale is: ', cdelt)
    