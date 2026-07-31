from dataclasses import dataclass
import pysiaf
from typing import Union
import copy

def gaiastars(ra, dec, radius=1.0, star=0.7, maglim=16.5, obs_time='2026-10-31T00:00:00'):
    """
    Inputs
    ------
    ra (float): Right ascension of the field center
    dec (float): Declination of the field center
    radius (float): Search radius in degrees
    star (float): Threshold to select starlike sources
    maglim (float): Limit in brightness of the accepted sources
    obs_time (float): Time of the observation

    Output
    ------
    Catalog of GAIA sources

    Description
    -----------
    To use this function, select at least ra,dec of the intended field:

        >>> gaia_catalog = gaiastars(ra=30, dec=-45, radius=1.0)
    
    The function will return a catalog of GAIA stars in the Roman I-Sim format.
    Any sources without fluxes or positions is filtered out.
    """
    from astroquery.gaia import Gaia
    from romanisim import gaia, bandpass
    from astropy.time import Time
    import numpy as np

    query = f'SELECT * FROM gaiadr3.gaia_source WHERE distance({ra}, {dec}, ra, dec) < {radius}'
    job = Gaia.launch_job_async(query)
    result = job.get_results()

    result = result[result['classprob_dsc_combmod_star'] >= star]
    result = result[result['phot_g_mean_mag'] < maglim]

    # Make the Roman I-Sim formatted catalog
    gaia_catalog=gaia.gaia2romanisimcat(
        result,Time(obs_time),fluxfields=set(bandpass.galsim2roman_bandpass.values()))


    # Reject anything with missing fluxes or positions
    names = [f for f in gaia_catalog.dtype.names if f[0] == 'F']
    names += ['ra', 'dec']

    bad = np.zeros(len(gaia_catalog), dtype='bool')
    for b in names:
          bad = ~np.isfinite(gaia_catalog[b])
          if hasattr(gaia_catalog[b], 'mask'):
               bad |= gaia_catalog[b].mask
          gaia_catalog = gaia_catalog[~bad]
    
    return gaia_catalog

def plotstars(catalog, color='blue', ms=3, fig=None, ax=None, size=8, alpha=1):
    """
    Inputs
    ------

    catalog: romanisim catalog of stars
    color: color of the dots representing stars
    fig: matplotlib figure if already defined
    ax:  matplotlib ax if already defined
    size: size of the square plot

    Outputs
    -------

    fig: matplotlib figure
    ax: matplotlib ax

    Description
    -----------

    The function plots stars from a RomanISim catalog (simulated or GAIA) as dots.

        >> fig, ax = plotstars(catalog, color='red', size=10)

    If a figure is already defined, one can overplot the starts by passing the fig and ax arguments:

        >> plotstars(catalog, fig=fig, ax=ax)
    """
    # Matplotlib plot style
    import matplotlib.pyplot as plt
    from astropy.visualization.wcsaxes import WCSAxes
    from astropy.wcs import WCS
    import numpy as np
    
    if fig is None and ax is None:
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.titlesize'] = 14
        cra = np.median(catalog['ra'].value)
        cdec = np.median(catalog['dec'].value)
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1, 1]
        wcs.wcs.cdelt = np.array([-1/360,1/360])
        wcs.wcs.crval = [cra, cdec]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        fig = plt.figure(figsize=[size, size])
        ax = fig.add_subplot(111, projection=wcs)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('R.A. (J2000)')
        ax.set_ylabel('Dec. (J2000)')
        ax.grid()


    if isinstance(ax, WCSAxes):
        ax.scatter(catalog['ra'].value, catalog['dec'].value, s=ms, facecolors='none', 
                   edgecolors=color, linewidths=0.5, alpha=alpha, transform=ax.get_transform('icrs'))
    else:
        ax.scatter(catalog['ra'].value, catalog['dec'].value,'.', s=ms,facecolors='none', 
                   edgecolors=color, alpha=alpha, linewidths=0.5)

    return fig, ax


def plotgalaxies(catalog, color='red', filter='F106', fig=None, ax=None, size=8, magnify=1):

    """
    Inputs
    ------

    catalog: romanisim catalog of stars
    color: color of the dots representing stars
    filter: selected filter in the catalog
    fig: matplotlib figure if already defined
    ax:  matplotlib ax if already defined
    size: size of the square plot

    Outputs
    -------

    fig: matplotlib figure
    ax: matplotlib ax

    Description
    -----------

    The function plots galaxies from a RomanISim catalog (simulated or GAIA) as dots.

        >> fig, ax = plotgalaxies(catalog, color='red', size=10)

    If a figure is already defined, one can overplot the starts by passing the fig and ax arguments:

        >> plotstars(catalog, fig=fig, ax=ax)
    """
    # Matplotlib plot style
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from astropy.visualization.wcsaxes import WCSAxes
    from astropy.wcs import WCS
    import numpy as np
    from matplotlib.collections import EllipseCollection

    
    pa = catalog['pa']
    ba = catalog['ba']
    hlr = catalog['half_light_radius']
    flux = catalog[filter]*1.e6

    hlr = hlr/ np.nanmax(hlr)
    ra_, dec_ = catalog['ra'].value, catalog['dec'].value

    if fig is None and ax is None:
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.titlesize'] = 14
        cra = np.median(catalog['ra'].value)
        cdec = np.median(catalog['dec'].value)
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1, 1]
        wcs.wcs.cdelt = np.array([-1/360,1/360])
        wcs.wcs.crval = [cra, cdec]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        fig = plt.figure(figsize=[size, size])
        ax = fig.add_subplot(111, projection=wcs)
        ax.set_xlabel('R.A. (J2000)')
        ax.set_ylabel('Dec. (J2000)')


    xy = np.column_stack((ra_,dec_))
    a = hlr*20*magnify
    b = ba * a
    if isinstance(ax, WCSAxes):
        ec = EllipseCollection(a, b, pa, units='xy', offsets=xy,
                               offset_transform=ax.get_transform('icrs'), cmap='viridis',transform=ax.get_transform('icrs'))
    else:
        ec = EllipseCollection(a, b, pa, units='xy', offsets=xy,
                               offset_transform=ax.transData, cmap='viridis')
        
    ec.set_array(flux)
    ax.add_collection(ec)
    ax.set_aspect('equal', 'box')
    ax.autoscale_view()
    ax.grid()
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cbar = plt.colorbar(ec, cax=cax)
    cbar.set_label('Flux')
    return fig, ax
    
@dataclass(init=True, repr=True)
class PointWFI:
    """
    Inputs
    ------
    ra (float): Right ascension of the target placed at the geometric 
                center of the Wide Field Instrument (WFI) focal plane
                array. This has units of degrees.
    dec (float): Declination of the target placed at the geometric
                 center of the WFI focal plane array. This has units
                 of degrees.
    position_angle (float): Position angle of the WFI relative to the V3 axis
                            measured from North to East. A value of 0.0 degrees
                            would place the WFI in the "smiley face" orientation
                            (U-shaped) on the celestial sphere. To place WFI
                            such that the position angle of the V3 axis is 
                            zero degrees, use a WFI position angle of -60 degrees.

    Description
    -----------
    To use this class, instantiate it with your initial pointing like so:

        >>> point = PointWFI(ra=30, dec=-45, position_angle=10)
    
    and then dither using the dither method:

        >>> point.dither(x_offset=10, y_offset=140)

    This would shift the WFI 10 arcseconds along the X-axis of the WFI
    and 140 arcseconds along the Y-axis of the WFI. These axes are in the ideal
    coordinate system of the WFI, i.e, with the WFI oriented in a U-shape with 
    +x to the right and +y up. You can pull the new pointing info out of the object 
    either as attributes or by just printing the object:

        >>> print(point.ra, point.dec)
        >>> 29.95536280064078 -44.977122003232786

    or

        >>> point
        >>> PointWFI(ra=29.95536280064078, dec=-44.977122003232786, position_angle=10)
    """
    # Set default pointing parameters
    ra: float = 0.0
    dec: float = 0.0
    position_angle: float = -60.0

    # Post init method sets some other defaults and initializes
    # the attitude matrix using PySIAF.
    def __post_init__(self) -> None:
        self.siaf_aperture = pysiaf.Siaf('Roman')['WFI_CEN']
        self.v2_ref = self.siaf_aperture.V2Ref
        self.v3_ref = self.siaf_aperture.V3Ref
        self.attitude_matrix = pysiaf.utils.rotations.attitude(self.v2_ref, self.v3_ref, self.ra,
                                        self.dec, self.position_angle)
        self.siaf_aperture.set_attitude_matrix(self.attitude_matrix)

        # Compute the V3 position angle
        self.tel_roll = pysiaf.utils.rotations.posangle(self.attitude_matrix, 0, 0)

        # Save initial pointing
        self.att0 = self.attitude_matrix.copy()

        # Save a copy of the input RA and Dec in case someone needs it
        self.ra0 = copy.copy(self.ra)
        self.dec0 = copy.copy(self.dec)

    def dither(self, x_offset: Union[int, float],
               y_offset: Union[int, float]) -> None:
        """
        Purpose
        -------
        Take in an ideal X and Y offset in arcseconds and shift the telescope
        pointing to that position.

        Inputs
        ------
        x_offset (float): The offset in arcseconds in the ideal X direction.

        y_offset (float): The offset in arcseconds in the ideal Y direction.
        """

        self.ra, self.dec = self.siaf_aperture.idl_to_sky(x_offset, y_offset)
        

def plotWFI(pointing, color='blue', fig=None, ax=None, label=False, size=8, wfi=None):
    """
    Inputs
    ------

    pointing: pointing of detector defined with class PointWFI
    color: color of the detector
    fig: matplotlib figure if already defined
    ax:  matplotlib ax if already defined
    label: boolean variable to print the label of each detector

    Outputs
    -------

    fig: matplotlib figure
    ax: matplotlib ax

    Description
    -----------

    Overplots the WFI detectors on an image in RA, Dec coordinates.

        >> plotWFI(pointing, color='red', fig=fig, ax=ax)
    
    """
    from matplotlib.collections import PolyCollection
    from astropy.wcs import WCS
    from astropy.visualization.wcsaxes import WCSAxes
    import numpy as np
    import matplotlib.pyplot as plt

    # List of detectors
    if wfi is None:
        apertureNames = ['WFI01_FULL', 'WFI02_FULL', 'WFI03_FULL', 'WFI04_FULL', 'WFI05_FULL', 'WFI06_FULL',
                         'WFI07_FULL', 'WFI08_FULL', 'WFI09_FULL', 'WFI10_FULL', 'WFI11_FULL', 'WFI12_FULL',
                         'WFI13_FULL', 'WFI14_FULL', 'WFI15_FULL', 'WFI16_FULL', 'WFI17_FULL', 'WFI18_FULL']
    else:
        apertureNames = []
        for wfi_ in wfi:
            apertureNames.append('WFI'+'{0:02d}'.format(wfi_)+'_FULL')
    telescopeSiaf = pysiaf.Siaf('roman')
    apertureList = []
    for name in apertureNames:
        apertureList.append(telescopeSiaf[name])

    if (fig is None) and (ax is None):
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.titlesize'] = 14
        cra = np.median(pointing.ra)
        cdec = np.median(pointing.dec)
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1, 1]
        wcs.wcs.cdelt = np.array([-1/360,1/360])
        wcs.wcs.crval = [cra, cdec]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        fig = plt.figure( figsize=[size, size])
        if np.abs(pointing.dec) < 4:
            ax = fig.add_subplot(111, projection='polar')
        else:
            ax = fig.add_subplot(111, projection=wcs)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('R.A. (J2000)')
        ax.set_ylabel('Dec. (J2000)')
        ax.grid()
        
    verts = []
    # Updating the attitude matrix in case the reference coordinates changed because of dithering
    attmat = pysiaf.utils.rotations.attitude(pointing.v2_ref, pointing.v3_ref, pointing.ra,
                                        pointing.dec, pointing.position_angle)
    for n, apertureSiaf in enumerate(apertureList):
        apertureSiaf.set_attitude_matrix(attmat)
        xVertices = np.array([apertureSiaf.XIdlVert1,apertureSiaf.XIdlVert2,apertureSiaf.XIdlVert3,apertureSiaf.XIdlVert4])        
        yVertices = np.array([apertureSiaf.YIdlVert1,apertureSiaf.YIdlVert2,apertureSiaf.YIdlVert3,apertureSiaf.YIdlVert4])        
        skyRa, skyDec = apertureSiaf.idl_to_sky(xVertices, yVertices)
        verts.append(list(zip(skyRa, skyDec)))
        if label:
            center = [np.median(skyRa), np.median(skyDec)]
            if isinstance(ax, WCSAxes):
                ax.text(center[0], center[1], str(n+1), ha='center', va='center', color=color,transform=ax.get_transform('icrs'))
            else:
                ax.text(center[0], center[1], str(n+1), ha='center', va='center', color=color)
    if isinstance(ax, WCSAxes):
        poly = PolyCollection(verts, facecolor="none", edgecolor=color, transform=ax.get_transform('icrs'))
    else:
        poly = PolyCollection(verts, facecolor="none", edgecolor=color)
    ax.add_collection(poly)
    ax.autoscale_view()

    return fig, ax


def run_romanisim(catalog, ra=80.0, dec=30.0,angle=0, obs_date = '2026-10-31T00:00:00', sca=1, expnum=None, optical_element='F106', 
                  ma_table_number=3, level=2, filename=f'r0003201001001001004', seed=5346):
    """
    Inputs
    ------

    catalog: catalog of sources
    ra: RA of WFI field center
    dec: Dec of WFI field center
    obs_date: date of observation
    sca: detector number (sensor chip assembly)
    expnum: number of exposures
    optical_element: filter used for observation
    ma_table_number: Multi-accumulation table used
    level: product level (1=ramps, 2=images)
    filename: name of the file
    seed: seed used for pseudo-random number generator

    Outputs
    -------

    None
    
    Description
    -----------

    Run romanisim to generate products (level 1 or 2) for a detector, a filter, and the WFI centerfield position.
    
    """

    import argparse
    import galsim
    from romanisim.models import parameters
    from romanisim import wcs, persistence, ris_make_utils as ris
    from astropy.coordinates import SkyCoord
    import warnings
    import logging
    
    # Ignore warnings
    warnings.filterwarnings('ignore')
    # Ignore INFO
    logging.getLogger().setLevel(logging.CRITICAL)
    
    match level:
        case 1:
            cal_level = 'uncal'
        case 2:
            cal_level = 'cal'
        case _:
            print('Please select either 1 or 2 for level')
            exit
    if expnum is None:
        filename = f'{filename}_wfi{sca:02d}_{optical_element.lower()}_{cal_level}.asdf'
    else:
        filename = f'{filename}_{expnum:04d}_wfi{sca:02d}_{optical_element.lower()}_{cal_level}.asdf'

    # Set other arguments for use in Roman I-Sim. 

    parser = argparse.ArgumentParser()
    parser.set_defaults(usecrds=True, level=level, filename=filename, 
                        drop_extra_dq=True, sca=sca, bandpass=optical_element,
                        pretend_spectral=None, psftype='stpsf')
    args = parser.parse_args([])


    # Set reference files to None for CRDS
    for k in parameters.reference_data:
        parameters.reference_data[k] = None

    # Set Galsim RNG object
    rng = galsim.UniformDeviate(seed)

    # Set default persistance information
    persist = persistence.Persistence()

    # Set metadata
    metadata = ris.set_metadata(date=obs_date, bandpass=optical_element, sca=sca, ma_table_number=ma_table_number, usecrds=True)

    # Update the WCS info
    # PA - 60, since the WFI is oriented  at -60 degs wrt Y3
    wcs.fill_in_parameters(metadata, SkyCoord(ra, dec, unit='deg', frame='icrs'), boresight=False, pa_aper=angle)

    # Run the simulation
    sim_result = ris.simulate_image_file(args, metadata, catalog, rng, persist)

    # Clean up the memory
    del sim_result


def rotatecoords(catalog, center, angle):
    """
    Inputs
    ------

    catalog: a roman-i-sim catalog
    center: the center of rotation (ra,dec) in degrees
    angle:  the angle used for rotation in degrees
    
    Outputs
    -------

    None.
    The coordinates of the input catalog are modified.
    
    Description
    -----------
    Rotate the coordinates in the input catalog by angle around the direction center.
    The code implements the Rodrigues' rotation formula (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Derivation)
    """
    import numpy as np
    ra0, dec0 = center
    phi = catalog['ra'].value * np.pi/180
    theta = (90 - catalog['dec'].value) * np.pi/180
    ax = np.sin(theta) * np.cos(phi)
    ay = np.sin(theta) * np.sin(phi)
    az = np.cos(theta)
    Phi = ra0 * np.pi/180
    Theta = (90 - dec0) * np.pi/180
    kx = np.sin(Theta) * np.cos(Phi)
    ky = np.sin(Theta) * np.sin(Phi)
    kz = np.cos(Theta)
    beta = angle * np.pi/180
    cbeta = np.cos(beta)
    sbeta = np.sin(beta)
    ka = kx * ax + ky * ay + kz * az
    Ax = ax * cbeta + (ky * az - kz * ay) * sbeta + ka * kx * (1-cbeta)
    Ay = ay * cbeta + (kz * ax - kx * az) * sbeta + ka * ky * (1-cbeta)
    Az = az * cbeta + (kx * ay - ky * ax) * sbeta + ka * kz * (1-cbeta)
    phi = np.atan2(Ay,Ax) * 180/np.pi
    theta = 90 - np.atan2(np.sqrt(Ax*Ax+Ay*Ay), Az)*180/np.pi
    catalog['ra'] = phi
    catalog['dec'] = theta
