"""Add information about a particular object - orbit, known lightcurve, and predicted magnitudes."""

import numpy as np
import pandas as pd
from astroquery.jplsbdb import SBDB

__all__ = ['queryJPL', 'queryJPL_many']


def queryJPL(designation):
    """Query JPL Horizons for information about object 'designation'.

    Parameters
    ----------
    designation: str
        A name for the object that JPL Horizons will understand. (try ztfname_to_designation for ssnamenr).

    Returns
    -------
    pd.Series
        Series containing orbit and select physical information.
    """
    sbdb = SBDB.query(designation, phys=True)
    orbval = sbdb['orbit']['elements']
    phys =  sbdb['phys_par']
    if 'diameter' in phys:
        diam = phys['diameter'].to('km')
    else:
        diam = np.NaN
    if 'albedo' in phys:
        albedo = float(phys['albedo'])
    else:
        albedo = np.NaN
    if 'rot_per' in phys:
        rot = phys['rot_per'].to('hr')
    else:
        rot = np.NaN
    orbit = pd.Series(data={'objId': designation,
                            'FORMAT': 'COM',
                            'a': orbval['a'].to('AU'),
                            'q': orbval['q'].to('AU'),
                            'e': float(orbval['e']),
                            'inc': orbval['i'].to('deg'),
                            'Omega': orbval['om'].to('deg'),
                            'argPeri': orbval['w'].to('deg'),
                            'tPeri': orbval['tp'].to('d') - 2400000.5,  # to MJD
                            'meanAnomaly': orbval['ma'].to('deg'),
                            'epoch': sbdb['orbit']['epoch'].to('d') - 2400000.5,  # to MJD
                            'H': float(sbdb['phys_par']['H']),
                            'g': 0.15,
                            'diam': diam,
                            'albedo': albedo,
                            'rot': rot
                            })
    return orbit


def queryJPL_many(designations):
    orbits = []
    for desig in designations:
        orbits.append(queryJPL(desig))
    orbits = pd.DataFrame(orbits)
    return orbits



