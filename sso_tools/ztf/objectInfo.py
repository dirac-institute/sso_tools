"""Add information about a particular object - orbit, known lightcurve, and predicted magnitudes."""

import numpy as np
import pandas as pd
from astroquery.jplsbdb import SBDB
from ..catalogs import read_sdss_moc

__all__ = ['queryJPL', 'queryJPL_many', 'sdss_colors']


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
    orbit = pd.Series(data={'name': designation,
                            'jpl_des': sbdb['object']['des'],
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


def sdss_colors(orbits, sdss_moc_filename='ADR4.dat'):
    sdss = read_sdss_moc(sdss_moc_filename)
    sdss = sdss.query('Name != "-"')
    jpl_des = np.zeros(len(sdss), dtype=object)
    for c, name in enumerate(sdss.Name.values):
        sbdb = SBDB.query(name.replace('_', ' '))
        if 'object' in sbdb:
            jpl_des[c] = sbdb['object']['des']
    sdss = sdss.assign(jpl_des = jpl_des)
    sdsscolors = {}
    bands = ('u', 'g', 'r', 'i', 'z', 'sdssa')
    errs = ('uerr', 'gerr', 'rerr', 'ierr', 'zerr', 'aerr')
    for band in bands:
        sdsscolors[band] = np.zeros(len(data)) - 999
    for err in errs:
        sdsscolors[err] = np.zeros(len(data)) - 999
    for c, des in enumerate(orbits.jpl_des.values):
        tmp = sdss.query('jpl_des == @des')
        if len(tmp) > 0:
            for band, err in zip(bands, errs):
                mags = sdss[des][band].values
                errors = sdss[des][err].values
                good = np.where((mags < 40) & (errors < 2))
                if len(good[0]) > 0:
                    mag = np.mean(mags[good])
                    errval = np.sqrt(np.sum(errors[good] ** 2))
                    sdsscolors[band][c] = mag
                    sdsscolors[err][c] = errval
    colors = pd.DataFrame({'jpl_des': orbits.jpl_des.values,
                           'sdssa': sdsscolors['sdssa'], 'aerr': sdsscolors['aerr'],
                           'u': sdsscolors['u'], 'uerr': sdsscolors['uerr'],
                           'g': sdsscolors['g'], 'gerr': sdsscolors['gerr'],
                           'i': sdsscolors['i'], 'ierr': sdsscolors['ierr'],
                           'z': sdsscolors['z'], 'zerr': sdsscolors['zerr']})
    return colors
