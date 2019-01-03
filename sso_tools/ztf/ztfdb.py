import numpy as np
import pandas as pd
import mysql.connector as mariadb
from astropy.time import Time
import matplotlib.pyplot as plt
import lsst.sims.movingObjects as mo
from .objectInfo import queryJPL

__all__ = ['fetch_alert_data', 'read_alert_datafile',
           'identify_candidates', 'get_obj',
           '_obj_obs', '_add_ztf_magcorr', '_add_oorb_magcorr', '_translate_df',
           'ztfname_to_designation',
           'check_astrometry', 'vis_psf_ap_photometry']

# ZTF filters / integer id's
filterdict = {1: 'g', 2: 'r', 3: 'i'}
filterdict_inv = {'g': 1, 'r': 2, 'i': 3}
filtercolors = {1: 'g', 2: 'r', 3: 'y'}

pyOrb = mo.PyOrbEphemerides()


def fetch_alert_data(jd_start=None):
    """Query alert database for all known SSO entries.

    Parameters
    ----------
    jd_start: float, opt
        Starting JD to query. If not specified, then 6/18/2018 will be used (time of the photometry update).

    Returns
    -------
    pd.DataFrame
        The alert observations of all objects.
    """
    # Connect to database
    con = mariadb.connect(user='ztf', database='ztf')
    # These are the time of various updates to the alert database.
    """
    # first attribution fix
    sso_alert_fix_date1 = Time('2018-05-16T23:30:00', format='isot', scale='utc')
    # second attribution fix
    sso_alert_fix_date2 = Time('2018-06-08T23:30:00', format='isot', scale='utc')
    """
    # photometry fix date
    sso_alert_phot_fix_date = Time('2018-06-18T23:30:00', format='isot', scale='utc')

    if jd_start is None:
        jd_start = sso_alert_phot_fix_date.jd

    query = 'select * from alerts where ssdistnr >=0 and jd > %f' % (jd_start)
    print('Querying from ZTF alert database: %s' % query)
    all_sso = pd.read_sql_query(query, con)
    # Add more easily readable date of observation
    obsdate = Time(all_sso['jd'], format='jd', scale='tai')
    all_sso['obsdate'] = obsdate.isot
    return all_sso


def read_alert_datafile(filename):
    all_sso = pd.read_csv(filename, low_memory=False)
    return all_sso


def identify_candidates(all_sso, min_obs=40, dist_cutoff=10):
    """Identify the objects which might be lightcurve determination candidates.

    Parameters
    ----------
    all_sso: pd.DataFrame
        The alert observations of the SSOs.
    min_obs: int, opt
        Minimum number of observations to consider an object a candidate for lightcurve determination.
        Default 40.
    dist_cutoff: int, opt
        The maximum ssdistnr value in the alert == arcseconds between measured position and expected position.
        Default 10.

    Returns
    -------
    list of str
        List of the object names.
    """
    # Pull out the list of objects with many observations, within @dist_cutoff of the attributed sso.
    objs = all_sso.query('ssdistnr < @dist_cutoff').groupby('ssnamenr')[['jd']].count().query('jd > %d'
                                                                                              % min_obs)
    names = objs.sort_values('jd', ascending=False)
    objnames = names.index.values
    return objnames


def get_obj(all_sso, ztfname, minJD=None, maxJD=None, magcol='magpsf', pred='oo'):
    """Pull out the observations of a given object.

    Parameters
    ----------
    all_sso: pd.DataFrame
        DataFrame with observations of all objects.
    ztfname: str
        Name (number or designation) of object to pull out of all_sso
    minJD: float, opt
        Minimum JD to pull out of DataFrame. Default None == no minimum.
    maxJD: float, opt
        Maximum JD to pull out of DataFrame. Default None == no maximum.
    magcol: str, opt
        Name of the dataframe column to use for the original magnitudes. Default magpsf.

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        DataFrame containing the object observations, all columns,
        DataFrame containing the minimal columns for lc_utils.
    """
    obj = _obj_obs(all_sso, ztfname, minJD=minJD, maxJD=maxJD)
    obj = _add_ztf_magcorr(obj, magcol=magcol)
    obj = _add_oorb_magcorr(obj, magcol=magcol)
    df = _translate_df(obj, magcol=magcol, pred=pred)
    return obj, df


def _obj_obs(all_sso, ztfname, minJD=None, maxJD=None):
    """Pull out the observations of a given object.

    Parameters
    ----------
    all_sso: pd.DataFrame
        DataFrame with observations of all objects.
    ztfname: str
        Name (number or designation) of object to pull out of all_sso
    minJD: float, opt
        Minimum JD to pull out of DataFrame. Default None == no minimum.
    maxJD: float, opt
        Maximum JD to pull out of DataFrame. Default None == no maximum.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the object observations
    """
    obj = all_sso.query('ssnamenr == @ztfname')
    if minJD is not None:
        obj = obj.query('jd > @minJD')
    if maxJD is not None:
        obj = obj.query('jd < @maxJD')
    filterlist = obj.fid.unique()
    for f in filterlist:
        o = obj.query('fid == @f')
        print('Filter %s (%s) has %d observations' % (f, filterdict[f], len(o)))
    return obj


def _add_ztf_magcorr(obj, magcol='magpsf'):
    """Add the corrected (for distance and phase angle) magnitudes.

    Parameters
    ----------
    obj: pd.DataFrame
        DataFrame with observations of this object.
    magcol: str, opt
        Name of the dataframe column to use for the original magnitudes. Default magpsf.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the object observations, plus a corrected magnitude.
    """
    if magcol not in ['magpsf', 'magap']:
        raise ValueError('magcol %s should be either magpsf or magap.' % magcol)
    # Add the 'corrected' magnitude values using ZTF values.
    magcorr = obj[magcol].values - obj.ssmagnr.values
    return obj.assign(magcorrZTF=magcorr)


def _add_oorb_magcorr(obj, magcol='magpsf'):
    """Add the corrected (for distance and phase angle) magnitudes.

    Parameters
    ----------
    obj: pd.DataFrame
        DataFrame with observations of this object.
    magcol: str, opt
        Name of the dataframe column to use for the original magnitudes. Default magpsf.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the object observations, plus a corrected magnitude.
    """
    if magcol not in ['magpsf', 'magap']:
        raise ValueError('magcol %s should be either magpsf or magap.' % magcol)
    # Add the 'corrected' magnitude values using OOrb predicted values.
    designation = ztfname_to_designation(obj['ssnamenr'].iloc[0])
    orbit = queryJPL(designation)
    orb = mo.Orbits()
    orb.setOrbits(orbit)
    pyOrb.setOrbits(orb)
    ephs = pyOrb.generateEphemerides(obj['jd'] - 2400000.5, obscode='I41')
    # The predicted magnitudes from PyOrb can be way off the reported ZTF mags.
    # Not sure why -- zeropoints? filters? (?) -- but add a correction.
    predmag = ephs[0]['magV']
    midpoint = np.where(obj['jd'] == np.median(obj['jd']))[0]
    predmag += obj[magcol].values[midpoint] - predmag[midpoint]
    magcorr = obj[magcol].values - predmag
    return obj.assign(magOO=predmag, magcorrOO=magcorr,
                      phaseangle=ephs[0]['phase'],
                      heliodist=ephs[0]['helio_dist'],
                      geodist=ephs[0]['geo_dist'],
                      velocity=ephs[0]['velocity'])


def _translate_df(obj, magcol='magpsf', pred='oo'):
    """Translate object observation DataFrame into the columns/format for lc_utils code.

    Parameters
    ----------
    obj: pd.DataFrame
        DataFrame with object observations.
    magcol: str, opt
        Name of the dataframe column to use for the original magnitudes. Default magpsf.

    Returns
    -------
    pd.DataFrame
    """
    if magcol not in ['magpsf', 'magap']:
        raise ValueError('magcol %s should be either magpsf or magap.' % magcol)
    pred = pred.lower()
    if pred not in ['oo', 'ztf']:
        raise ValueError('pred %s should be either oo or ztf.' % pred)
    if magcol == 'magpsf':
        sigmamag = 'sigmapsf'
    if magcol == 'magap':
        sigmamag = 'sigmagap'
    if pred == 'oo':
        predmagcol = 'magOO'
        magcorrcol = 'magcorrOO'
    else:
        predmagcol = 'ssmagnr'
        magcorrcol = 'magcorrZTF'
    newcols = ['objId', 'jd', 'fid', 'mag', 'sigmamag', 'predmag', 'magcorr', 'night', 'phaseangle']
    oldcols = ['ssnamenr', 'jd', 'fid', magcol, sigmamag, predmagcol, magcorrcol, 'nid', 'phaseangle']
    df = obj[oldcols]
    df.columns = newcols
    return df


def ztfname_to_designation(ztfname):
    """Translate the ZTF name (no spaces, includes 0's) into an MPC style designation.

    Parameters
    ----------
    ztfname: str

    Returns
    -------
    str
    """
    # Is it a pure number? if so, no conversion needed.
    try:
        desig = int(ztfname)
    # Otherwise some conversion required. Perhaps could get away with only last bit of this?
    except ValueError:
        if ztfname[1] == '/':
            i = 6
        else:
            i = 4
        desig = ztfname[0:i] + ' ' + ztfname[i:i + 2]
        i += 2
        val = int(ztfname[i:])
        if val > 0:
            desig += str(int(ztfname[i:]))
    return desig


def check_astrometry(obj):
    """Plot astrometric residuals of the object.

    Parameters
    ----------
    obj: pd.DataFrame
        The dataframe of all observations of a given object.

    Returns
    -------
    plt.figure
    """
    # Check astrometry residuals?
    fig = plt.figure(figsize=(8, 8))
    plt.plot(obj.jd, obj.ssdistnr, 'k.')
    plt.figtext(0.2, 0.8, 'Nobs = %d' % (len(obj)))
    plt.xlabel('JD', fontsize='x-large')
    plt.ylabel('ssdistnr (arcsec)', fontsize='x-large')
    plt.title(name)
    return fig


def vis_psf_ap_photometry(obj, fulldates=False, fullEph=True):
    """Plot the unphased, uncorrected photometry (PSF and aperture).

    Parameters
    ----------
    obj: pd.DataFrame
        The dataframe of all observations of a given object.
    fulldates: bool, opt
        Plot the full dates (True) or just the change in dates after the start (False).

    Returns
    -------
    plt.figure
    """
    ephMags = None
    if fullEph:
        try:
            designation = ztfname_to_designation(obj['ssnamenr'].iloc[0])
            orbit = queryJPL(designation)
            orb = mo.Orbits()
            orb.setOrbits(orbit)
            pyOrb.setOrbits(orb)
            t = np.arange(obj.jd.min(), obj.jd.max()+0.5, 1)
            ephs = pyOrb.generateEphemerides(t - 2400000.5, obscode='I41')
            # match near midpoint of observations
            dt = abs(t - np.median(obj.jd))
            midpointEphs = np.where(dt == min(dt))[0]
            offset = ephs[0]['magV'][midpointEphs] - obj.magpsf.median()
            print(ephs[0]['magV'][midpointEphs], obj.magpsf.median(), offset)
            ephMags = ephs[0]['magV'] - offset
            if ~fulldates:
                t = t - obj.jd.iloc[0]
        except:
            raise
    fig = plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    filterlist = obj.fid.unique()
    for f in filterlist:
        o = obj.query('fid == @f')
        if fulldates:
            times = o.jd
        else:
            times = o.jd - obj.jd.iloc[0]
        plt.errorbar(times, o.magap, yerr=o.sigmagap, color=filtercolors[f],
                     marker='.', linestyle='')
    if fulldates:
        times = obj.jd
    else:
        times = obj.jd - obj.jd.iloc[0]
    plt.plot(times, obj.ssmagnr, 'c-')
    plt.plot(times, obj.magOO, 'r-')
    if ephMags is not None:
        plt.plot(t, ephMags, 'r:')
    plt.figtext(0.15, 0.8, 'Nobs = %d, Nnights = %d' % (len(obj), len(obj.nid.unique())))
    plt.xticks(rotation=90)
    if fulldates:
        plt.xlabel('JD', fontsize='x-large')
    else:
        plt.xlabel('Delta JD', fontsize='x-large')
    plt.ylabel('ApMag', fontsize='x-large')
    plt.title(obj.ssnamenr.unique()[0])
    plt.subplot(1, 2, 2)
    for f in filterlist:
        o = obj.query('fid == @f')
        if fulldates:
            times = o.jd
        else:
            times = o.jd - obj.jd.iloc[0]
        plt.errorbar(times, o.magpsf, yerr=o.sigmapsf, color=filtercolors[f],
                     marker='.', linestyle='')
    if fulldates:
        times = obj.jd
    else:
        times = obj.jd - obj.jd.iloc[0]
    plt.plot(times, obj.ssmagnr, 'c-')
    plt.plot(times, obj.magOO, 'r-')
    if ephMags is not None:
        plt.plot(t, ephMags, 'r:')
    plt.figtext(0.58, 0.8, 'Nobs = %d, Nnights = %d' % (len(obj), len(obj.nid.unique())))
    plt.xticks(rotation=90)
    if fulldates:
        plt.xlabel('JD', fontsize='x-large')
    else:
        plt.xlabel('Delta JD', fontsize='x-large')
    plt.ylabel('PSF mag', fontsize='x-large')
    plt.title(obj.ssnamenr.unique()[0])
    if len(filterlist) > 1:
        colorname = '%s-%s' % (filterdict[filterlist[0]], filterdict[filterlist[1]])
        color = (obj.query('fid == @filterlist[0]').magpsf.mean()
                 - obj.query('fid == @filterlist[1]').magpsf.mean())
        print('Average color %s = %.2f' % (colorname, color))
    return fig

