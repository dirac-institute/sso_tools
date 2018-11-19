import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mysql.connector as mariadb
from astropy.time import Time
from gatspy import periodic

# TODO : make the lightcurve code sections more generic and move ZTF specific items into "ZTF"

__all__ = ['fetch_alert_data', 'identify_candidates', 'obj_obs', 'check_astrometry',
           'vis_photometry', 'vis_corr_photometry', 'fit_model', 'make_periodogram',
           'make_predictions', 'plot_unphased', 'phase_times', 'plot_phased']


# ZTF filters / integer id's
filterdict = {1: 'g', 2: 'r', 3: 'i'}
filterdict_inv = {'g': 1, 'r': 2, 'i': 3}


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
    sso_alert_fix_date1 = Time('2018-05-16T23:30:00', format='isot', scale='utc')  # first attribution fix
    sso_alert_fix_date2 = Time('2018-06-08T23:30:00', format='isot', scale='utc')  # second attribution fix
    sso_alert_phot_fix_date = Time('2018-06-18T23:30:00', format='isot', scale='utc')  # photometry fix date

    if jd_start is None:
        jd_start = sso_alert_phot_fix_date

    query = 'select * from alerts where ssdistnr >=0 and jd > %f' % (jd_start)
    print('Querying from ZTF alert database: %s' % query)
    all_sso = pd.read_sql_query(query, con)
    # Add more easily readable date of observation
    obsdate = Time(all_sso['jd'], format='jd', scale='tai')
    all_sso['obsdate'] = obsdate.isot
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


def obj_obs(all_sso, name, minJD=None, maxJD=None):
    """Pull out the observations of a given object, adding the distance/phase corrected magnitude.

    Parameters
    ----------
    all_sso: pd.DataFrame
        Dataframe with observations of all objects.
    name: str
        Name (number or designation) of object to pull out of all_sso
    minJD: float, opt
        Minimum JD to pull out of DataFrame. Default None == no minimum.
    maxJD: float, opt
        Maximum JD to pull out of DataFrame. Default None == no maximum.

    Returns
    -------
    pd.DataFrame, dictionary of pd.DataFrame(s)
        DataFrame containing the object observations, including corrected magnitude.
        Dictionary of DataFrames containing observations separated by filter.
    """
    obj = all_sso.query('ssnamenr == @name')
    if minJD is not None:
        obj = obj.query('jd > @minJD')
    if maxJD is not None:
        obj = obj.query('jd < @maxJD')
    # Add the 'corrected' magnitude values --- REPLACE WITH OORB PREDICTED VALUES
    magcorr = obj.magpsf.values - obj.ssmagnr.values
    obj['magcorr'] = magcorr
    # And pull out the observations in each filter.
    filterlist = obj.fid.unique()
    o = {}
    for f in filterlist:
        o[f] = obj.query('fid == @f')
        print('Filter %s (%s) has %d observations' % (f, filterdict[f], len(o[f])))
    obj.to_csv(name + '.csv')
    return obj, o


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


def vis_photometry(obj, o, fulldates=False):
    """Plot the unphased, uncorrected photometry (PSF and aperture).

    Parameters
    ----------
    obj: pd.DataFrame
        The dataframe of all observations of a given object.
    o: dictionary of pd.DataFrames
        Dictionary of dataframes of observations, separated by filter.
    fulldates: bool, opt
        Plot the full dates (True) or just the change in dates after the start (False).

    Returns
    -------
    plt.figure
    """
    fig = plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    for f in o:
        if fulldates:
            times = o[f].jd
        else:
            times = o[f].jd - obj.jd.iloc[0]
        plt.errorbar(times, o[f].magap, yerr=o[f].sigmagap, color=filterdict[f],
                     marker='.', linestyle='')
    if fulldates:
        times = obj.jd
    else:
        times = obj.jd - obj.jd.iloc[0]
    plt.plot(times, obj.ssmagnr)
    plt.figtext(0.15, 0.8, 'Nobs = %d, Nnights = %d' % (len(obj), len(obj.nid.unique())))
    plt.xticks(rotation=90)
    if fulldates:
        plt.xlabel('JD', fontsize='x-large')
    else:
        plt.xlabel('Delta JD', fontsize='x-large')
    plt.ylabel('ApMag', fontsize='x-large')
    plt.title(name)
    plt.subplot(1, 2, 2)
    for f in o:
        if fulldates:
            times = o[f].jd
        else:
            times = o[f].jd - obj.jd.iloc[0]
        plt.errorbar(times, o[f].magpsf, yerr=o[f].sigmapsf, color=filterdict[f],
                     marker='.', linestyle='')
    if fulldates:
        times = obj.jd
    else:
        times = obj.jd - obj.jd.iloc[0]
    plt.plot(times, obj.ssmagnr)
    plt.figtext(0.58, 0.8, 'Nobs = %d, Nnights = %d' % (len(obj), len(obj.nid.unique())))
    plt.xticks(rotation=90)
    if fulldates:
        plt.xlabel('JD', fontsize='x-large')
    else:
        plt.xlabel('Delta JD', fontsize='x-large')
    plt.ylabel('PSF mag', fontsize='x-large')
    plt.title(name)
    if '1' in o and '2' in o:
        objgr = o[filterdict_inv['g']].magpsf.mean() - o[filterdict_inv['r']].magpsf.mean()
        print('Average colors (g-r) %.2f' % (objgr))
    return fig


def vis_corr_photometry(obj, o):
    """Plot the unphased, corrected photometry.

    Parameters
    ----------
    obj: pd.DataFrame
        The dataframe of all observations of a given object.
    o: dictionary of pd.DataFrames
        Dictionary of dataframes of observations, separated by filter.

    Returns
    -------
    plt.figure
    """
    # Look at photometry after subtracting expected magnitude
    # (i.e. subtract an approximate phase curve, but the ZTF predicted values do have errors sometimes)
    fig = plt.figure(figsize=(8, 6))
    for f in o:
        plt.errorbar(o[f].jd - obj.jd.iloc[0], o[f].magcorr, yerr=o[f].sigmapsf, color=filterdict[f],
                     marker='.', linestyle='')
    plt.figtext(0.15, 0.8, 'Nobs = %d, Nnights = %d' % (len(obj), len(obj.nid.unique())))
    plt.xticks(rotation=90)
    plt.xlabel('delta JD', fontsize='x-large')
    plt.ylabel('MagCorr (magpsf - ssmagnr)', fontsize='x-large')
    plt.title(name)
    plt.grid(True, alpha=0.3)
    if '1' in o and '2' in o:
        objgr = o[filterdict_inv['g']].magcorr.mean() - o[filterdict_inv['r']].magcorr.mean()
        print('Average magcorr colors (g-r) %.2f' % (objgr))
    return fig


def fit_model(obj, Nterms_base=2, Nterms_band=1):
    """Fit the lightcurve data using periodic.LombScargleMultiband.

    Parameters
    ----------
    obj: pd.DataFrame
        The dataframe of all observations of a given object.
    Nterms_base: int, opt
        Number of terms for the base LS fit. Default 2.
    Nterms_band: int, opt
        Number of terms to allow between bandpasses. Default 1.

    Returns
    -------
    periodic.LombScargleMultiBand, float, list of floats
        The gatspy LS model fit, best fit period (days), and list of best fit periods (in days).
    """
    # Let's try to fit these values with the gatspy multiband fitter
    model = periodic.LombScargleMultiband(fit_period=True,
                                          Nterms_base=Nterms_base,
                                          Nterms_band=Nterms_band)
    model.optimizer.period_range = (2.0/24.0, 2.0)
    model.optimizer.first_pass_coverage = 100
    model.fit(obj.jd, obj.magcorr, obj.sigmapsf, obj.fid)
    top_periods = model.find_best_periods()
    print('Top Periods (and doubles):')
    print(' '.join(['%.3f (%.3f) hours\n' % (p*24.0, p*24.0*2) for p in top_periods if p < 1]),
          ' '.join(['%.3f (%.3f) days\n' % (p, p*2) for p in top_periods if p > 1]))
    print()
    print('Best fit period: %.3f hours' % (model.best_period * 24))
    return model, model.best_period, top_periods


def make_periodogram(model):
    """Plot the periodogram for the model (after fit).

    Parameters
    ----------
    model: periodic.LombScargleMultiBand model
        Gatspy model, already fit.

    Returns
    -------
    periods, power, plt.figure
        Periods and power from periodogram calculation, and figure.
    """
    # Look at the periodogram
    periods, power = model.periodogram_auto(oversampling=100, nyquist_factor=5)
    fig = plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(periods, power)
    for p in top_periods:
        plt.axvline(p, color='k', linestyle=':')
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.xlim(0, 3)
    plt.subplot(1, 2, 2)
    plt.plot(periods * 24, power)
    for p in top_periods:
        plt.axvline(p*24, color='k', linestyle=':')
    plt.xlabel('Period (hours)')
    plt.ylabel('Power')
    plt.xlim(2, 12)
    return model, top_periods, model.best_period, fig


def make_predictions(obj, period, model):
    """Generate predicted magnitudes at the times of the observations.

    Parameters
    ----------
    obj: pd.DataFrame
        Dataframe containing the observations of the object.
    period: float
        Period to use for the predictions.
    model: gatspy model
        Model to use for the predictions (periodic.LombScargleMultiBand, etc.).

    Returns
    -------
    np.array, np.array, np.array
        Time of the observations, Magnitude of the predicted observations in the first filter (g),
        and Magnitude of the predicted observations in the second filter (r)."""
    # Generate predicted data, using one of the periods.
    tfit = np.arange(obj.jd.min(), obj.jd.max(), 0.02)
    filts = np.ones(len(tfit))
    pred = model.predict(tfit, filts=filts, period=period)
    pred2 = model.predict(tfit, filts=filts+1, period=period)
    return tfit, pred, pred2


def plot_unphased(obj, o, period, model, name):
    # I'm not sure wth is going on with the predicted values here ...
    fig = plt.figure(figsize=(10, 8))
    tfit, pred, pred2 = make_predictions(obj, period, model)
    plt.plot(tfit, pred, color=filterdict[1], linestyle=':', alpha=0.2)
    plt.plot(tfit, pred2, color=filterdict[2], linestyle=':', alpha=0.2)
    for f in o:
        mag = o[f].magcorr
        if f == filterdict_inv['r']:
            mag = mag #+ objgr
        plt.errorbar(o[f].jd, mag, yerr=o[f].sigmapsf, color=filterdict[f],
                     marker='.', linestyle='')

    objgr = np.mean(pred2 - pred)
    objgr_2 = np.mean(o[filterdict_inv['g']].magcorr) - np.mean(o[filterdict_inv['r']].magcorr)
    plt.figtext(0.15, 0.8, 'Nobs = %d, Nnights = %d\n Period %.2f hours \n Mean object color %.2f (%.2f)'
                % (len(obj), len(obj.nid.unique()), period * 24., objgr, objgr_2))
    plt.xlabel('JD', fontsize='x-large')
    plt.ylabel('magcorr', fontsize='x-large')
    plt.title(name)
    return fig


def phase_times(times, period, offset=0):
    """Phase times by period."""
    phased = (times - offset) % period
    phased = phased / period
    return phased


def plot_phased(obj, o, period, model, name):
    # Phased
    tfit = np.arange(0, period, 0.002)
    filts = np.ones(len(tfit))
    pred = model.predict(tfit, filts=filts, period=period)
    pred2 = model.predict(tfit, filts=filts+1, period=period)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(phase_times(tfit, period), pred, color=filterdict[1], linestyle=':')
    plt.plot(phase_times(tfit, period), pred2, color=filterdict[2], linestyle=':')

    for f in o:
        mag = o[f].magcorr
        if f == filterdict_inv['r']:
            mag = mag
        plt.errorbar(phase_times(o[f].jd, period), mag, yerr=o[f].sigmapsf, color=filterdict[f],
                     marker='.', linestyle='')

    objgr = np.mean(pred2 - pred)
    objgr_2 = np.mean(o[filterdict_inv['g']].magcorr) - np.mean(o[filterdict_inv['r']].magcorr)
    plt.figtext(0.15, 0.8, 'Nobs = %d, Nnights = %d\n Period %.2f hours \n Mean object color %.2f (%.2f)'
                % (len(obj), len(obj.nid.unique()), period * 24., objgr, objgr_2))
    plt.xlabel('Phase', fontsize='x-large')
    plt.ylabel('magcorr', fontsize='x-large')
    plt.title(name)
    plt.grid(True, alpha=0.3)
    return fig
