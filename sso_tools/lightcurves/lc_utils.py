"""
lc_utils

Minimal DataFrame for lightcurve fitting:
[objId, jd, fid, mag, sigmamag, predmag, magcorr, night]
"""

import numpy as np
import matplotlib.pyplot as plt
from gatspy import periodic


__all__ = ['vis_photometry', 'vis_corr_photometry', 'fit_model', 'make_periodogram',
           '_make_predictions', 'plot_unphased', 'phase_times', 'plot_phased']

# ZTF filters / integer id's
filterdict = {1: 'g', 2: 'r', 3: 'i'}
filterdict_inv = {'g': 1, 'r': 2, 'i': 3}
filtercolors = {1: 'g', 2: 'r', 3: 'y'}


def _make_figurelabel(obj):
    label = 'Nobs = %d, Nnights = %d' % (len(obj), len(obj.night.unique()))
    filterlist = obj.fid.unique()
    if len(filterlist) > 1:
        colorname = '%s-%s' % (filterdict[filterlist[0]], filterdict[filterlist[1]])
        color = (obj.query('fid == @filterlist[0]').magcorr.mean()
                 - obj.query('fid == @filterlist[1]').magcorr.mean())
        label += '\nAverage color %s = %.2f' % (colorname, color)
    return label


def vis_photometry(obj):
    """Plot the unphased, uncorrected photometry.

    Parameters
    ----------
    obj: pd.DataFrame
        The dataframe of all observations of a given object.

    Returns
    -------
    plt.figure
    """
    fig = plt.figure(figsize=(8, 6))
    for f in obj.fid.unique():
        o = obj.query('fid == @f')
        plt.errorbar(o.jd - obj.jd.iloc[0], o.mag, yerr=o.sigmamag, color=filtercolors[f],
                     marker='.', linestyle='')
    plt.xticks(rotation=90)
    plt.xlabel('delta JD', fontsize='x-large')
    plt.ylabel('Mag', fontsize='x-large')
    plt.title(obj.objId.unique()[0])
    plt.grid(True, alpha=0.3)
    label = _make_figurelabel(obj)
    plt.figtext(0.15, 0.8, label)
    return fig


def vis_corr_photometry(obj):
    """Plot the unphased, corrected photometry.

    Parameters
    ----------
    obj: pd.DataFrame
        The dataframe of all observations of a given object.

    Returns
    -------
    plt.figure
    """
    # Look at photometry after subtracting expected magnitude
    # (i.e. subtract an approximate phase curve, but the ZTF predicted values do have errors sometimes)
    fig = plt.figure(figsize=(8, 6))
    for f in obj.fid.unique():
        o = obj.query('fid == @f')
        plt.errorbar(o.jd - obj.jd.iloc[0], o.magcorr, yerr=o.sigmamag, color=filtercolors[f],
                     marker='.', linestyle='')
    plt.xticks(rotation=90)
    plt.xlabel('delta JD', fontsize='x-large')
    plt.ylabel('MagCorr (magpsf - ssmagnr)', fontsize='x-large')
    plt.title(obj.objId.unique()[0])
    plt.grid(True, alpha=0.3)
    label = _make_figurelabel(obj)
    plt.figtext(0.15, 0.8, label)
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
    model.fit(obj.jd, obj.magcorr, obj.sigmamag, obj.fid)
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
    top_periods = model.find_best_periods()
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
    return periods, power


def _make_predictions(tfit, period, model):
    """Generate predicted magnitudes at the times of the observations.

    Parameters
    ----------
    times: np.ndarray
        The times to create predictions for.
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
    filts = np.ones(len(tfit))
    pred = model.predict(tfit, filts=filts, period=period)
    pred2 = model.predict(tfit, filts=filts+1, period=period)
    return pred, pred2


def plot_unphased(obj, period, model):
    """Plot the unphased corrected photometry and model predictions.

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
    plt.figure
    """
    fig = plt.figure(figsize=(10, 8))
    times = np.arange(obj.jd.min(), obj.jd.max() + 0.01, 0.01)
    pred, pred2 = _make_predictions(times, period, model)
    plt.plot(times, pred, color=filtercolors[1], linestyle=':', alpha=0.2)
    plt.plot(times, pred2, color=filtercolors[2], linestyle=':', alpha=0.2)
    for f in obj.fid.unique():
        o = obj.query('fid == @f')
        plt.errorbar(o.jd, o.magcorr, yerr=o.sigmamag, color=filtercolors[f],
                     marker='.', linestyle='')
    label = _make_figurelabel(obj)
    objgr = np.mean(pred2 - pred)
    label += ' (%.2f)' % objgr
    label += '\nPeriod %.2f hours' % (period * 24)
    plt.figtext(0.15, 0.8, label)
    plt.xlabel('JD', fontsize='x-large')
    plt.ylabel('MagCorr', fontsize='x-large')
    plt.title(obj.objId.unique()[0])
    return fig


def phase_times(times, period, offset=0):
    """Phase times by period."""
    phased = (times - offset) % period
    phased = phased / period
    return phased


def plot_phased(obj, period, model):
    # Phased
    """
    tfit = np.arange(0, period, 0.002)
    filts = np.ones(len(tfit))
    pred = model.predict(tfit, filts=filts, period=period)
    pred2 = model.predict(tfit, filts=filts+1, period=period)
    """
    times = np.arange(0, period, 0.0001)
    pred, pred2 = _make_predictions(times, period, model)
    phase_t = phase_times(times, period)
    fig = plt.figure(figsize=(10, 8))
    plt.plot(phase_t, pred, color=filtercolors[1], 
             linestyle=':')
    plt.plot(phase_t, pred2, color=filtercolors[2], 
             linestyle=':')
    for f in obj.fid.unique():
        o = obj.query('fid == @f')
        plt.errorbar(phase_times(o.jd, period), o.magcorr, yerr=o.sigmamag, color=filtercolors[f],
                     marker='.', markersize=10, linestyle='')
    label = _make_figurelabel(obj)
    objgr = np.mean(pred2 - pred)
    label += ' (%.2f)' % objgr
    label += '\nPeriod %.2f hours' % (period * 24)
    plt.figtext(0.15, 0.8, label)
    plt.xlabel('Lightcurve Phase', fontsize='x-large')
    plt.ylabel('MagCorr', fontsize='x-large')
    plt.title(obj.objId.unique()[0])
    plt.grid(True, alpha=0.3)
    return fig
