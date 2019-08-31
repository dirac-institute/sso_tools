"""
lc_utils

Minimal DataFrame for lightcurve fitting:
[objId, jd, fid, mag, sigmamag, predmag, magcorr, night]

Because of how magcorr is generated,
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from gatspy import periodic
from scipy.signal import find_peaks

__all__ = ['phase_times', 'LCObject']

# ZTF filters / integer id's
filterdict = {1: 'g', 2: 'r', 3: 'i'}
filterdict_inv = {'g': 1, 'r': 2, 'i': 3}
filtercolors = {1: 'g', 2: 'r', 3: 'y'}


def phase_times(times, period, offset=0):
    """Phase times by period."""
    phased = (times - offset) % period
    phased = phased / period
    return phased


class LCObject():
    def __init__(self, min_period=1.0/24.0, max_period=60.0/24.0,
                 nsigma=3, Nterms_base=2, Nterms_band=0):
        """
        Parameters
        ----------
        min_period: float, opt
            Minimum period to fit for, in days.
        max_period: float, opt
            Maximum period to fit for, in days.
        nsigma: int, opt
            Number of standard deviations to allow around the mean, before rejecting input data points.
        Nterms_base: int, opt
            Number of terms for the base LS fit. Default 2.
        Nterms_band: int, opt
            Number of terms to allow between bandpasses. Default 0.
        """
        self.min_period = min_period
        self.max_period = max_period
        self.nsigma = nsigma
        self.Nterms_base = Nterms_base
        self.Nterms_band = Nterms_band

    def __call__(self, lcobs, photoffsets=None, outfile=None):
        self.setObs(lcobs)
        self.photoffsets = photoffsets

        figs = {}
        figs['corrphot'] = self.vis_corr_photometry()
        self.fit_model()
        self.print_top_periods(outfile=outfile)
        if outfile is None:
            print('chi2DOF', self.chis2dof)
        else:
            print(self.chis2dof, file=outfile)
        figs['periodogram'] = self.make_linear_periodogram()
        figs['phased'] = self.plot_phased()
        return figs

    def setObs(self, lcobs):
        self.lcobs = lcobs
        self.name = self.lcobs.objId.unique()[0]
        self.outlier_rejection()
        self.filterlist = self.lcobs.fid.unique()
        self.nobs = len(self.lcobs)
        self.nnights = len(self.lcobs.night.unique())

    def outlier_rejection(self):
        # Calculate RMS.
        # Just calculate RMS across both bands, because most of color-variation
        # already lost in magcorr correction.
        rms = np.std(self.lcobs.magcorr)
        nsigma = self.nsigma
        self.lcobs_reject = self.lcobs.query('abs(magcorr) >= @nsigma*@rms')
        self.lcobs = self.lcobs.query('abs(magcorr) < @nsigma*@rms')
        print('Rejected %d observations from %s' % (len(self.lcobs_reject), self.name))

    def _make_figurelabel(self):
        label = 'Nobs = %d, Nnights = %d' % (self.nobs, self.nnights)
        return label

    def vis_photometry(self):
        """Plot the unphased, uncorrected photometry.

        Returns
        -------
        plt.figure
        """
        fig = plt.figure(figsize=(8, 6))
        for f in self.filterlist:
            o = self.lcobs.query('fid == @f')
            plt.errorbar(o.jd - self.lcobs.jd.iloc[0], o.mag, yerr=o.sigmamag, color=filtercolors[f],
                         marker='.', linestyle='')
        plt.xticks(rotation=90)
        plt.xlabel('delta JD', fontsize='x-large')
        plt.ylabel('Mag', fontsize='x-large')
        plt.gca().invert_yaxis()
        plt.title(self.name)
        plt.grid(True, alpha=0.3)
        label = self._make_figurelabel()
        plt.figtext(0.15, 0.8, label)
        return fig

    def vis_corr_photometry(self):
        """Plot the unphased, corrected photometry.

        Returns
        -------
        plt.figure
        """
        # Look at photometry after subtracting expected magnitude
        # (i.e. subtract an approximate phase curve, but the ZTF predicted values do have errors sometimes)
        fig = plt.figure(figsize=(8, 6))
        for f in self.filterlist:
            o = self.lcobs.query('fid == @f')
            plt.errorbar(o.jd - self.lcobs.jd.iloc[0], o.magcorr, yerr=o.sigmamag, color=filtercolors[f],
                         marker='.', linestyle='')
        plt.xticks(rotation=90)
        plt.xlabel('delta JD', fontsize='x-large')
        plt.ylabel('MagCorr (mag_obs - pred)', fontsize='x-large')
        plt.gca().invert_yaxis()
        plt.title(self.name)
        plt.grid(True, alpha=0.3)
        label = self._make_figurelabel()
        plt.figtext(0.15, 0.8, label)
        return fig

    def fit_model(self):
        """Fit the lightcurve data using periodic.LombScargleMultiband.
        """
        # Let's try to fit these values with the gatspy multiband fitter
        self.model = periodic.LombScargleMultiband(fit_period=True,
                                                   Nterms_base=self.Nterms_base,
                                                   Nterms_band=self.Nterms_band)
        big_period = np.min([self.max_period, (self.lcobs.jd.max() - self.lcobs.jd.min())])
        self.model.optimizer.period_range = (self.min_period, big_period)
        self.model.optimizer.first_pass_coverage = 200
        self.model.fit(self.lcobs.jd, self.lcobs.magcorr, self.lcobs.sigmamag, self.lcobs.fid)
        self.top_periods = self.model.find_best_periods()
        self.best_period = self.top_periods[0]
        times = np.arange(0, self.best_period, self.best_period/100)
        predictions = self.make_predictions(times, self.best_period)
        # Should we probably double this peak?
        idx = find_peaks(predictions[1])[0]
        self.npeaks = len(idx)
        if len(idx) == 1:
            self.best_period *= 2
        # Calculate amplitude
        self.amp = 0
        for k in predictions:
            amp = predictions[k].max() - predictions[k].min()
            self.amp = max(amp, self.amp)
        # Calculate chisq of fit
        self.calc_chisq()
        # Calculate updated colors
        if self.photoffsets is not None:
            if 1 in self.photoffsets and 2 in self.photoffsets:
                self.gr = self.photoffsets[1] - self.photoffsets[2]
                # And check if we can refine this from the lc fit
                self.gr = self.gr + (predictions[1] - predictions[2])[0]
            else:
                self.gr = -999
            if 3 not in lc.photoffsets:
                self.ri = -999
            else:
                self.ri = self.photoffsets[2] - self.photoffsets[3]
                self.ri = self.ri + (predictions[2] - predictions[3])[0]

    def print_top_periods(self, outfile=None):
        if outfile is None:
            print('Top Periods (and doubles):')
            print(' '.join(['%.3f (%.3f) hours\n' % (p*24.0, p*24.0*2) for p in self.top_periods if p < 1]),
                  ' '.join(['%.3f (%.3f) days\n' % (p, p*2) for p in self.top_periods if p > 1]))
            print('Best fit period: %.3f hours' % (self.best_period * 24))
            print()
        else:
            print('Top Periods (and doubles):', file=outfile)
            print(' '.join(['%.3f (%.3f) hours\n' % (p*24.0, p*24.0*2) for p in self.top_periods if p < 1]),
                  ' '.join(['%.3f (%.3f) days\n' % (p, p*2) for p in self.top_periods if p > 1]),
                  file=outfile)
            print('Best fit period: %.3f hours' % (self.best_period * 24), file=outfile)
            print(file=outfile)

    def calc_chisq(self, period=None):
        if period is None:
            period = self.best_period
        z = None
        for f in self.filterlist:
            o = self.lcobs.query('fid == @f')
            predictions = self.make_predictions(o.jd.values, period)
            if z is None:
                z = (o.magcorr - predictions[f]) / o.sigmamag
            else:
                z = z.append((o.magcorr - predictions[f]) / o.sigmamag)
        self.chis2 = np.sum(z ** 2)
        self.chis2dof = self.chis2 / (self.nobs - 1)

    def make_auto_periodogram(self):
        """Plot the periodogram for the model (after fit).

        Returns
        -------
        periods, power, plt.figure
            Periods and power from periodogram calculation, and figure.
        """
        # Look at the periodogram
        periods, power = self.model.periodogram_auto(oversampling=100, nyquist_factor=5)
        fig = plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.plot(periods * 24, power)
        for p in self.top_periods:
            plt.axvline(p * 24, color='k', linestyle=':')
        plt.xlabel('Period (hours)')
        plt.ylabel('Power')
        plt.subplot(1, 2, 2)
        plt.plot(periods * 24, power)
        for p in self.top_periods:
            plt.axvline(p*24, color='k', linestyle=':')
        plt.xlabel('Period (hours)')
        plt.ylabel('Power')
        plt.xlim(0, 10)
        self.auto_periods = periods
        self.auto_power = power
        return fig

    def make_linear_periodogram(self):
        """Plot the periodogram for the model (after fit).

        Returns
        -------
        periods, scores, plt.figure
            Periods and scores from periodogram calculation, and figure.
        """
        # Look at the periodogram
        big_period = np.min([self.max_period, (self.lcobs.jd.max() - self.lcobs.jd.min())])
        periods = np.linspace(self.min_period, big_period, 100000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = self.model.score(periods)
        fig = plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.plot(periods * 24, scores)
        for p in self.top_periods:
            plt.axvline(p * 24, color='k', linestyle=':')
        plt.xlabel('Period (hours)')
        plt.ylabel('Power')
        plt.subplot(1, 2, 2)
        plt.plot(periods * 24, scores)
        for p in self.top_periods:
            plt.axvline(p*24, color='k', linestyle=':')
        plt.xlabel('Period (hours)')
        plt.ylabel('Power')
        plt.xlim(0, 10)
        self.lin_periods = periods
        self.lin_scores = scores
        return fig

    def make_predictions(self, tfit, period):
        """Generate predicted magnitudes at the times of the observations.

        Parameters
        ----------
        times: np.ndarray
            The times to create predictions for.
        period: float
            Period to use for the predictions.

        Returns
        -------
        np.array, np.array, np.array
            Time of the observations, Magnitude of the predicted observations in the first filter (g),
            and Magnitude of the predicted observations in the second filter (r)."""
        # Generate predicted data, using one of the periods.
        predictions = {}
        for f in self.filterlist:
            filts = np.zeros(len(tfit)) + f
            predictions[f] = self.model.predict(tfit, filts=filts, period=period)
        return predictions

    def plot_unphased(self, period=None):
        """Plot the unphased corrected photometry and model predictions.

        Parameters
        ----------
        period: float
            Period to use for the predictions.

        Returns
        -------
        plt.figure
        """
        if period is None:
            period = self.best_period
        fig = plt.figure(figsize=(10, 8))
        times = np.arange(self.lcobs.jd.min(), self.lcobs.jd.max() + 0.01, 0.01)
        predictions = self.make_predictions(times, period)
        for f in self.filterlist:
            plt.plot(times, predictions[f], color=filtercolors[f], linestyle=':', alpha=0.2)
            o = self.lcobs.query('fid == @f')
            plt.errorbar(o.jd, o.magcorr, yerr=o.sigmamag, color=filtercolors[f],
                         marker='.', linestyle='')
        label = self._make_figurelabel()
        label += '\nPeriod %.2f hours' % (period * 24)
        plt.figtext(0.15, 0.8, label)
        plt.xlabel('JD', fontsize='x-large')
        plt.ylabel('MagCorr', fontsize='x-large')
        plt.gca().invert_yaxis()
        plt.title(self.name)
        return fig

    def plot_phased(self, period=None):
        # Phased
        if period is None:
            period = self.best_period
        times = np.arange(0, period, period/100)
        predictions = self.make_predictions(times, period)
        phase_t = phase_times(times, period)
        fig = plt.figure(figsize=(10, 8))
        for f in self.filterlist:
            plt.plot(phase_t, predictions[f], color=filtercolors[f],
                     linestyle=':')
            o = self.lcobs.query('fid == @f')
            plt.errorbar(phase_times(o.jd, period), o.magcorr, yerr=o.sigmamag, color=filtercolors[f],
                         marker='.', markersize=10, linestyle='')
        label = self._make_figurelabel()
        label += '\nPeriod %.2f hours' % (period * 24)
        plt.figtext(0.15, 0.8, label)
        plt.xlabel('Lightcurve Phase', fontsize='x-large')
        plt.ylabel('MagCorr', fontsize='x-large')
        plt.gca().invert_yaxis()
        plt.title(self.name)
        plt.grid(True, alpha=0.3)
        return fig
