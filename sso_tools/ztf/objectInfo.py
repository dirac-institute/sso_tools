"""Information about a particular object - orbit, known lightcurve, and predicted magnitudes."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.jplsbdb import SBDB
from astroquery.mpc import MPC
import lsst.sims.movingObjects as mo
from .ztfdb import ztfname_to_designation
from ..catalogs import read_sdss_moc

__all__ = ['queryJPL', 'queryMPC', 'sdss_colors', 'AsteroidObj']

# ZTF filters / integer id's
filterdict = {1: 'g', 2: 'r', 3: 'i'}
filterdict_inv = {'g': 1, 'r': 2, 'i': 3}
filtercolors = {1: 'g', 2: 'r', 3: 'y'}


def queryJPL(designation):
    """Query JPL Horizons for information about object 'designation'.

    Parameters
    ----------
    designation: str
        A name for the object that JPL Horizons will understand.

    Returns
    -------
    pd.Series
        Series containing orbit and select physical information.
    """
    sbdb = SBDB.query(designation, phys=True, full_precision=True)
    if 'orbit' not in sbdb:
        raise ValueError('Problem identifying orbit in returned information: %s', sbdb)
    orbval = sbdb['orbit']['elements']
    phys = sbdb['phys_par']
    if 'diameter' in phys:
        diam = phys['diameter'].to('km')
    else:
        diam = np.NaN
    if 'albedo' in phys:
        albedo = float(phys['albedo'])
    else:
        albedo = np.NaN
    if 'H' in phys:
        H = phys['H']
    else:
        H = 999.
    if 'rot_per' in phys:
        rot = phys['rot_per'].to('hr')
    else:
        rot = np.NaN
    if 'fullname' in sbdb['object']:
        fullname = sbdb['object']['fullname']
    else:
        fullname = sbdb['object']['des']
    orbit = pd.Series(data={'des': sbdb['object']['des'],
                            'fullname': fullname,
                            'FORMAT': 'KEP',
                            'a': orbval['a'].to('AU'),
                            'q': orbval['q'].to('AU'),
                            'e': float(orbval['e']),
                            'inc': orbval['i'].to('deg'),
                            'Omega': orbval['om'].to('deg'),
                            'argPeri': orbval['w'].to('deg'),
                            'tPeri': orbval['tp'].to('d') - 2400000.5,  # to MJD
                            'meanAnomaly': orbval['ma'].to('deg'),
                            'epoch': sbdb['orbit']['epoch'].to('d') - 2400000.5,  # to MJD
                            'H': H,
                            'g': 0.15,
                            'diam': diam,
                            'albedo': albedo,
                            'rot': rot
                            })
    return orbit


def queryMPC(designation):
    """Query JPL Horizons for information about object 'designation'.

    Parameters
    ----------
    designation: str
        A name for the object that the MPC will understand.
        This can be a number, proper name, or the packed designation.

    Returns
    -------
    pd.Series
        Series containing orbit and select physical information.
    """
    try:
        number = int(designation)
        mpc = MPC.query_object('asteroid', number=number)
    except ValueError:
        mpc = MPC.query_object('asteroid', designation=designation)
    mpc = mpc[0]
    orbit = pd.Series(data={'des': designation,
                            'fullname': mpc['name'],
                            'FORMAT': 'KEP',
                            'a': float(mpc['semimajor_axis']),
                            'q': float(mpc['perihelion_distance']),
                            'e': float(mpc['eccentricity']),
                            'inc': float(mpc['inclination']),
                            'Omega': float(mpc['ascending_node']),
                            'argPeri': float(mpc['argument_of_perihelion']),
                            'tPeri': float(mpc['perihelion_date_jd']) - 2400000.5,  # to MJD
                            'meanAnomaly': float(mpc['mean_anomaly']),
                            'epoch': float(mpc['epoch_jd']) - 2400000.5,  # to MJD
                            'H': float(mpc['absolute_magnitude']),
                            'g': float(mpc['phase_slope']),
                            'diam': -999,
                            'albedo': -999,
                            'rot': -999
                            })
    return orbit


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


class AsteroidObj():
    def __init__(self, magcol='magpsf'):
        self.Orb = mo.Orbits()
        self.pyOrb = mo.PyOrbEphemerides()
        self.obs = None
        self.orbit = None
        self.magcol = magcol
        self.ephMags = None

    def __call__(self, all_sso, ztfname, minJD=None, maxJD=None):
        self.getObs(all_sso=all_sso, ztfname=ztfname, minJD=minJD, maxJD=maxJD)
        self.setOrbit()
        self.add_ztf_magcorr()
        self.add_oorb_magcorr()
        return

    def getObs(self, all_sso, ztfname, minJD=None, maxJD=None, dist_cutoff=10):
        """Pull out the observations of a given object, sets self.obs

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
        dist_cutoff: int, opt
            The maximum ssdistnr value in the alert == arcseconds between measured position and expected position.
            Default 10.
        """
        self.name = ztfname
        self.dist_cutoff = dist_cutoff
        obs = all_sso.query('ssnamenr == @ztfname and ssdistnr <= @dist_cutoff')
        if minJD is not None:
            obs = obs.query('jd > @minJD')
        if maxJD is not None:
            obs = obs.query('jd < @maxJD')
        self.obs = obs
        self.filterlist = obs.fid.unique()
        for f in self.filterlist:
            o = self.obs.query('fid == @f')
            print('Filter %s (%s) has %d observations' % (f, filterdict[f], len(o)))
        self.nobs = len(self.obs)
        self.nnights = len(self.obs.nid.unique())
        gobs = self.obs.query('fid == 1')
        if len(gobs) > 0:
            self.med_g = np.median(gobs[self.magcol])
        else:
            self.med_g = -999
        robs = self.obs.query('fid == 2')
        if len(robs) > 0:
            self.med_r = np.median(robs[self.magcol])
        else:
            self.med_r = -999

    def setOrbit(self, source='SBDB'):
        """Query JPL SBDB for orbit and physical properties, sets up internal orbit/ephemeris objects."""
        desig = ztfname_to_designation(self.name)
        if isinstance(source, pd.DataFrame):
            self.orbit = source.query('ztfname == @self.name')
        elif source == 'MPC':
            self.orbit = queryMPC(desig)
        else:
            self.orbit = queryJPL(desig)
        self.Orb.setOrbits(self.orbit)
        self.pyOrb.setOrbits(self.Orb)

    def add_ztf_magcorr(self):
        """Add the corrected (for distance and phase angle) magnitudes using ZTF ssmagnr values.
        """
        # Add the 'corrected' magnitude values using ZTF values.
        magcorr = self.obs[self.magcol].values - self.obs.ssmagnr.values
        self.obs = self.obs.assign(magcorrZTF=magcorr)

    def add_oorb_magcorr(self):
        """Add the corrected (for distance and phase angle) magnitudes using OO predicted values.
        """
        # Add the 'corrected' magnitude values using OOrb predicted values.
        ephs = self.pyOrb.generateEphemerides(self.obs.mjd.values, timeScale='UTC', obscode='I41')
        self.ephs = ephs
        self.obs = self.obs.assign(magOO=ephs[0]['magV'],
                                   phaseangle=ephs[0]['phase'],
                                   heliodist=ephs[0]['helio_dist'],
                                   geodist=ephs[0]['geo_dist'],
                                   velocity=ephs[0]['velocity'])
        # The predicted magnitudes from PyOrb can be way off the reported ZTF mags.
        # Not sure why -- zeropoints? filters? (?) -- but add a correction, per filter.
        self.offsets = {}
        for f in self.filterlist:
            o = self.obs.query('fid == @f')
            self.offsets[f] = np.mean(o[self.magcol] - o.magOO)
        predmag = ephs[0]['magV']
        for f in self.filterlist:
            mask = np.where(self.obs['fid'] == f)
            predmag[mask] += self.offsets[f]
        magcorr = self.obs[self.magcol].values - predmag
        self.obs = self.obs.assign(magOO=predmag, magcorrOO=magcorr)
        # And generate ephemerides with the same offsets for all times.
        self.ephMags = {}
        t = np.arange(self.obs.mjd.min(), self.obs.mjd.max()+0.5, 1.0)
        self.ephMags['t'] = t + 2400000.5
        ephsT = self.pyOrb.generateEphemerides(t, timeScale='UTC', obscode='I41')
        self.H = {}
        for f in self.filterlist:
            self.ephMags[f] = ephsT[0]['magV'] + self.offsets[f]
            self.H['V'] = self.orbit.H
            self.H[f] = self.orbit.H + self.offsets[f]

    def translate_df(self, pred='oo'):
        """Translate object observation DataFrame into the columns/format for lc_utils code.

        Parameters
        ----------
        pred: str, opt
            Which predicted magnitude/corrected magnitude values to use. Default 'oo' ('oo' or 'ztf').

        Returns
        -------
        pd.DataFrame
        """
        pred = pred.lower()
        if pred not in ['oo', 'ztf']:
            raise ValueError('pred %s should be either oo or ztf.' % pred)
        if self.magcol == 'magpsf':
            sigmamag = 'sigmapsf'
        if self.magcol == 'magap':
            sigmamag = 'sigmagap'
        if pred == 'oo':
            predmagcol = 'magOO'
            magcorrcol = 'magcorrOO'
        else:
            predmagcol = 'ssmagnr'
            magcorrcol = 'magcorrZTF'
        newcols = ['objId', 'jd', 'fid', 'mag', 'sigmamag', 'predmag', 'magcorr', 'night', 'phaseangle']
        oldcols = ['ssnamenr', 'jd', 'fid', self.magcol, sigmamag, predmagcol, magcorrcol, 'nid',
                   'phaseangle']
        df = self.obs[oldcols]
        df.columns = newcols
        return df

    def check_astrometry(self):
        """Plot astrometric residuals of the object.

        Returns
        -------
        plt.figure
        """
        # Check astrometry residuals?
        fig = plt.figure(figsize=(8, 8))
        plt.plot(self.obs.jd, self.obs.ssdistnr, 'k.')
        plt.figtext(0.2, 0.8, 'Nobs = %d' % (self.nobs))
        plt.xlabel('JD', fontsize='x-large')
        plt.ylabel('ssdistnr (arcsec)', fontsize='x-large')
        plt.title(self.name)
        return fig

    def vis_psf_ap_photometry(self, fulldates=False):
        """Plot the unphased, uncorrected photometry (PSF and aperture).

        Parameters
        ----------
        fulldates: bool, opt
            Plot the full dates (True) or just the change in dates after the start (False).

        Returns
        -------
        plt.figure
        """
        fig = plt.figure(figsize=(16, 6))
        t0 = self.obs.jd.iloc[0]
        plt.subplot(1, 2, 1)
        for f in self.filterlist:
            o = self.obs.query('fid == @f')
            if fulldates:
                times = o.jd
            else:
                times = o.jd - t0
            plt.errorbar(times, o.magap, yerr=o.sigmagap, color=filtercolors[f],
                         marker='.', linestyle='')
        """
        if fulldates:
            times = self.obs.jd
        else:
            times = self.obs.jd - t0
        plt.plot(times, self.obs.ssmagnr, 'c-')
        """
        if self.ephMags is not None:
            t = self.ephMags['t']
            if not fulldates:
                t = t - t0
            for f in self.filterlist:
                plt.plot(t, self.ephMags[f], color=filtercolors[f], linestyle=':')
        plt.figtext(0.15, 0.8, 'Nobs = %d, Nnights = %d' % (self.nobs, self.nnights))
        plt.xticks(rotation=90)
        if fulldates:
            plt.xlabel('JD', fontsize='x-large')
        else:
            plt.xlabel('Delta JD', fontsize='x-large')
        plt.ylabel('ApMag', fontsize='x-large')
        plt.gca().invert_yaxis()
        plt.title(self.name)
        plt.subplot(1, 2, 2)
        for f in self.filterlist:
            o = self.obs.query('fid == @f')
            if fulldates:
                times = o.jd
            else:
                times = o.jd - t0
            plt.errorbar(times, o.magpsf, yerr=o.sigmapsf, color=filtercolors[f],
                         marker='.', linestyle='')
        """
        if fulldates:
            times = self.obs.jd
        else:
            times = self.obs.jd - t0
        plt.plot(times, self.obs.ssmagnr, 'c-')
        """
        if self.ephMags is not None:
            t = self.ephMags['t']
            if not fulldates:
                t = t - t0
            for f in self.filterlist:
                plt.plot(t, self.ephMags[f], color=filtercolors[f], linestyle=':')
        plt.figtext(0.58, 0.8, 'Nobs = %d, Nnights = %d' % (self.nobs, self.nnights))
        plt.xticks(rotation=90)
        if fulldates:
            plt.xlabel('JD', fontsize='x-large')
        else:
            plt.xlabel('Delta JD', fontsize='x-large')
        plt.ylabel('PSF mag', fontsize='x-large')
        plt.gca().invert_yaxis()
        plt.title(self.name)
        if len(self.filterlist) > 1:
            if 1 in self.filterlist:
                f0 = 1
                if 2 in self.filterlist:
                    f1 = 2
                else:
                    f2 = self.filterlist[1]
            else:
                f0 = self.filterlist[0]
                f1 = self.filterlist[1]
            colorname = '%s-%s' % (filterdict[f0], filterdict[f1])
            color = (self.obs.query('fid == @f0').magpsf.mean()
                     - self.obs.query('fid == @f1').magpsf.mean())
            print('Average color %s = %.2f' % (colorname, color))
        return fig

