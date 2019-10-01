import pandas as pd
import mysql.connector as mariadb
from astropy.time import Time

__all__ = ['fetch_alert_data', 'fetch_alert_data_single', 'read_alert_datafile', 'simple_summary_info',
           'identify_candidates', 'ztfname_to_designation']


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
    all_sso = all_sso.assign(obsdate = obsdate.isot, mjd=obsdate.utc.mjd)
    return all_sso


def fetch_alert_data_single(ztfname, jd_start=None):
    """Query alert database for a single known SSO -- this is super slow, so only for curious one-offs.

    Parameters
    ----------
    ztfname: str
        The ZTF-style "ssnamenr" designation.
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

    query = 'select * from alerts where ssdistnr >=0 and jd > %f and ssnamenr=="%s"' % (jd_start, ztfname)
    print('Querying from ZTF alert database: %s' % query)
    all_sso = pd.read_sql_query(query, con)
    # Add more easily readable date of observation
    obsdate = Time(all_sso['jd'], format='jd', scale='tai')
    all_sso = all_sso.assign(obsdate = obsdate.isot, mjd=obsdate.utc.mjd)
    return all_sso


def read_alert_datafile(filename):
    all_sso = pd.read_csv(filename, low_memory=False)
    if 'mjd' not in all_sso.columns:
        obsdate = Time(all_sso['jd'], format='jd', scale='tai')
        all_sso = all_sso.assign(mjd=obsdate.utc.mjd)
    if 'obsdate' not in all_sso.columns:
        obsdate = Time(all_sso['jd'], format='jd', scale='tai')
        all_sso = all_sso.assign(obsdate=obsdate.isot)
    return all_sso


def simple_summary_info(all_sso):
    print('Alerts ranging from %s to %s' % (all_sso.obsdate.min(), all_sso.obsdate.max()))
    print("Total SSO alerts %d" % len(all_sso))
    print("Number of different objects %d" % len(all_sso.groupby('ssnamenr')))
    print("Number of nights included %d" % len(all_sso.groupby('nid')))


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
    print(f'# Found {len(objnames)} objects with more than {min_obs} observations')
    return objnames


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
        try:
            val = int(ztfname[i:])
            if val > 0:
                desig += str(int(ztfname[i:]))
        except ValueError:  # comets might be 246P, for example
            desig += ztfname[i:]
    return desig
