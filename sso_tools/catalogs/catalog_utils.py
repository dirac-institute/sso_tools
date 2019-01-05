import numpy as np
import pandas as pd
from sbpy.data import Names
from sbpy.data.names import TargetNameParseError
from astropy.time import Time

__all__ = ['read_mpcorb', 'read_sdss_moc', 'read_astorb', 'read_lcdb']


def _parse_name(s):
    """Use sbpy Names to parse an asteroid name string into number, name & designation.
    Name = name or designation."""
    try:
        namedict = Names.parse_asteroid(s)
        namedict.setdefault('number', -999)
        namedict.setdefault('desig', 'NULL')
        namedict.setdefault('name', namedict['desig'])
    except (TargetNameParseError, AttributeError) as e:
        namedict = None
    return namedict


def _parse_mpc_names_and_epochs(x):
    """Turn the MPC name into separate number/name/designation, and parse MPC epochs into MJD.
    Sets objId to be either the number or designation. """
    namedict = _parse_name(x.readableName)
    if namedict is None:
        try:
            x.numberId = int(x.mpcId)
        except ValueError:
            x.numberId = -999
        x.Name = x.readableName
        x.Desig = x.readableName
    else:
        x.numberId = int(namedict['number'])
        x.Name = namedict['name']
        x.Desig = namedict['desig']
    if x.numberId >= 0:
        x.objId = x.numberId
    else:
        x.objId = x.Desig
    try:
        x.epoch = _unpack_mpc_date(x.epoch)
    except:
        print(x.epoch)
    return x


def _unpack_mpc_date(packed_date):
    # See https://minorplanetcenter.net/iau/info/PackedDates.html
    # for MPC documentation on packed dates.
    # Examples:
    #    1998 Jan. 18.73     = J981I73
    #    2001 Oct. 22.138303 = K01AM138303
    packed_date = str(packed_date)
    year = _mpc_lookup(packed_date[0]) * 100 + int(packed_date[1:2])
    # Month is encoded in third column.
    month = _mpc_lookup(packed_date[3])
    day = _mpc_lookup(packed_date[4])
    isot_string = '%d-%02d-%02d' % (year, month, day)
    if len(packed_date) > 5:
        fractional_day = float(packed_date[5:])
        isot_string += '.%f' % fractional_day
    t = Time(isot_string, format='isot', scale='tt')
    return t.mjd


def _mpc_lookup(x):
    # Convert the single character dates into integers.
    try:
        x = int(x)
    except ValueError:
        x = ord(x) - 55
    if x < 0 or x > 31:
        raise ValueError
    return x


def _parse_astorb_names_and_epochs(x):
    """Turn the astorb name into separate number/name/designation, and parse astorb epochs into MJD."""
    if x.numberId >= 0:
        x.objId = x.numberId
    else:
        x.objId = x.Name
    namedict = _parse_name(x.Name)
    if namedict is None:
        x.Desig = x.Name
    else:
        x.Desig = namedict['desig']
    try:
        x.epoch = _convert_astorb_date(x.epoch)
    except:
        print(x.epoch)
    return x


def _convert_astorb_date(date):
    date = str(date)
    year = int(date[0:4])
    month = int(date[5:6])
    day = int(date[6:8])
    isot_string = '%d-%02d-%02d' % (year, month, day)
    t = Time(isot_string, format='isot', scale='tt')
    return t.mjd


def _parse_sdss_names(x):
    """Convert the SDSS names into number/name/designation.
    Sets objId to be the number or designation."""
    x.Name = x.Name.replace('_', ' ')
    """
    namedict = _parse_name(x.Name)
    if namedict is None:
        x.Desig = x.Name
    else:
        x.Desig = namedict['desig']
    if x.numberId >= 0:
        x.objId = x.numberId
    else:
        x.objId = x.Name
    """
    return x


def _parse_lcdb_names(x):
    "Parse the LCDB names."
    namedict = _parse_name(x.Desig)
    if namedict is not None:
        x.Desig = namedict['desig']
    if x.numberId >= 0:
        x.objId = x.numberId
    else:
        x.objId = x.Name
    return x


def read_mpcorb(filename='MPCORB.DAT', header=True, parse_names=True):
    """Read Minor Planet Center Orbit Database
    File contains published orbital elements for all numbered and unnumbered
    multi-opposition minor planets.
    Get the file at:
    https://www.minorplanetcenter.net/iau/MPCORB.html
    See documentation at:
    https://minorplanetcenter.net/iau/info/MPOrbitFormat.html

    Parameters
    ----------
    filename: str, opt
        The full path to the MPCORB catalog file.
    header: bool, opt
        Flag indicating whether this file contains the typical MPCORB header or not.
        If True, then will skip the first 43 lines of the input file.

    Returns
    -------
    pandas.DataFrame
    """
    if header:
        skiprows = 43
    else:
        skiprows = 0
    names = ['mpcId', 'H', 'G', 'epoch', 'meanAnomaly', 'argPeri',
             'Omega', 'inc', 'e', 'meanDailyMotion', 'a', 'reference',
             'N_Obs', 'N_Opp', 'yr_1st&last_Obs', 'r.m.s',
             'coarsePerts', 'precisePerts', 'computer',
             'readableName', 'lastObs']
    colspecs = [(0,7),(8,13),(14,19),(20,25),(26,35),(37,46),
                (48,57),(59,68),(70,79),(80,91),(92,103),(108,116),
                (117,122),(123,126),(127,136),(137,141),
                (142, 145),(146,149),(150,160),(166,194),(194,202)]
    mpc = pd.read_fwf(filename, names=names, skiprows=skiprows, colspecs=colspecs, index=False)
    mpc['numberId'] = np.zeros(len(mpc), int) - 999
    mpc['Name'] = np.empty(len(mpc), str)
    mpc['Desig'] = np.empty(len(mpc), str)
    mpc['objId'] = np.empty(len(mpc), str)
    if parse_names:
        mpc = mpc.apply(_parse_mpc_names_and_epochs, axis=1)
    return mpc


def read_astorb(filename='astorb.dat'):
    """Read ASTORB data file.
    ASTORB available from:
    ftp://ftp.lowell.edu/pub/elgb/astorb.html
    ASTORB is made available by Dr. Edward Bowell.
    ASTORB is made possible by funding from NASA grant NAG5-4741 and the Lowell Observatory Endowment.
    This provides updated osculating element orbits, and an estimate of their ephemeris uncertainty.
    To get the updated ephemeris uncertainty, it makes sense to update this input data frequently.
    Parameters
    ----------
    filename : str, opt
        The full path to the MOC catalog file.
    Returns
    -------
    pandas.DataFrame
    """
    names=['numberId', 'Name', 'computer', 'H', 'G1', 'B-V',
           'diameter_IRAS_km', 'classification_IRAS',
           'planet_crossing', 'orbit_comp', 'survey', 'mpc_crit', 'lowell_obs', 'rank',
           'obsArc', '#Obs', 'epoch', 'meanAnomaly', 'argPeri', 'Omega', 'inc', 'e', 'a',
           'date_comp', 'ceu', 'deltaceu', 'ceudate',
           'peumax', 'peumaxdate', 'peumax10', 'peumax10date', 'peumax2', 'peumax2date']
    widths = [7, 19, 16, 6, 6, 5, 6, 5, 4, 4, 4, 4, 4, 4, 6, 6, 9, 11, 11, 11, 10, 11, 13,
              9, 8, 8, 9, 8, 9, 8, 9, 8, 9]
    astorb = pd.read_fwf(filename, index_col=False, names=names, widths=widths)
    astorb['numberId'] = astorb['numberId'].fillna(-999).astype(int)
    astorb['Desig'] = np.empty(len(astorb), str)
    astorb['objId'] = np.empty(len(astorb), str)
    #astorb = astorb.apply(_parse_astorb_names_and_epochs, axis=1)
    return astorb


def read_sdss_moc(filename='ADR4.dat'):
    """Read SDSS Moving Object Catalog.
    SDSS MOC available from:
    http://faculty.washington.edu/ivezic/sdssmoc/sdssmoc.html
    Please see information on that page for citation information.
    Ivezic, Z., Juric, M., Lupton, R.H., Tabachnik, S. & Quinn, T. (the SDSS Collaboration) 2002,
    Survey and Other Telescope Technologies and Discoveries, J.A. Tyson, S. Wolff, Editors,
    Proceedings of SPIE Vol. 4836 (2002)
    The SDSS MOC provides visible colors for moving objects, with varying numbers of
    observations per object. Estimates of the uncertainties are provided.
    SDSS observations provide simultaneous ugrizy measurements.
    Parameters
    ----------
    filename : str, opt
        The full path to the MOC catalog file.
    Returns
    -------
    pandas.DataFrame
    """
    names=['sdssID', 'run', 'col', 'field', 'obj', 'pixrow', 'pixcol',
           'time', 'ra', 'dec', 'l', 'b', 'phi',
           'vmu', 'vmuerr', 'vnu', 'vnuerr', 'vl', 'vb',
           'u', 'uerr', 'g', 'gerr', 'r', 'rerr', 'i', 'ierr', 'z', 'zerr',
           'sdssa', 'aerr', 'V', 'B', 'idFlag', 'numberId', 'Name',
           'detectionCount', 'totalCount', 'sdss_flags',
           'calcRA', 'calcDec', 'calcAppMag', 'helioDistance', 'geoDistance', 'phase',
           'catalogId', 'H', 'G1', 'obsArc', 'epoch', 'a', 'e', 'inc', 'Omega', 'argPeri',
           'meanAnomaly','elemCatalogId', 'aProper', 'eProper', 'siniProper' ]
    # a* color = 0.89 (g - r) + 0.45 (r - i) - 0.57
    moc = pd.read_table(filename, delim_whitespace=True, names=names, usecols=names)
    #moc['Desig'] = np.empty(len(moc), str)
    #moc['objId'] = np.empty(len(moc), str)
    #moc = moc.apply(_parse_sdss_names, axis=1)
    return moc


# ASTDYS proper elements
# family classifications
# http://hamilton.dm.unipi.it/astdys/index.php?pc=5

def read_lcdb(filename='LC_SUM_PUB.TXT'):
    """Read LCDB file tabulating periods and associated information.
    LCDB available from:
    http://www.minorplanet.info/lightcurvedatabase.html
    Please see that page for citation information.
    Warner, B.D., Harris, A.W., Pravec, P. (2009). Icarus 202, 134-146.
    Updated <Date of last update>. http://www.MinorPlanet.info/lightcurvedatabase.html
    The Asteroid Lightcurve Database is a listing of asteroid lightcurve parameters and other information,
    e.g., estimated/measured diameters, absolute magnitudes (H), phase slope parameters (G), albedos, + more.
    Parameters
    ----------
    filename : str, opt
        The full path to the LCDB file (LC_SUM_PUB.TXT or similarly formatted files)
    Returns
    -------
    pandas.DataFrame
    """
    names = ['numberId', 'new', 'Name', 'Desig', 'family', 'Csource', 'class', 'Dsource',
            'Dflag', 'diameter','Hsource', 'H', 'Hband', 'Asource', 'Albedoflag',
            'albedo', 'Pflag', 'period', 'Pdescr','Ampflag', 'AmpMin', 'AmpMax',
            'U', 'notes', 'binary', 'priv' , 'pole', 'sparse', 'widefield']
    colspecs = [(0, 7), (8, 9), (10, 40), (41, 61), (62, 70), (71, 72), (73, 78),
                (79, 80), (81, 82), (83, 92), (93, 94), (95, 100), (101, 102),
                (104, 105), (106, 107), (108, 114), (115, 116), (117, 130), (131, 146),
                (147, 148), (149, 153), (154, 158), (159, 161),(162, 167), (168, 171),
                (172, 175), (176, 179), (180, 182), (183, 184)]
    lcdata = pd.read_fwf(filename, index_col=False, skiprows=5, names=names, colspecs=colspecs)
    lcdata['Frequency'] = 1.0/lcdata.period
    # Map flags to ints.
    tmp = lcdata.new.values
    tmp = np.where(tmp == '*', 1, 0)
    lcdata.new = tmp
    tmp = lcdata.sparse.values
    tmp = np.where(tmp == 'Y', 1, 0)
    lcdata.sparse = tmp
    tmp = lcdata.widefield.values
    tmp = np.where(tmp == 'Y', 1, 0)
    lcdata.widefield = tmp
    #lcdata['objId'] = np.empty(len(lcdata), str)
    #lcdata = lcdata.apply(_parse_lcdb_names, axis=1)
    return lcdata
