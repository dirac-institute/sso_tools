import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sso_tools import ztf
from sso_tools import lightcurves as lcu

filterdict = {1: 'g', 2: 'r', 3: 'i'}
filterdict_inv = {'g': 1, 'r': 2, 'i': 3}

def identify_candidates():

    output_file = 'ztf_lc_log.txt'
    log = open(output_file, 'w')

    all_sso = ztf.fetch_alert_data()
    all_sso.to_csv('ztf_alert_sso.csv', index=False)

    print("Total SSO alerts %d" % len(all_sso), file=log)
    print("Number of different objects %d" % len(all_sso.groupby('ssnamenr')), file=log)
    print("Number of nights included %d" % len(all_sso.groupby('nid')), file=log)

    objnames = ztf.identify_candidates(all_sso, min_obs=40, dist_cutoff=10)
    print("Found %d objects with more than 40 observations" % len(objnames), file=log)

    # Save objectnames to a file.
    with (open('objnames.dat', 'w')) as namefile:
        for name in objnames:
            namefile.write('%s\n' % name)

    log.close()


def fit_all_lightcurves(logfile, datafile, objfile):

    log = open(logfile, 'w')
    data = open(datafile, 'w')

    with open(objfile, 'r') as o:
        objnames = []
        for line in o:
            objnames.append(line.rstrip('\n'))

    all_sso = ztf.read_alert_datafile('ztf_alerts_sso.csv')
    print("Total SSO alerts %d" % len(all_sso), file=log)
    print("Number of different objects %d" % len(all_sso.groupby('ssnamenr')), file=log)
    print("Number of nights included %d" % len(all_sso.groupby('nid')), file=log)
    print("Looking for %d objects in this run." % (len(objnames)), file=log)
    ast = ztf.AsteroidObj()
    lc = lcu.LCObject(Nterms_base=2, Nterms_band=0, nsigma=3)

    log.flush()

    #  Pull out observations of each object, returning :
    #  the subset of observations of that object in ZTF alert format (obj)
    #  and a minimal set of these columns, renamed, for the lc_utils code (lc_df)

    header = 'Name,Nobs,Nobs_g,Nobs_r,Nobs_i,Nights,g-r,Period,chis2dof'
    data.write('%s\n' % header)

    for name in objnames:
        # Make a version of the name for output files. (no slashes).
        nameroot = name.replace('/', '-')
        print('Working on %s' % name)
        ast(all_sso, name)

        nobs = ast.nobs
        nobs_g = len(ast.obs.query('fid == 1'))
        nobs_r = len(ast.obs.query('fid == 2'))
        nobs_i = len(ast.obs.query('fid == 3'))
        nnights = ast.nnights
        print('Working on %s, %d observations over %d nights\n' % (name, nobs, nnights), file=log)

        fig = ast.vis_psf_ap_photometry(fulldates=True)
        fig.savefig('%s_rawphot.png' % nameroot, format='png')

        df = ast.translate_df()
        # Fit lightcurve and make figures.
        try:
            figs = lc(df, ast.offsets, outfile=log)

            fig = figs['corrphot']
            fig.savefig('%s_corrphot.png' % nameroot, format='png')
            import pickle
            with open('%s_pickle' % nameroot, 'wb') as m:
                pickle.dump(lc, m)
            fig = figs['periodogram']
            fig.savefig('%s_linearperiodogram.png' % nameroot, format='png')
            fig = figs['phased']
            fig.savefig('%s_phased.png' % nameroot, format='png')
            plt.close('all')

            if filterdict_inv['g'] in lc.photoffsets and filterdict_inv['r'] in lc.photoffsets:
                grcolor = lc.photoffsets[filterdict_inv['g']] - lc.photoffsets[filterdict_inv['r']]
            else:
                grcolor = -999
            datstr = '%s,%d,%d,%d,%d,%d,%f,%f,%f' % (name, nobs, nobs_g, nobs_r, nobs_i, nnights, 
                                                     grcolor,
                                                     lc.best_period*24.0, lc.chis2dof)
            data.write("%s\n" % datstr)
            data.flush()
        except:
            print('Failed to fit model for %s' % name, file=log)
        log.flush()

    data.close()
    log.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pull SSO alerts from the ZTF database OR fit lightcurves")
    parser.add_argument('--pull', dest='pull', action='store_true', default=False,
                        help="Query alerts from database, save to disk and identify objects for lightcurve fitting?")
    parser.add_argument('--objfile', type=str, default='objnames.dat',
                        help="File containing list of object names to which to fit lightcurves.")
    args = parser.parse_args()

    if args.pull:
        print('Identifying candidates')
        identify_candidates()

    else:
        print('Fitting lightcurves')
        logfile = '%s_log.txt' % (args.objfile)
        datafile = '%s_data.txt' % (args.objfile)
        fit_all_lightcurves(logfile, datafile, args.objfile)
