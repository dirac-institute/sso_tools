import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sso_tools import ztf
from sso_tools import lightcurves as lcu


filterdict = {1: 'g', 2: 'r', 3: 'i'}
filterdict_inv = {'g': 1, 'r': 2, 'i': 3}
filtercolors = {1: 'g', 2: 'r', 3: 'y'}


def fit_lightcurves(alertfile, objfile, orbitfile, outputfile):

    with open(objfile, 'r') as o:
        objnames = []
        for line in o:
            objnames.append(line.rstrip('\n'))

    all_sso = ztf.read_alert_datafile('ztf_alerts_sso.csv')
    print("# Total SSO alerts %d" % len(all_sso), file=log)
    print("# Number of different objects %d" % len(all_sso.groupby('ssnamenr')), file=log)
    print("# Number of nights included %d" % len(all_sso.groupby('nid')), file=log)
    print("# Looking for %d objects in this run." % (len(objnames)), file=log)

    if orbitfile is not None:
        orbits = pd.read_csv(orbitfile)
    else:
        orbits = 'JPL'

    output = open(outputfile, 'w')
    header = 'Name,Nobs,N_rejected,Nobs_g,Nobs_r,Nobs_i,Nights,ObsLen,g-r,r-i,H,med_g,med_r,mag_range,Period,ModelP,MedSigMag,Amp,chis2dof'
    output.write('%s\n' % header)
    output.flush()

    for name in objfile:
        nameroot = name.replace('/', '-')
        ast = ztf.AsteroidObj()
        ast.getObs(all_sso=all_sso, ztfname=name)
        ast.setOrbit(orbits)
        ast.add_oorb_magcorr()
        lcobs = ast.translate_df()
        lcobj = lcc.LCObject(Nterms_base=2, Nterms_band=0, nsigma=3)
        figs = lcobj(lcobs, ast.offsets)

        fig = figs['corrphot']
        fig.savefig('%s_corrphot.png' % nameroot, format='png')
        import pickle
        with open('%s_pickle' % nameroot, 'wb') as m:
            pickle.dump(lcobj, m)
        fig = figs['phased']
        fig.savefig('%s_phased.png' % nameroot, format='png')
        plt.close('all')

        nobs = len(lcobj.lcobs)
        n_reject = len(lcobj.lcobs_reject)
        nobs_g = len(lcobj.lcobs.query('fid == 1'))
        nobs_r = len(lcobj.lcobs.query('fid == 2'))
        nobs_i = len(lcobj.lcobs.query('fid == 3'))
        nnights = len(lcobj.lcobs.night.unique())
        obs_len = lcobj.obs.jd.max() - lcobj.lcobs.jd.min()
        period = lcobj.best_period * 24
        datstr = f'{name},{nobs},{n_reject},{nobs_g},{nobs_r},{nobs_i},{nnights},{obs_len}'
        datstr += f'{lcobj.gr},{lcobj.ri},{ast.orbit.H},{lcobj.med_g},{lcobj.med_r},{lcobj.mag_range},'
        datstr += f'{period},{lcobj.model.best_period * 24},{lcobj.med_sigma},{lcobj.amp},{lcobj.chis2dof}'
        output.write("%s\n" % datstr)
        output.flush()
    print('# done all')
    output.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Fit lightcurves")
    parser.add_argument('--alertfile', type=str, default='ztf_alert_sso.csv',
                        help="File containing the alert information.")
    parser.add_argument('--objfile', type=str, default='objnames.dat',
                        help="File containing list of object names to which to fit lightcurves.")
    parser.add_argument('--orbitfile', type=str, default=None,
                        help="File containing orbit information (if previously generated).")
    parser.add_argument('--outputfile', type=str, default=None,
                        help="File into which to write output information about fit.")
    args = parser.parse_args()

    if args.outputfile is None:
        args.outputfile = args.objfile.replace('.dat', '') + "_data.csv"

    fit_all_lightcurves(args.alertfile, args.objfile, args.orbitfile, args.outputfile)
