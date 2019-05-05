import numpy as np
import xarray as xr
import pf_dynamic_sph as pfs
from timeit import default_timer as timer
import time
import os
# import matplotlib
# import matplotlib.pyplot as plt


if __name__ == "__main__":

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (60, 60, 60)
    (dx, dy, dz) = (0.25, 0.25, 0.25)
    higherCutoff = False; cutoffRat = 1.0
    betterResolution = True; resRat = 0.5

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.375, 0.375, 0.375)
    # higherCutoff = False; cutoffRat = 1.0
    # betterResolution = False; resRat = 1.0

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'real', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon'}

    # ---- SET OUTPUT DATA FOLDER ----

    mRat = 1

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/scratchlfs/demler_lab/kis/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    if higherCutoff is True:
        datapath = datapath + '_cutoffRat_{:.2f}'.format(cutoffRat)
    if betterResolution is True:
        datapath = datapath + '_resRat_{:.2f}'.format(resRat)
    datapath = datapath + '/massRatio={:.1f}'.format(mRat)
    if toggleDict['Dynamics'] == 'real':
        innerdatapath = datapath + '/redyn'
    elif toggleDict['Dynamics'] == 'imaginary':
        innerdatapath = datapath + '/imdyn'
    if toggleDict['Grid'] == 'cartesian':
        innerdatapath = innerdatapath + '_cart'
    elif toggleDict['Grid'] == 'spherical':
        innerdatapath = innerdatapath + '_spherical'
    if toggleDict['Coupling'] == 'frohlich':
        innerdatapath = innerdatapath + '_froh'
    elif toggleDict['Coupling'] == 'twophonon':
        innerdatapath = innerdatapath

    interpdatapath = innerdatapath + '/interp'

    if os.path.isdir(interpdatapath) is False:
        os.mkdir(interpdatapath)

    # # Analysis of Total Dataset

    aIBi = -5
    Pnorm_des = 0.12
    qds = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    n0 = qds.attrs['n0']; gBB = qds.attrs['gBB']; mI = qds.attrs['mI']; mB = qds.attrs['mB']
    nu = np.sqrt(n0 * gBB / mB)
    mc = mI * nu

    PVals = qds['P'].values
    Pnorm = PVals / mc
    Pind = np.abs(Pnorm - Pnorm_des).argmin().astype(int)
    P = PVals[Pind]

    # # FULL RECONSTRUCTION OF 3D CARTESIAN BETA_K FROM 2D SPHERICAL BETA_K (doing actual interpolation in 2D spherical instead of 3D nonlinear cartesian)

    if Lx == 60:
        CSAmp_ds = (qds['Real_CSAmp'] + 1j * qds['Imag_CSAmp']).sel(P=P).isel(tc=-1); CSAmp_ds.attrs = qds.attrs; CSAmp_ds.attrs['Nph'] = qds['Nph'].sel(P=P).isel(t=-1).values
    else:
        CSAmp_ds = (qds['Real_CSAmp'] + 1j * qds['Imag_CSAmp']).sel(P=P).isel(t=-1); CSAmp_ds.attrs = qds.attrs; CSAmp_ds.attrs['Nph'] = qds['Nph'].sel(P=P).isel(t=-1).values

    # Generate data

    # dkxL = 1e-2; dkyL = 1e-2; dkzL = 1e-2
    # linDimList = [(2, 2)]

    dkxL = 5e-2; dkyL = 5e-2; dkzL = 5e-2
    linDimList = [(10, 10)]

    for ldtup in linDimList:
        tupstart = timer()
        linDimMajor, linDimMinor = ldtup
        print('lDM: {0}, lDm: {1}'.format(linDimMajor, linDimMinor))
        interp_ds = pfs.reconstructMomDists(CSAmp_ds, linDimMajor, linDimMinor, dkxL, dkyL, dkzL)
        interp_ds.to_netcdf(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, linDimMajor, linDimMinor))
        tupend = timer()
        print('Total Time: {0}'.format(tupend - tupstart))
        time.sleep(2)
        print('\n')
