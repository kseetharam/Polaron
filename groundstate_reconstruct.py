import numpy as np
import xarray as xr
import pf_dynamic_sph as pfs
from timeit import default_timer as timer
import time
# import matplotlib
# import matplotlib.pyplot as plt


if __name__ == "__main__":

    # ---- INITIALIZE GRIDS ----

    # (Lx, Ly, Lz) = (105, 105, 105)
    # (dx, dy, dz) = (0.375, 0.375, 0.375)

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'imaginary', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, 1)
        animpath = '/home/kis/Dropbox/VariationalResearch/DataAnalysis/figs'
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, 1)
        animpath = '/media/kis/Storage/Dropbox/VariationalResearch/DataAnalysis/figs'
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/regal/demler_lab/kis/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, 1)
        animpath = ''

    if toggleDict['Dynamics'] == 'real':
        innerdatapath = datapath + '/redyn'
        animpath = animpath + '/rdyn'
    elif toggleDict['Dynamics'] == 'imaginary':
        innerdatapath = datapath + '/imdyn'
        animpath = animpath + '/idyn'

    if toggleDict['Grid'] == 'cartesian':
        innerdatapath = innerdatapath + '_cart'
    elif toggleDict['Grid'] == 'spherical':
        innerdatapath = innerdatapath + '_spherical'

    if toggleDict['Coupling'] == 'frohlich':
        innerdatapath = innerdatapath + '_froh'
        animpath = animpath + '_frohlich'
    elif toggleDict['Coupling'] == 'twophonon':
        innerdatapath = innerdatapath
        animpath = animpath + '_twophonon'

    interpdatapath = innerdatapath + '/interp'

    # # Analysis of Total Dataset

    aIBi = -10
    Pnorm_des = 2.0
    qds = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    n0 = qds.attrs['n0']; gBB = qds.attrs['gBB']; mI = qds.attrs['mI']; mB = qds.attrs['mB']
    nu = np.sqrt(n0 * gBB / mB)
    mc = mI * nu

    PVals = qds['P'].values
    Pnorm = PVals / mc
    Pind = np.abs(Pnorm - Pnorm_des).argmin().astype(int)
    P = PVals[Pind]

    # # FULL RECONSTRUCTION OF 3D CARTESIAN BETA_K FROM 2D SPHERICAL BETA_K (doing actual interpolation in 2D spherical instead of 3D nonlinear cartesian)

    CSAmp_ds = (qds['Real_CSAmp'] + 1j * qds['Imag_CSAmp']).sel(P=P).isel(t=-1); CSAmp_ds.attrs = qds.attrs; CSAmp_ds.attrs['Nph'] = qds['Nph'].sel(P=P).isel(t=-1).values

    # Generate data

    # dkxL = 1e-4; dkyL = 1e-4; dkzL = 1e-3
    # linDimList = [(0.1, 0.01)]

    # dkxL = 1e-3; dkyL = 1e-3; dkzL = 1e-3
    # # linDimList = [(0.1, 0.1), (0.2, 0.2), (0.5, 0.5), (1, 1), (1.5, 1.5), (2, 2), (2.5, 2.5), (3, 3), (3.5, 3.5), (4, 4), (4.5, 4.5), (5, 5), (5.5, 5.5), (6, 6), (6.5, 6.5)]
    # linDimList = [(0.2, 0.2)]

    dkxL = 1e-2; dkyL = 1e-2; dkzL = 1e-2
    linDimList = [(2, 2)]

    # dkxL = 1e-3; dkyL = 1e-3; dkzL = 1e-3
    # linDimList = [(1, 1)]

    for ldtup in linDimList:
        tupstart = timer()
        linDimMajor, linDimMinor = ldtup
        print('lDM: {0}, lDm: {1}'.format(linDimMajor, linDimMinor))
        interp_ds = pfs.reconstructDistributions(CSAmp_ds, linDimMajor, linDimMinor, dkxL, dkyL, dkzL)
        interp_ds.to_netcdf(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, linDimMajor, linDimMinor))
        tupend = timer()
        print('Total Time: {0}'.format(tupend - tupstart))
        time.sleep(2)
        print('\n')
