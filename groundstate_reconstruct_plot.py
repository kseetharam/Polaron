import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pf_dynamic_cart as pfc
import Grid
from scipy import interpolate
from timeit import default_timer as timer


if __name__ == "__main__":

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (105, 105, 105)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # Toggle parameters

    toggleDict = {'Location': 'home', 'Dynamics': 'imaginary', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon'}

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

    # # Analysis of Total Dataset
    interpdatapath = innerdatapath + '/interp'
    aIBi = -10
    P = 1.54

    linDimList = [(0.1, 0.01), (0.11, 0.02)]
    linDimMajor, linDimMinor = linDimList[1]

    # Plot

    interp_ds = xr.open_dataset(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, linDimMajor, linDimMinor))
    kxL = interp_ds['kx'].values
    kzL = interp_ds['kz'].values
    xL = interp_ds['x'].values
    zL = interp_ds['z'].values
    kxLg_xz_slice, kzLg_xz_slice = np.meshgrid(kxL, kzL, indexing='ij')
    xLg_xz_slice, zLg_xz_slice = np.meshgrid(xL, zL, indexing='ij')
    PhDenLg_xz_slice = interp_ds['PhDen_xz'].values
    np_xz_slice = interp_ds['np_xz'].values
    na_xz_slice = interp_ds['na_xz'].values

    # Interpolate 2D slice of position distribution
    posmult = 5
    poslinDim = 2500

    zL_xz_slice_interp = np.linspace(-1 * poslinDim, poslinDim, posmult * zL.size); xL_xz_slice_interp = np.linspace(-1 * poslinDim, poslinDim, posmult * xL.size)
    xLg_xz_slice_interp, zLg_xz_slice_interp = np.meshgrid(xL_xz_slice_interp, zL_xz_slice_interp, indexing='ij')
    np_xz_slice_interp = interpolate.griddata((xLg_xz_slice.flatten(), zLg_xz_slice.flatten()), np_xz_slice.flatten(), (xLg_xz_slice_interp, zLg_xz_slice_interp), method='cubic')
    na_xz_slice_interp = interpolate.griddata((xLg_xz_slice.flatten(), zLg_xz_slice.flatten()), na_xz_slice.flatten(), (xLg_xz_slice_interp, zLg_xz_slice_interp), method='cubic')

    xLg_xz_slice = xLg_xz_slice_interp
    zLg_xz_slice = zLg_xz_slice_interp
    np_xz_slice = np_xz_slice_interp
    na_xz_slice = na_xz_slice_interp

    # All Plotting:

    fig2, ax2 = plt.subplots()
    quad2 = ax2.pcolormesh(kzLg_xz_slice, kxLg_xz_slice, PhDenLg_xz_slice[:-1, :-1], norm=colors.LogNorm(vmin=1e-5, vmax=np.max(PhDenLg_xz_slice)), cmap='inferno')
    ax2.set_xlim([-1 * linDimMajor, linDimMajor])
    ax2.set_ylim([-1 * linDimMinor, linDimMinor])
    ax2.set_xlabel('kz (Impurity Propagation Direction)')
    ax2.set_ylabel('kx')
    ax2.set_title('Individual Phonon Momentum Distribution (Interp)')
    fig2.colorbar(quad2, ax=ax2, extend='both')

    fig3, ax3 = plt.subplots()
    quad3 = ax3.pcolormesh(zLg_xz_slice, xLg_xz_slice, np_xz_slice[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(np_xz_slice)), vmax=np.max(np_xz_slice)), cmap='inferno')
    ax3.set_xlabel('z (Impurity Propagation Direction)')
    ax3.set_ylabel('x')
    ax3.set_title('Individual Phonon Position Distribution (Interp)')
    fig3.colorbar(quad3, ax=ax3, extend='both')

    fig4, ax4 = plt.subplots()
    quad4 = ax4.pcolormesh(zLg_xz_slice, xLg_xz_slice, na_xz_slice[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(na_xz_slice)), vmax=np.max(na_xz_slice)), cmap='inferno')
    ax4.set_xlabel('z (Impurity Propagation Direction)')
    ax4.set_ylabel('x')
    ax4.set_title('Individual Atom Position Distribution (Interp)')
    fig4.colorbar(quad4, ax=ax4, extend='both')

    plt.show()
