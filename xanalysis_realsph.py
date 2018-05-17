import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
import os
import itertools
import pf_dynamic_cart as pfc
import pf_dynamic_sph as pfs
import Grid
from scipy import interpolate


if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # ---- INITIALIZE GRIDS ----

    # (Lx, Ly, Lz) = (30, 30, 30)
    (Lx, Ly, Lz) = (21, 21, 21)
    # (Lx, Ly, Lz) = (12, 12, 12)
    (dx, dy, dz) = (0.375, 0.375, 0.375)
    # (dx, dy, dz) = (0.75, 0.75, 0.75)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    # NGridPoints_cart = 1.37e5

    # Toggle parameters

    toggleDict = {'Location': 'home', 'Dynamics': 'real', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'frohlich', 'Longtime': 'false'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
        animpath = '/home/kis/Dropbox/VariationalResearch/DataAnalysis/figs'
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
        animpath = '/media/kis/Storage/Dropbox/VariationalResearch/DataAnalysis/figs'

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

    if toggleDict['Longtime'] == 'true':
        innerdatapath = innerdatapath + '_longtime'
    elif toggleDict['Longtime'] == 'false':
        innerdatapath = innerdatapath

    # # # Concatenate Individual Datasets

    # ds_list = []; P_list = []; aIBi_list = []; mI_list = []
    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     if filename == 'quench_Dataset.nc':
    #         continue
    #     print(filename)
    #     ds = xr.open_dataset(innerdatapath + '/' + filename)
    #     ds_list.append(ds)
    #     P_list.append(ds.attrs['P'])
    #     aIBi_list.append(ds.attrs['aIBi'])
    #     mI_list.append(ds.attrs['mI'])

    # s = sorted(zip(aIBi_list, P_list, ds_list))
    # g = itertools.groupby(s, key=lambda x: x[0])

    # aIBi_keys = []; aIBi_groups = []; aIBi_ds_list = []
    # for key, group in g:
    #     aIBi_keys.append(key)
    #     aIBi_groups.append(list(group))

    # for ind, group in enumerate(aIBi_groups):
    #     aIBi = aIBi_keys[ind]
    #     _, P_list_temp, ds_list_temp = zip(*group)
    #     ds_temp = xr.concat(ds_list_temp, pd.Index(P_list_temp, name='P'))
    #     aIBi_ds_list.append(ds_temp)

    # ds_tot = xr.concat(aIBi_ds_list, pd.Index(aIBi_keys, name='aIBi'))
    # del(ds_tot.attrs['P']); del(ds_tot.attrs['aIBi']); del(ds_tot.attrs['nu']); del(ds_tot.attrs['gIB'])
    # ds_tot.to_netcdf(innerdatapath + '/quench_Dataset.nc')

    # # Analysis of Total Dataset

    qds = xr.open_dataset(innerdatapath + '/quench_Dataset.nc')
    # qds = xr.open_dataset(innerdatapath + '/P_0.900_aIBi_-6.23.nc')
    tVals = qds['t'].values
    dt = tVals[1] - tVals[0]
    PVals = qds['P'].values
    n0 = qds.attrs['n0']
    gBB = qds.attrs['gBB']
    nu = pfc.nu(gBB)
    mI = qds.attrs['mI']
    mB = qds.attrs['mB']

    aIBi = -10
    qds_aIBi = qds.sel(aIBi=aIBi)

    fig, ax = plt.subplots()
    for P in PVals:
        Nph = qds_aIBi.sel(P=P)['Nph'].values
        dNph = np.diff(Nph)
        ax.plot(dNph / dt)

    plt.show()

    # # # PHONON MODE CHARACTERIZATION (SPHERICAL)

    # CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # list_of_unit_vectors = list(kgrid.arrays.keys())
    # list_of_functions = [lambda k: (2 * np.pi)**(-2) * k**2, np.sin]
    # sphfac = kgrid.function_prod(list_of_unit_vectors, list_of_functions)
    # kDiff = kgrid.diffArray('k')
    # thDiff = kgrid.diffArray('th')

    # kAve_Vals = np.zeros(PVals.size)
    # thAve_Vals = np.zeros(PVals.size)
    # kFWHM_Vals = np.zeros(PVals.size)
    # thFWHM_Vals = np.zeros(PVals.size)
    # PhDen_k_Vec = np.empty(PVals.size, dtype=np.object)
    # PhDen_th_Vec = np.empty(PVals.size, dtype=np.object)
    # CSAmp_ds_inf = CSAmp_ds.isel(t=-1)
    # for Pind, P in enumerate(PVals):
    #     CSAmp = CSAmp_ds_inf.sel(P=P).values
    #     Nph = qds_aIBi.isel(t=-1).sel(P=P)['Nph'].values
    #     PhDen = (1 / Nph) * sphfac * np.abs(CSAmp.reshape(CSAmp.size))**2

    #     PhDen_mat = PhDen.reshape((len(kVec), len(thVec)))
    #     PhDen_k = np.dot(PhDen_mat, thDiff); PhDen_k_Vec[Pind] = PhDen_k
    #     PhDen_th = np.dot(np.transpose(PhDen_mat), kDiff); PhDen_th_Vec[Pind] = PhDen_th

    #     # PhDen_k = kgrid.integrateFunc(PhDen, 'th'); PhDen_k_Vec[Pind] = PhDen_k
    #     # PhDen_th = kgrid.integrateFunc(PhDen, 'k'); PhDen_th_Vec[Pind] = PhDen_th

    #     kAve_Vals[Pind] = np.dot(kVec, PhDen_k * kDiff)
    #     thAve_Vals[Pind] = np.dot(thVec, PhDen_th * thDiff)

    #     kFWHM_Vals[Pind] = pfc.FWHM(kVec, PhDen_k)
    #     thFWHM_Vals[Pind] = pfc.FWHM(thVec, PhDen_th)

    # fig1, ax1 = plt.subplots(1, 2)
    # ax1[0].plot(PVals, kAve_Vals, 'b-', label='Mean')
    # ax1[0].plot(PVals, kFWHM_Vals, 'g-', label='FWHM')
    # ax1[0].legend()
    # ax1[0].set_xlabel('P')
    # ax1[0].set_title('Characteristics of ' + r'$|\vec{k}|$' + ' Distribution of Individual Phonons (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))

    # ax1[1].plot(PVals, thAve_Vals, 'b-', label='Mean')
    # ax1[1].plot(PVals, thFWHM_Vals, 'g-', label='FWHM')
    # ax1[1].legend()
    # ax1[1].set_xlabel('P')
    # ax1[1].set_title('Characteristics of ' + r'$\theta$' + ' Distribution of Individual Phonons (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))

    # fig2, ax2 = plt.subplots()
    # curve2 = ax2.plot(kVec, PhDen_k_Vec[0], color='g', lw=2)[0]
    # P_text2 = ax2.text(0.85, 0.9, 'P: {:.2f}'.format(PVals[0]), transform=ax2.transAxes, color='r')
    # ax2.set_xlim([-0.01, np.max(kVec)])
    # ax2.set_ylim([0, 5])
    # ax2.set_title('Individual Phonon Momentum Magnitude Distribution (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax2.set_ylabel(r'$\int n_{\vec{k}} \cdot d\theta$' + '  where  ' + r'$n_{\vec{k}}=\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2} |\vec{k}|^{2} \sin(\theta)$')

    # ax2.set_xlabel(r'$|\vec{k}|$')

    # def animate2(i):
    #     curve2.set_ydata(PhDen_k_Vec[i])
    #     P_text2.set_text('P: {:.2f}'.format(PVals[i]))
    # anim2 = FuncAnimation(fig2, animate2, interval=1000, frames=range(PVals.size))
    # anim2.save(animpath + '/aIBi_{0}'.format(aIBi) + '_PhononDist_kmag.gif', writer='imagemagick')

    # fig3, ax3 = plt.subplots()
    # curve3 = ax3.plot(thVec, PhDen_th_Vec[0], color='g', lw=2)[0]
    # P_text3 = ax3.text(0.85, 0.9, 'P: {:.2f}'.format(PVals[0]), transform=ax3.transAxes, color='r')
    # ax3.set_xlim([-0.01, np.max(thVec)])
    # ax3.set_ylim([0, 5])
    # ax3.set_title('Individual Phonon Momentum Direction Distribution (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax3.set_ylabel(r'$\int n_{\vec{k}} \cdot d|\vec{k}|$' + '  where  ' + r'$n_{\vec{k}}=\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2} |\vec{k}|^{2} \sin(\theta)$')
    # ax3.set_xlabel(r'$\theta$')

    # def animate3(i):
    #     curve3.set_ydata(PhDen_th_Vec[i])
    #     P_text3.set_text('P: {:.2f}'.format(PVals[i]))
    # anim3 = FuncAnimation(fig3, animate3, interval=1000, frames=range(PVals.size))
    # anim3.save(animpath + '/aIBi_{0}'.format(aIBi) + '_PhononDist_theta.gif', writer='imagemagick')

    # plt.draw()
    # plt.show()
