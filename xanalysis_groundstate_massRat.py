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

    NGridPoints = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    # NGridPoints_cart = 1.37e5

    # Basic parameters

    mI = 5
    mB = 1

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'imaginary', 'Coupling': 'twophonon', 'Grid': 'cartesian', 'Longtime': 'false'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints, mI / mB)
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints, mI / mB)
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/regal/demler_lab/kis/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints, mI / mB)

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

    # # # Analysis of Total Dataset
    # qds = xr.open_dataset(innerdatapath + '/quench_Dataset.nc')
    # PVals = qds['P'].values
    # tVals = qds['t'].values
    # n0 = qds.attrs['n0']
    # gBB = qds.attrs['gBB']
    # nu = pfc.nu(gBB)
    # mI = qds.attrs['mI']
    # mB = qds.attrs['mB']

    # aIBi = -10
    # qds_aIBi = qds.sel(aIBi=aIBi)

    # # IMPURITY DISTRIBUTION CHARACTERIZATION (CARTESIAN)

    # nPIm_FWHM_Vals = np.zeros(PVals.size)
    # nPIm_distPeak_Vals = np.zeros(PVals.size)
    # nPIm_deltaPeak_Vals = np.zeros(PVals.size)
    # nPIm_Tot_Vals = np.zeros(PVals.size)
    # nPIm_Vec = np.empty(PVals.size, dtype=np.object)
    # PIm_Vec = np.empty(PVals.size, dtype=np.object)
    # fig, ax = plt.subplots()
    # for ind, P in enumerate(PVals):
    #     qds_nPIm_inf = qds_aIBi['nPI_mag'].sel(P=P).isel(t=-1).dropna('PI_mag')
    #     PIm_Vals = qds_nPIm_inf.coords['PI_mag'].values
    #     dPIm = PIm_Vals[1] - PIm_Vals[0]

    #     # # Plot nPIm(t=inf)
    #     qds_nPIm_inf.plot(ax=ax, label='P: {:.1f}'.format(P))
    #     nPIm_Vec[ind] = qds_nPIm_inf.values
    #     PIm_Vec[ind] = PIm_Vals

    #     # # Calculate nPIm(t=inf) normalization
    #     nPIm_Tot_Vals[ind] = np.sum(qds_nPIm_inf.values * dPIm) + qds_aIBi.sel(P=P).isel(t=-1)['mom_deltapeak'].values

    #     # Calculate FWHM, distribution peak, and delta peak
    #     nPIm_FWHM_Vals[ind] = pfc.FWHM(PIm_Vals, qds_nPIm_inf.values)
    #     nPIm_distPeak_Vals[ind] = np.max(qds_nPIm_inf.values)
    #     nPIm_deltaPeak_Vals[ind] = qds_aIBi.sel(P=P).isel(t=-1)['mom_deltapeak'].values

    # # Plot nPIm(t=inf)
    # ax.plot(mI * nu * np.ones(PIm_Vals.size), np.linspace(0, 1, PIm_Vals.size), 'k--', label=r'$m_{I}c$')
    # ax.legend()
    # ax.set_xlabel(r'$|P_{I}|$')
    # ax.set_ylabel(r'$n_{|P_{I}|}$')
    # ax.set_title('Ground state impurity distribution (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # # plt.show()

    # # # Plot characterization of nPIm(t=inf)
    # # ax.plot(PVals, nPIm_FWHM_Vals, 'b-', label='Incoherent Dist FWHM')
    # # ax.plot(PVals, nPIm_distPeak_Vals, 'g-', label='Incoherent Dist Peak')
    # # ax.plot(PVals, nPIm_deltaPeak_Vals, 'r-', label='Delta Peak (Z-factor)')
    # # ax.legend()
    # # ax.set_xlabel('$P$')
    # # ax.set_title(r'$n_{|P_{I}|}$' + ' Characterization (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # # plt.show()

    # fig2, ax2 = plt.subplots()
    # ax2.plot(mI * nu * np.ones(PIm_Vals.size), np.linspace(0, 1, PIm_Vals.size), 'k--', label=r'$m_{I}c$')
    # curve = ax2.plot(PIm_Vec[0], nPIm_Vec[0], color='k', lw=2, label='')[0]
    # line = ax2.plot(PVals[0] * np.ones(PIm_Vals.size), np.linspace(0, nPIm_deltaPeak_Vals[0], PIm_Vals.size), 'go', label='')[0]
    # P_text = ax2.text(0.85, 0.85, 'P: {:.2f}'.format(PVals[0]), transform=ax2.transAxes, color='r')
    # norm_text = ax2.text(0.7, 0.8, r'$\int n_{|\vec{P_{I}}|} d|\vec{P_{I}}| = $' + '{:.3f}'.format(nPIm_Tot_Vals[0]), transform=ax.transAxes, color='b')

    # ax2.legend()
    # ax2.set_xlim([-0.01, np.max(PIm_Vec[0])])
    # ax2.set_ylim([0, 1.2])
    # ax2.set_title('Impurity Momentum Magnitude Distribution (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax2.set_ylabel(r'$n_{|\vec{P_{I}}|}$')
    # ax2.set_xlabel(r'$|\vec{P_{I}}|$')

    # def animate2(i):
    #     curve.set_xdata(PIm_Vec[i])
    #     curve.set_ydata(nPIm_Vec[i])
    #     line.set_xdata(PVals[i])
    #     line.set_ydata(np.linspace(0, nPIm_deltaPeak_Vals[i], PIm_Vals.size))
    #     P_text.set_text('P: {:.2f}'.format(PVals[i]))
    #     norm_text.set_text(r'$\int n_{|\vec{P_{I}}|} d|\vec{P_{I}}| = $' + '{:.3f}'.format(nPIm_Tot_Vals[i]))

    # anim2 = FuncAnimation(fig2, animate2, interval=1000, frames=range(PVals.size))
    # anim2.save(animpath + '/aIBi_{0}'.format(aIBi) + '_ImpDist.gif', writer='imagemagick')
    # plt.show()

    # # DISTRIBUTION CHARACTERIZATION SATURATION

    # nPIm_FWHM_Vals = np.zeros((PVals.size, tVals.size))
    # nPIm_distPeak_Vals = np.zeros((PVals.size, tVals.size))
    # nPIm_deltaPeak_Vals = np.zeros((PVals.size, tVals.size))

    # for Pind, P in enumerate(PVals):
    #     for tind, t in enumerate(tVals):
    #         qds_nPIm_inf = qds_aIBi['nPI_mag'].sel(P=P, t=t).dropna('PI_mag')
    #         PIm_Vals = qds_nPIm_inf.coords['PI_mag'].values
    #         dPIm = PIm_Vals[1] - PIm_Vals[0]

    #         # # Plot nPIm(t=inf)
    #         # qds_nPIm_inf.plot(ax=ax, label='P: {:.1f}'.format(P))

    #         # # Calculate nPIm(t=inf) normalization
    #         nPIm_Tot = np.sum(qds_nPIm_inf.values * dPIm) + qds_aIBi.sel(P=P, t=t)['mom_deltapeak'].values

    #         # Calculate FWHM, distribution peak, and delta peak
    #         nPIm_FWHM_Vals[Pind, tind] = pfc.FWHM(PIm_Vals, qds_nPIm_inf.values)
    #         nPIm_distPeak_Vals[Pind, tind] = np.max(qds_nPIm_inf.values)
    #         nPIm_deltaPeak_Vals[Pind, tind] = qds_aIBi.sel(P=P, t=t)['mom_deltapeak'].values

    #     # fig, ax = plt.subplots()
    #     # # ax.plot(tVals, nPIm_FWHM_Vals, 'b-', label='Incoherent Dist FWHM')
    #     # ax.plot(tVals, nPIm_distPeak_Vals, 'g-', label='Incoherent Dist Peak')
    #     # ax.plot(tVals, nPIm_deltaPeak_Vals, 'r-', label='Delta Peak (Z-factor)')
    #     # ax.legend()
    #     # ax.set_xscale('log')
    #     # ax.set_xlabel('Imaginary Time')
    #     # ax.set_yscale('log')
    #     # ax.set_title(r'$n_{|P_{I}|}$' + ' Characteristics Saturation (' + r'$aIB^{-1}=$' + '{0}'.format(aIBi) + ', P={:.2f})'.format(P))
    #     # plt.show()

    # fig, ax = plt.subplots()
    # quadFWHM = ax.pcolormesh(tVals, PVals, nPIm_FWHM_Vals, norm=colors.LogNorm())
    # ax.set_xscale('log')
    # ax.set_xlabel('Imaginary Time')
    # ax.set_ylabel('P')
    # ax.set_title('Incoherent Dist FWHM (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # fig.colorbar(quadFWHM, ax=ax, extend='max')
    # plt.show()

    # fig, ax = plt.subplots()
    # quaddistP = ax.pcolormesh(tVals, PVals, nPIm_distPeak_Vals, norm=colors.LogNorm())
    # ax.set_xscale('log')
    # ax.set_xlabel('Imaginary Time')
    # ax.set_ylabel('P')
    # ax.set_title('Incoherent Dist Peak (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # fig.colorbar(quaddistP, ax=ax, extend='max')
    # plt.show()

    # fig, ax = plt.subplots()
    # quaddeltP = ax.pcolormesh(tVals, PVals, nPIm_deltaPeak_Vals, norm=colors.LogNorm())
    # ax.set_xscale('log')
    # ax.set_xlabel('Imaginary Time')
    # ax.set_ylabel('P')
    # ax.set_title('Delta Peak (Z-factor) (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # fig.colorbar(quaddeltP, ax=ax, extend='max')
    # plt.show()
