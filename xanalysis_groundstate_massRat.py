import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
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

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    NGridPoints = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    # NGridPoints_cart = 1.37e5

    massRat_Vals = [1, 2, 5, 10]
    toggleDict = {'Location': 'work'}
    datapathDict = {}
    for mR in massRat_Vals:
        if toggleDict['Location'] == 'home':
            datapathDict[mR] = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}/imdyn_cart'.format(NGridPoints, mR)
        elif toggleDict['Location'] == 'work':
            datapathDict[mR] = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}/imdyn_cart'.format(NGridPoints, mR)

    # # # Concatenate Individual Datasets (everything)

    # mR = 10
    # innerdatapath = datapathDict[mR]
    # ds_list = []; P_list = []; aIBi_list = []; mI_list = []
    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     print(ind)
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

    # # # Concatenate Individual Datasets (aIBi specific - note that for some reason there is a chunk of memory from the initial for loop through filenames that is not being freed up)

    # mR = 1
    # innerdatapath = datapathDict[mR]
    # aIBi_List = [-10, -5, -2]
    # # aIBi_List = [-2]
    # for aIBi in aIBi_List:
    #     ds_list = []; P_list = []; mI_list = []
    #     for ind, filename in enumerate(os.listdir(innerdatapath)):
    #         if filename[0:14] == 'quench_Dataset':
    #             continue
    #         ds = xr.open_dataset(innerdatapath + '/' + filename)
    #         aIBi_temp = ds.attrs['aIBi']
    #         if aIBi_temp != aIBi:
    #             continue
    #         print(filename)
    #         ds_list.append(ds)
    #         P_list.append(ds.attrs['P'])
    #         mI_list.append(ds.attrs['mI'])

    #     s = sorted(zip(P_list, ds_list))
    #     g = itertools.groupby(s, key=lambda x: x[0])

    #     P_keys = []; P_ds_list = []; aIBi_ds_list = []
    #     for key, group in g:
    #         P_temp_list, ds_temp_list = zip(*list(group))
    #         P_keys.append(key)  # note that key = P_temp_list[0]
    #         P_ds_list.append(ds_temp_list[0])

    #     with xr.concat(P_ds_list, pd.Index(P_keys, name='P')) as ds_tot:
    #         # ds_tot = xr.concat(P_ds_list, pd.Index(P_keys, name='P'))
    #         del(ds_tot.attrs['P']); del(ds_tot.attrs['nu']); del(ds_tot.attrs['gIB'])
    #         ds_tot.to_netcdf(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))

    # # Analysis of Total Dataset

    aIBi = -2
    qdsDict = {}
    for mR in massRat_Vals:
        # qdsDict[mR] = xr.open_dataset(datapathDict[mR] + '/quench_Dataset.nc').sel(aIBi=aIBi)
        qdsDict[mR] = xr.open_dataset(datapathDict[mR] + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))

    PVals = qdsDict[1]['P'].values
    tVals = qdsDict[1]['t'].values
    n0 = qdsDict[1].attrs['n0']
    gBB = qdsDict[1].attrs['gBB']
    nu = pfc.nu(gBB)
    mI = qdsDict[1].attrs['mI']
    mB = qdsDict[1].attrs['mB']

    # IMPURITY DISTRIBUTION CHARACTERIZATION (CARTESIAN)
    nPIm_FWHM_Dict = {}
    nPIm_distPeak_Dict = {}
    nPIm_deltaPeak_Dict = {}
    nPIm_Tot_Dict = {}
    for mind, mR in enumerate(massRat_Vals):
        nPIm_FWHM_Vals = np.zeros(PVals.size)
        nPIm_distPeak_Vals = np.zeros(PVals.size)
        nPIm_deltaPeak_Vals = np.zeros(PVals.size)
        nPIm_Tot_Vals = np.zeros(PVals.size)
        nPIm_Vec = np.empty(PVals.size, dtype=np.object)
        PIm_Vec = np.empty(PVals.size, dtype=np.object)
        for ind, P in enumerate(PVals):
            qds_nPIm_inf = qdsDict[mR]['nPI_mag'].sel(P=P).isel(t=-1).dropna('PI_mag')
            PIm_Vals = qds_nPIm_inf.coords['PI_mag'].values
            dPIm = PIm_Vals[1] - PIm_Vals[0]

            # # Calculate nPIm(t=inf) normalization
            nPIm_Tot_Vals[ind] = np.sum(qds_nPIm_inf.values * dPIm) + qdsDict[mR].sel(P=P).isel(t=-1)['mom_deltapeak'].values

            # Calculate FWHM, distribution peak, and delta peak
            nPIm_FWHM_Vals[ind] = pfc.FWHM(PIm_Vals, qds_nPIm_inf.values)
            nPIm_distPeak_Vals[ind] = np.max(qds_nPIm_inf.values)
            nPIm_deltaPeak_Vals[ind] = qdsDict[mR].sel(P=P).isel(t=-1)['mom_deltapeak'].values

        # Plot characterization of nPIm(t=inf)
        nPIm_FWHM_Dict[mR] = nPIm_FWHM_Vals
        nPIm_distPeak_Dict[mR] = nPIm_distPeak_Vals
        nPIm_deltaPeak_Dict[mR] = nPIm_deltaPeak_Vals

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    legend_elements = []
    fig, ax = plt.subplots()
    for mind, mR in enumerate(massRat_Vals):
        mIc = mR * mB * nu
        mininds = np.argpartition(nPIm_FWHM_Dict[mR], 2)[:2]
        Pcrit = np.average(PVals[mininds])  # estimate of critical momentum based on two minimum values of FWHM
        if mR == 10:
            Pcrit = mIc
        # Pcrit = mIc
        ax.plot(PVals / (Pcrit), nPIm_FWHM_Dict[mR], color=colors[mind], linestyle='-')
        ax.plot(PVals / (Pcrit), nPIm_distPeak_Dict[mR], color=colors[mind], linestyle='--')
        ax.plot(PVals / (Pcrit), nPIm_deltaPeak_Dict[mR], color=colors[mind], linestyle=':')
        legend_elements.append(Line2D([0], [0], color=colors[mind], lw=2, label=r'$\frac{m_{I}}{m_{B}}=$' + '{:.1f}'.format(mR)))

    legend_elements.append(Line2D([0], [0], color='k', linestyle='-', lw=1, label='Incoherent Dist FWHM'))
    legend_elements.append(Line2D([0], [0], color='k', linestyle='--', lw=1, label='Incoherent Dist Peak'))
    legend_elements.append(Line2D([0], [0], color='k', linestyle=':', lw=1, label='Delta Peak (Z-factor)'))
    ax.legend(handles=legend_elements)
    ax.set_xlabel(r'$\frac{P}{P_{crit}}$')
    ax.set_xlabel(r'$\frac{P}{m_{I}c_{BEC}}$')
    ax.set_title(r'$n_{|P_{I}|}$' + ' Characterization (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    plt.show()

    # fig2, ax2, = plt.subplots()
    # Pinit = 6
    # for mind, mR in enumerate(massRat_Vals):
    #     qds_nPIm_inf = qdsDict[mR]['nPI_mag'].sel(P=Pinit, method='nearest').isel(t=-1).dropna('PI_mag')
    #     Pinit = 1 * qds_nPIm_inf['P'].values
    #     PIm_Vals = qds_nPIm_inf.coords['PI_mag'].values
    #     ax2.plot(PIm_Vals, qds_nPIm_inf.values, color=colors[mind], linestyle='-', label=r'$\frac{m_{I}}{m_{B}}=$' + '{:.1f}'.format(mR))
    # ax2.set_xlabel(r'$|P_{I}|$')
    # ax2.set_title(r'$n_{|P_{I}|}$' + ' (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$P=$' + '{:.2f})'.format(Pinit))
    # ax2.legend()
    # plt.show()

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
