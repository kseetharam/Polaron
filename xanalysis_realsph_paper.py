import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import writers
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import os
import itertools
import pf_dynamic_cart as pfc
import pf_dynamic_sph as pfs
import Grid
import warnings
from scipy import interpolate
from scipy.optimize import curve_fit, OptimizeWarning
from timeit import default_timer as timer


if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})
    mpegWriter = writers['ffmpeg'](fps=20, bitrate=1800)

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)
    # (dx, dy, dz) = (0.25, 0.25, 0.25)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    # NGridPoints_cart = 1.37e5

    massRat = 1.0
    IRrat = 1

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'real', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'frohlich', 'ReducedInterp': 'false', 'kGrid_ext': 'false'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, massRat)
        animpath = '/home/kis/Dropbox/VariationalResearch/DataAnalysis/figs'
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, massRat)
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
        innerdatapath = innerdatapath + '_froh_new'
        animpath = animpath + '_frohlich'
    elif toggleDict['Coupling'] == 'twophonon':
        innerdatapath = innerdatapath
        animpath = animpath + '_twophonon'

    IRrat_Vals = np.array([2, 5, 10, 100, 4e3])
    qdatapath_Dict = {1.0: innerdatapath}

    for IRrat_val in IRrat_Vals:
        qdatapath_Dict[IRrat_val] = innerdatapath[0:-4] + '_IRcuts' + '/IRratio_{:.1E}'.format(IRrat_val)

    qdatapath = qdatapath_Dict[IRrat]

    # # # Concatenate Individual Datasets (aIBi specific)

    # print(innerdatapath)

    # aIBi_List = [-10.0, -5.0, -2.0]
    # for aIBi in aIBi_List:
    #     ds_list = []; P_list = []; mI_list = []
    #     for ind, filename in enumerate(os.listdir(innerdatapath)):
    #         if filename[0:14] == 'quench_Dataset':
    #             continue
    #         if filename[0:6] == 'interp':
    #             continue
    #         if filename[0:2] == 'mm':
    #             continue
    #         if float(filename[13:-3]) != aIBi:
    #             continue
    #         print(filename)
    #         ds = xr.open_dataset(innerdatapath + '/' + filename)
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

    # # # Concatenate Individual Datasets (aIBi specific, IRcuts)

    # IRrat_Vals = [2, 5, 10, 100, 4e3]
    # aIBi_List = [-10.0, -5.0, -2.0]
    # for IRrat in IRrat_Vals:
    #     IRdatapath = innerdatapath + '/IRratio_{:.1E}'.format(IRrat)
    #     for aIBi in aIBi_List:
    #         ds_list = []; P_list = []; mI_list = []
    #         for ind, filename in enumerate(os.listdir(IRdatapath)):
    #             if filename[0:14] == 'quench_Dataset':
    #                 continue
    #             if filename[0:6] == 'interp':
    #                 continue
    #             if filename[0:2] == 'mm':
    #                 continue
    #             if float(filename[13:-3]) != aIBi:
    #                 continue
    #             print(filename)
    #             ds = xr.open_dataset(IRdatapath + '/' + filename)
    #             ds_list.append(ds)
    #             P_list.append(ds.attrs['P'])
    #             mI_list.append(ds.attrs['mI'])

    #         s = sorted(zip(P_list, ds_list))
    #         g = itertools.groupby(s, key=lambda x: x[0])

    #         P_keys = []; P_ds_list = []; aIBi_ds_list = []
    #         for key, group in g:
    #             P_temp_list, ds_temp_list = zip(*list(group))
    #             P_keys.append(key)  # note that key = P_temp_list[0]
    #             P_ds_list.append(ds_temp_list[0])

    #         with xr.concat(P_ds_list, pd.Index(P_keys, name='P')) as ds_tot:
    #             # ds_tot = xr.concat(P_ds_list, pd.Index(P_keys, name='P'))
    #             del(ds_tot.attrs['P']); del(ds_tot.attrs['nu']); del(ds_tot.attrs['gIB'])
    #             ds_tot.to_netcdf(IRdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))

    # # Analysis of Total Dataset

    aIBi = -2.0

    qds = xr.open_dataset(qdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    qds_aIBi = qds

    PVals = qds['P'].values
    tVals = qds['t'].values
    n0 = qds.attrs['n0']
    gBB = qds.attrs['gBB']
    mI = qds.attrs['mI']
    mB = qds.attrs['mB']
    nu = np.sqrt(n0 * gBB / mB)
    mc = mI * nu
    aBB = (mB / (4 * np.pi)) * gBB
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)
    tscale = xi / nu
    Pnorm = PVals / mc

    kArray = qds.coords['k'].values
    k0 = kArray[0]
    kf = kArray[-1]

    print(mI / mB, IRrat)
    IR_lengthscale = 1 / (k0 / (2 * np.pi)) / xi
    UV_lengthscale = 1 / (kf / (2 * np.pi)) / xi

    print(k0, 1 / IR_lengthscale, IR_lengthscale)
    print(kf, 1 / UV_lengthscale, UV_lengthscale)

    # aIBi_Vals = np.array([-10.0, -5.0, -2.0, -1.0, -0.75, -0.5])
    aIBi_Vals = np.array([-10.0, -5.0, -2.0])

    # # # # S(t) AND P_Imp CURVES

    # tau = 100
    # tsVals = tVals[tVals < tau]
    # qds_aIBi_ts = qds_aIBi.sel(t=tsVals)

    # # print(Pnorm)
    # Pnorm_des = np.array([0.1, 0.5, 0.8, 1.3, 1.35, 1.8, 3.0, 5.0])
    # # Pnorm_des = np.array([0.1, 0.5, 0.8, 1.3, 1.6, 2.3, 3.0])
    # # Pnorm_des = np.array([0.1, 0.5, 0.8, 1.0, 1.1, 1.3, 1.8, 3.0])

    # Pinds = np.zeros(Pnorm_des.size, dtype=int)
    # for Pn_ind, Pn in enumerate(Pnorm_des):
    #     Pinds[Pn_ind] = np.abs(Pnorm - Pn).argmin().astype(int)

    # fig, axes = plt.subplots(nrows=2, ncols=1)
    # for indP in Pinds:
    #     P = PVals[indP]
    #     DynOv = np.abs(qds_aIBi_ts.isel(P=indP)['Real_DynOv'].values + 1j * qds_aIBi_ts.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #     PImp = P - qds_aIBi_ts.isel(P=indP)['Pph'].values

    #     tfmask = tsVals > 60
    #     tfVals = tsVals[tfmask]
    #     z = np.polyfit(np.log(tfVals), np.log(DynOv[tfmask]), deg=1)
    #     tfLin = tsVals[tsVals > 10]
    #     fLin = np.exp(z[1]) * tfLin**(z[0])

    #     axes[0].plot(tsVals / tscale, DynOv, label='{:.2f}'.format(P / mc))
    #     axes[0].plot(tfLin / tscale, fLin, 'k--', label='')
    #     axes[1].plot(tsVals / tscale, PImp, label='{:.2f}'.format(P / mc))

    # axes[0].legend(title=r'$\frac{P}{m_{I}c_{BEC}}$', loc=3, ncol=2)
    # axes[0].set_xscale('log')
    # axes[0].set_yscale('log')
    # axes[0].set_xlim([1e-1, 1e2])
    # axes[0].set_title('Loschmidt Echo (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # axes[0].set_ylabel(r'$|S(t)|$')
    # axes[0].set_xlabel(r'$t$ [$\frac{\xi}{c}$]')

    # axes[1].plot(tsVals / tscale, mc * np.ones(tsVals.size), 'k--', label='$m_{I}c_{BEC}$')
    # axes[1].legend(title=r'$\frac{P}{m_{I}c_{BEC}}$', loc=1, ncol=2)
    # axes[1].set_xlim([-1, 100])
    # axes[1].set_title('Average Impurity Momentum (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # axes[1].set_ylabel(r'$<P_{I}>$')
    # axes[1].set_xlabel(r'$t$ [$\frac{\xi}{c}$]')

    # fig.tight_layout()
    # plt.show()

    # # # # S(t) AND P_Imp EXPONENTS

    # aIBi_des = np.array([-10.0, -5.0, -2.0])  # Data for stronger interactions (-1.0, -0.75, -0.5) is too noisy to get fits
    # # Another note: The fit for P_{Imp} is also difficult for anything other than very weak interactions -> this is probably because of the diverging convergence time to mI*c due to arguments in Nielsen

    # PVals = PVals[PVals <= 3.0]
    # Pnorm = PVals / mc

    # def powerfunc(t, a, b):
    #     return b * t**(-1 * a)

    # tmin = 40
    # tmax = 60
    # tfVals = tVals[(tVals <= tmax) * (tVals >= tmin)]
    # rollwin = 10

    # fig, ax = plt.subplots()
    # for inda, aIBi in enumerate(aIBi_des):
    #     qds_aIBi = xr.open_dataset(qdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #     qds_aIBi_ts = qds_aIBi.sel(t=tfVals)
    #     DynOv_Exponents = np.zeros(PVals.size)
    #     PImp_Exponents = np.zeros(PVals.size)

    #     for indP, P in enumerate(PVals):
    #         DynOv_raw = np.abs(qds_aIBi_ts.isel(P=indP)['Real_DynOv'].values + 1j * qds_aIBi_ts.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #         DynOv_ds = xr.DataArray(DynOv_raw, coords=[tfVals], dims=['t'])

    #         DynOv_ds = DynOv_ds.rolling(t=rollwin, center=True).mean().dropna('t')
    #         Pph_ds = qds_aIBi_ts.isel(P=indP)['Pph'].rolling(t=rollwin, center=True).mean().dropna('t')

    #         DynOv_Vals = DynOv_ds.values
    #         tDynOv_Vals = DynOv_ds['t'].values

    #         PImpc_Vals = (P - Pph_ds.values) - mc
    #         tPImpc_Vals = Pph_ds['t'].values

    #         with warnings.catch_warnings():
    #             warnings.simplefilter("error", OptimizeWarning)
    #             try:
    #                 Sopt, Scov = curve_fit(powerfunc, tDynOv_Vals, DynOv_Vals)
    #                 PIopt, PIcov = curve_fit(powerfunc, tPImpc_Vals, PImpc_Vals)
    #                 DynOv_Exponents[indP] = Sopt[0]
    #                 PImp_Exponents[indP] = PIopt[0]

    #                 if Sopt[0] < 0:
    #                     # DynOv_Exponents[indP] = np.nan
    #                     DynOv_Exponents[indP] = 0
    #                 if PIopt[0] < 0:
    #                     # PImp_Exponents[indP] = np.nan
    #                     PImp_Exponents[indP] = 0

    #             except OptimizeWarning:
    #                 DynOv_Exponents[indP] = 0
    #                 PImp_Exponents[indP] = 0

    #             except RuntimeError:
    #                 DynOv_Exponents[indP] = 0
    #                 PImp_Exponents[indP] = 0

    #     ax.plot(Pnorm, DynOv_Exponents, marker='x', label='{:.1f}'.format(aIBi))

    # ax.set_xlabel(r'$\frac{P}{m_{I}c_{BEC}}$')
    # ax.set_ylabel(r'$\gamma$' + ' for ' + r'$|S(t)|\propto t^{-\gamma}$')
    # ax.set_title('Long Time Power-Law Behavior of Loschmidt Echo')
    # ax.legend(title=r'$a_{IB}^{-1}$', loc=2)

    # # ax.plot(Pnorm, DynOv_Exponents, 'bo', markerfacecolor='none', label='Loschmidt Echo (' + r'$|S(t)|$' + ')')
    # # ax.set_xlabel(r'$\frac{P}{m_{I}c_{BEC}}$')
    # # ax.plot(Pnorm, PImp_Exponents, 'rx', label='Average Impurity Momentum (' + r'$<P_{I}>$' + ')')
    # # ax.legend(loc=2)
    # # ax.set_ylabel(r'$\gamma$' + ' for Observable ' + r'$\propto t^{-\gamma}$')
    # # ax.set_title('Long Time Power-Law Behavior of Observables')

    # plt.show()

    # # # RF SPECTRA

    prefac = -1
    prefac2 = 1

    # # Nph (SPHERICAL)

    Pind = np.argmin(np.abs(Pnorm - 5.0))
    aIBi = -2

    tau = 100
    tsVals = tVals[tVals < tau]
    qds_aIBi_ts = qds_aIBi.sel(t=tsVals)

    DynOv_IRVals = np.zeros(IRrat_Vals.size)
    for ind, IRrat_val in enumerate(IRrat_Vals):
        IRdatapath = qdatapath_Dict[IRrat_val]
        qds_aIBi_ts = xr.open_dataset(IRdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi)).sel(t=tsVals)
        St = np.abs(qds_aIBi_ts.isel(P=Pind)['Real_DynOv'].values + 1j * qds_aIBi_ts.isel(P=Pind)['Imag_DynOv'].values).real.astype(float)
        DynOv_IRVals[ind] = St[-1]

    qds_aIBi_orig = xr.open_dataset(qdatapath_Dict[1.0] + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi)).sel(t=tsVals)
    DynOv_orig = np.abs(qds_aIBi_orig.isel(P=Pind)['Real_DynOv'].values + 1j * qds_aIBi_orig.isel(P=Pind)['Imag_DynOv'].values).real.astype(float)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(tsVals / tscale, DynOv_orig, 'k-')
    axes[0].set_title('Loschmidt Echo (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c_{BEC}}=$' + '{:.1f})'.format(Pnorm[Pind]) + ', Original IR cutoff')
    axes[0].set_ylabel(r'$|S(t)|$')
    axes[0].set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlim([1e-1, 1e2])

    axes[1].plot(IRrat_Vals, 100 * np.abs(DynOv_IRVals - DynOv_IRVals[0]) / DynOv_IRVals[0], 'g-')
    axes[1].set_xlabel('IR Cutoff Increase Ratio')
    axes[1].set_ylabel('Percentage Difference in ' + r'$|S(t_{f})|$')
    axes[1].set_title('Percentage Difference in Loschmidt Echo Final Value (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c_{BEC}}=$' + '{:.1f})'.format(Pnorm[Pind]))
    axes[1].set_xscale('log')
    # axes[1].set_ylim([0, 1])

    # fig.tight_layout()
    plt.show()
