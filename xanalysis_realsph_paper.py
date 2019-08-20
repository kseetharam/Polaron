import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as colors
import os
import itertools
import pf_dynamic_cart as pfc
import pf_dynamic_sph as pfs
import pf_static_sph as pss
import Grid
import warnings
from scipy import interpolate
from scipy.optimize import curve_fit, OptimizeWarning, fsolve
from scipy.integrate import simps
import scipy.stats as ss
from timeit import default_timer as timer
from copy import copy


if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    mpegWriter = animation.writers['ffmpeg'](fps=2, bitrate=1800)
    # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    # Writer = animation.writers['ffmpeg']
    # mpegWriter = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

    higherCutoff = False; cutoffRat = 1.0
    betterResolution = False; resRat = 1.0

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (60, 60, 60)
    (dx, dy, dz) = (0.25, 0.25, 0.25)
    higherCutoff = False; cutoffRat = 1.5
    betterResolution = True; resRat = 0.5

    # (Lx, Ly, Lz) = (40, 40, 40)
    # (dx, dy, dz) = (0.25, 0.25, 0.25)

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.375, 0.375, 0.375)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    # NGridPoints_cart = 1.37e5

    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)
    linDimMajor = 0.99 * (k_max * np.sqrt(2) / 2)
    linDimMinor = linDimMajor

    massRat = 1
    IRrat = 1

    # git test

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'real', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon', 'noCSAmp': True}

    # ---- SET OUTPUT DATA FOLDER ----

    datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    animpath = '/media/kis/Storage/Dropbox/VariationalResearch/DataAnalysis/figs'
    if higherCutoff is True:
        datapath = datapath + '_cutoffRat_{:.2f}'.format(cutoffRat)
    if betterResolution is True:
        datapath = datapath + '_resRat_{:.2f}'.format(resRat)
    datapath = datapath + '/massRatio={:.1f}'.format(massRat)
    distdatapath = copy(datapath)

    if toggleDict['noCSAmp'] is True:
        datapath = datapath + '_noCSAmp'

    innerdatapath = datapath + '/redyn_spherical'
    distdatapath = distdatapath + '/redyn_spherical'

    if toggleDict['Coupling'] == 'frohlich':
        innerdatapath = innerdatapath + '_froh_new'
        distdatapath = distdatapath + '_froh'
        animpath = animpath + '/rdyn_frohlich'
    else:
        animpath = animpath + '/rdyn_twophonon'

    # IRrat_Vals = np.array([2, 5, 10, 100, 4e3])
    # qdatapath_Dict = {}
    # if toggleDict['Coupling'] == 'frohlich':
    #     qdatapath_Dict[1.0] = innerdatapath
    # elif toggleDict['Coupling'] == 'twophonon':
    #     qdatapath_Dict[1.0] = innerdatapath + '_IRcuts' + '/IRratio_{:.1E}'.format(1)

    # for IRrat_val in IRrat_Vals:
    #     if toggleDict['Coupling'] == 'twophonon':
    #         qdatapath_Dict[IRrat_val] = innerdatapath + '_IRcuts' + '/IRratio_{:.1E}'.format(IRrat_val)
    #     elif toggleDict['Coupling'] == 'frohlich':
    #         qdatapath_Dict[IRrat_val] = innerdatapath[0:-4] + '_IRcuts' + '/IRratio_{:.1E}'.format(IRrat_val)

    # qdatapath = qdatapath_Dict[IRrat]

    # # # Concatenate Individual Datasets (aIBi specific)

    # print(innerdatapath)

    # aIBi_List = [-10.0, -5.0, -2.0, -1.25, -1.0]
    # # aIBi_List = [-10.0, -5.0, -2.0]
    # for aIBi in aIBi_List:
    #     ds_list = []; P_list = []; mI_list = []
    #     for ind, filename in enumerate(os.listdir(innerdatapath)):
    #         if filename[0:14] == 'quench_Dataset':
    #             continue
    #         if filename[0:6] == 'interp':
    #             continue
    #         if filename[0:2] == 'mm':
    #             continue
    #         # if float(filename[-8:-3]) != aIBi and float(filename[-9:-3]) != aIBi:
    #         #     continue
    #         if filename[3] == '.':
    #             a = 0
    #         elif filename[4] == '.':
    #             a = 1
    #         if float(filename[13 + a:-3]) != aIBi:
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
    # IRrootpath = innerdatapath + '_IRcuts'
    # for IRrat in IRrat_Vals:
    #     IRdatapath = IRrootpath + '/IRratio_{:.1E}'.format(IRrat)
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

    # # # Remove Data From Data sets

    # outpath = datapath + '_noCSAmp/redyn_spherical'

    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     if filename[0:3] == 'amp':
    #         continue
    #     ds = xr.open_dataset(innerdatapath + '/' + filename)
    #     dropds = ds.drop(['Real_CSAmp', 'Imag_CSAmp'])
    #     dropds.to_netcdf(outpath + '/' + filename)
    # print('Drop Done')

    # # Analysis of Total Dataset

    aIBi = -1

    qds = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
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

    kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds.coords['k'].values); kgrid.initArray_premade('th', qds.coords['th'].values)
    kVals = kgrid.getArray('k')
    wk_Vals = pfs.omegak(kVals, mB, n0, gBB)
    bdiff = 100 * np.abs(wk_Vals - nu * kVals) / (nu * kVals)
    kind = np.abs(bdiff - 1).argmin().astype(int)
    klin = kVals[kind]
    tlin = 2 * np.pi / (nu * kVals[kind])
    tlin_norm = tlin / tscale
    print(klin, tlin_norm)

    print(kVals[-1], kVals[1] - kVals[0])
    print(qds.attrs['k_mag_cutoff'] * xi)

    # # # # Nph CURVES

    # tau = 100
    # tsVals = tVals[tVals < tau]
    # qds_aIBi_ts = qds_aIBi.sel(t=tsVals)

    # # print(Pnorm)

    # Pnorm_des = np.array([0.1, 0.5, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.4, 1.6, 2.5, 3.0, 5.0])

    # Pinds = np.zeros(Pnorm_des.size, dtype=int)
    # for Pn_ind, Pn in enumerate(Pnorm_des):
    #     Pinds[Pn_ind] = np.abs(Pnorm - Pn).argmin().astype(int)

    # fig, ax = plt.subplots()
    # for indP in Pinds:
    #     P = PVals[indP]
    #     Nph = qds_aIBi_ts.isel(P=indP)['Nph'].values
    #     ax.plot(tsVals / tscale, Nph, label='{:.2f}'.format(P / mc))

    # ax.legend(title=r'$\frac{P}{m_{I}c_{BEC}}$', loc=2, ncol=2)
    # ax.set_xscale('log')
    # ax.set_title('Total Phonon Number (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # ax.set_ylabel(r'$N_{ph}$')
    # ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')

    # plt.show()

    # # # # S(t) AND P_Imp CURVES

    # tailFit = True
    # logScale = True

    # tau = 100; tfCutoff = 90
    # tau = 300; tfCutoff = 290
    # # tau = 5
    # tsVals = tVals[tVals < tau]
    # qds_aIBi_ts = qds_aIBi.sel(t=tsVals)

    # # print(Pnorm)

    # # Pnorm_des = np.array([0.1, 0.8, 5.0, 10.0])

    # # Pnorm_des = np.array([0.1, 0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.4, 1.6, 2.5, 3.0, 5.0, 6.0, 7.0, 9.0])
    # # Pnorm_des = np.array([0.1, 0.5, 0.9, 1.4, 3.0, 5.0, 6.0, 7.0])
    # Pnorm_des = np.array([5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.0])

    # # Pnorm_des = np.array([0.1, 0.5, 0.8, 1.3, 1.6, 2.3, 3.0])
    # # Pnorm_des = np.array([0.1, 0.5, 1.0, 1.3, 1.5, 2.1, 2.5, 3.0, 4.0, 5.0])

    # Pinds = np.zeros(Pnorm_des.size, dtype=int)
    # for Pn_ind, Pn in enumerate(Pnorm_des):
    #     Pinds[Pn_ind] = np.abs(Pnorm - Pn).argmin().astype(int)

    # fig, axes = plt.subplots(nrows=2, ncols=1)
    # for indP in Pinds:
    #     P = PVals[indP]
    #     DynOv = np.abs(qds_aIBi_ts.isel(P=indP)['Real_DynOv'].values + 1j * qds_aIBi_ts.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #     PImp = P - qds_aIBi_ts.isel(P=indP)['Pph'].values

    #     if tailFit is True:
    #         tfmask = tsVals > tfCutoff
    #         tfVals = tsVals[tfmask]
    #         tfLin = tsVals[tsVals > 10]
    #         zD = np.polyfit(np.log(tfVals), np.log(DynOv[tfmask]), deg=1)
    #         fLinD = np.exp(zD[1]) * tfLin**(zD[0])
    #         zP = np.polyfit(np.log(tfVals), np.log(PImp[tfmask]), deg=1)
    #         fLinP = np.exp(zP[1]) * tfLin**(zP[0])
    #         axes[0].plot(tfLin / tscale, fLinD, 'k--', label='')

    #     axes[0].plot(tsVals / tscale, DynOv, label='{:.2f}'.format(P / mc))
    #     axes[1].plot(tsVals / tscale, PImp / (mI * nu), label='{:.2f}'.format(P / mc))
    #     # axes[1].plot(tfLin / tscale, fLinP, 'k--', label='')

    # axes[0].legend(title=r'$\frac{P}{m_{I}c_{BEC}}$', loc=3, ncol=2)
    # axes[0].set_title('Loschmidt Echo (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # axes[0].set_ylabel(r'$|S(t)|$')
    # axes[0].set_xlabel(r'$t$ [$\frac{\xi}{c}$]')

    # # axes[1].plot(tlin_norm * np.ones(PImp.size), np.linspace(np.min(PImp), np.max(PImp), PImp.size), 'ko')
    # axes[1].plot(tsVals / tscale, np.ones(tsVals.size), 'k--', label='$c_{BEC}$')
    # axes[1].legend(title=r'$\frac{P}{m_{I}c_{BEC}}$', loc=1, ncol=2)
    # # axes[1].set_xscale('log')
    # # axes[1].set_yscale('log')
    # # axes[1].set_xlim([1e-1, 1e2])
    # axes[1].set_title('Average Impurity Speed (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # axes[1].set_ylabel(r'$\frac{<P_{I}>}{m_{I}c_{BEC}}$')
    # axes[1].set_xlabel(r'$t$ [$\frac{\xi}{c}$]')

    # if logScale is True:
    #     axes[0].plot(tlin_norm * np.ones(DynOv.size), np.linspace(np.min(DynOv), np.max(DynOv), DynOv.size), 'k-')
    #     axes[0].set_xscale('log')
    #     axes[0].set_yscale('log')
    #     # axes[0].set_xlim([1e-1, 1e2])
    #     axes[1].set_xlim([-1, tau / tscale])

    # fig.tight_layout()
    # plt.show()

    # # # # S(t) AND P_Imp CURVES MULTIGRID

    # massRat = 1.0
    # aIBi = -2.0

    # Pnorm_des = np.array([0.1, 0.8, 1.2, 2.3, 2.5, 3.0, 5.0])
    # Pinds = np.zeros(Pnorm_des.size, dtype=int)
    # for Pn_ind, Pn in enumerate(Pnorm_des):
    #     Pinds[Pn_ind] = np.abs(Pnorm - Pn).argmin().astype(int)

    # dp1 = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_1.11E+08/massRatio={:.1f}/redyn_spherical'.format(massRat)
    # dp2 = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_1.11E+08_cutoffRat_1.50/massRatio={:.1f}/redyn_spherical'.format(massRat)
    # dp3 = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_1.11E+08_resRat_0.50/massRatio={:.1f}/redyn_spherical'.format(massRat)

    # dpList = [dp1, dp2, dp3]
    # qdsList = []
    # for dp in dpList:
    #     qdsList.append(xr.open_dataset(dp + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi)))

    # lineList = ['solid', 'dotted', 'dashed']
    # fig, ax = plt.subplots()

    # alegend_elements = []
    # for ind_ds, ds in enumerate(qdsList):
    #     n0 = ds.attrs['n0']
    #     gBB = ds.attrs['gBB']
    #     mI = ds.attrs['mI']
    #     mB = ds.attrs['mB']
    #     nu = np.sqrt(n0 * gBB / mB)

    #     tVals = qds.coords['t'].values
    #     tau = 100
    #     tsVals = tVals[tVals < tau]
    #     qds_aIBi_ts = ds.sel(t=tsVals)

    #     for indP in Pinds:
    #         P = PVals[indP]
    #         DynOv = np.abs(qds_aIBi_ts.isel(P=indP)['Real_DynOv'].values + 1j * qds_aIBi_ts.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #         PImp = P - qds_aIBi_ts.isel(P=indP)['Pph'].values
    #         ax.plot(tsVals / tscale, PImp / (mI * nu), linestyle=lineList[ind_ds])
    #     kVec = ds.coords['k'].values
    #     kmax = kVec[-1]
    #     dk = kVec[1] - kVec[0]
    #     alegend_elements.append(Line2D([0], [0], color='magenta', linestyle=lineList[ind_ds], label=r'$\Lambda=$' + '{:.2f}, '.format(kmax) + r'$dk=$' + '{:.2E}'.format(dk)))

    # ax.plot(tsVals / tscale, np.ones(tsVals.size), 'k--')

    # # ax.set_xscale('log')
    # # ax.set_yscale('log')
    # # ax.set_xlim([1e-1, 1e2])
    # alegend = ax.legend(handles=alegend_elements, loc=1, title='Grid Parameters')
    # ax.set_title('Average Impurity Speed (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # ax.set_ylabel(r'$\frac{<P_{I}>}{m_{I}c}$')
    # ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    # plt.show()

    # # # # S(t) AND P_Imp EXPONENTS

    # seperate = False

    # aIBi_des = np.array([-10.0, -5.0, -2.0, -1.25, -1.0])  # Data for stronger interactions (-1.0, -0.75, -0.5) is too noisy to get fits
    # # aIBi_des = np.array([-10.0, -5.0, -2.0])  # Data for stronger interactions (-1.0, -0.75, -0.5) is too noisy to get fits
    # # Another note: The fit for P_{Imp} is also difficult for anything other than very weak interactions -> this is probably because of the diverging convergence time to mI*c due to arguments in Nielsen

    # # PVals = PVals[(PVals / mc) <= 3.0]
    # Pnorm = PVals / mc

    # def powerfunc(t, a, b):
    #     return b * t**(-1 * a)

    # # tmin = 90; tmax = 100
    # tmin = 290; tmax = 300

    # tfVals = tVals[(tVals <= tmax) * (tVals >= tmin)]
    # rollwin = 1

    # colorList = ['red', '#7e1e9c', 'green', 'orange', '#60460f', 'blue']
    # lineList = ['solid', 'dotted', 'dashed']

    # fig, ax = plt.subplots()
    # if seperate:
    #     fig1, ax1 = plt.subplots()
    # for inda, aIBi in enumerate(aIBi_des):
    #     qds_aIBi = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #     qds_aIBi_ts = qds_aIBi.sel(t=tfVals)
    #     PVals = qds_aIBi['P'].values; Pnorm = PVals / mc
    #     DynOv_Exponents = np.zeros(PVals.size)
    #     DynOv_Cov = np.full(PVals.size, np.nan)
    #     vImp_Exponents = np.zeros(PVals.size)
    #     vImp_Cov = np.full(PVals.size, np.nan)

    #     DynOv_Exponents_LR = np.zeros(PVals.size)
    #     vImp_Exponents_LR = np.zeros(PVals.size)

    #     DynOv_Rvalues = np.zeros(PVals.size)
    #     DynOv_Pvalues = np.zeros(PVals.size)
    #     DynOv_stderr = np.zeros(PVals.size)
    #     DynOv_tstat = np.zeros(PVals.size)
    #     DynOv_logAve = np.zeros(PVals.size)

    #     for indP, P in enumerate(PVals):
    #         DynOv_raw = np.abs(qds_aIBi_ts.isel(P=indP)['Real_DynOv'].values + 1j * qds_aIBi_ts.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #         DynOv_ds = xr.DataArray(DynOv_raw, coords=[tfVals], dims=['t'])

    #         DynOv_ds = DynOv_ds.rolling(t=rollwin, center=True).mean().dropna('t')
    #         Pph_ds = qds_aIBi_ts.isel(P=indP)['Pph'].rolling(t=rollwin, center=True).mean().dropna('t')

    #         DynOv_Vals = DynOv_ds.values
    #         tDynOv_Vals = DynOv_ds['t'].values

    #         # vImpc_Vals = (P - Pph_ds.values) / mc - 1
    #         vImpc_Vals = (P - Pph_ds.values) / mI - nu
    #         tvImpc_Vals = Pph_ds['t'].values

    #         # with warnings.catch_warnings():
    #         #     warnings.simplefilter("error", OptimizeWarning)
    #         #     try:
    #         #         Sopt, Scov = curve_fit(powerfunc, tDynOv_Vals, DynOv_Vals)
    #         #         DynOv_Exponents[indP] = Sopt[0]
    #         #         # DynOv_Cov[indP] = Scov[0]
    #         #         if Sopt[0] < 0:
    #         #             DynOv_Exponents[indP] = 0
    #         #     except OptimizeWarning:
    #         #         DynOv_Exponents[indP] = 0
    #         #     except RuntimeError:
    #         #         DynOv_Exponents[indP] = 0

    #         # with warnings.catch_warnings():
    #         #     warnings.simplefilter("error", OptimizeWarning)
    #         #     try:
    #         #         vIopt, vIcov = curve_fit(powerfunc, tvImpc_Vals, vImpc_Vals)
    #         #         vImp_Exponents[indP] = vIopt[0]
    #         #         # vImp_Cov[indP] = vIcov[0]
    #         #         if vIopt[0] < 0:
    #         #             vImp_Exponents[indP] = 0
    #         #         if vImpc_Vals[-1] < 0:
    #         #             vImp_Exponents[indP] = 0
    #         #     except OptimizeWarning:
    #         #         vImp_Exponents[indP] = 0
    #         #     except RuntimeError:
    #         #         vImp_Exponents[indP] = 0

    #         S_slope, S_intercept, S_rvalue, S_pvalue, S_stderr = ss.linregress(np.log(tDynOv_Vals), np.log(DynOv_Vals))
    #         DynOv_Exponents_LR[indP] = -1 * S_slope
    #         DynOv_Rvalues[indP] = S_rvalue
    #         DynOv_Pvalues[indP] = S_pvalue
    #         DynOv_stderr[indP] = S_stderr
    #         DynOv_tstat[indP] = S_slope / S_stderr
    #         # print(S_rvalue, S_pvalue, S_stderr, S_slope / S_stderr)

    #         DynOv_logAve[indP] = np.average(np.log(DynOv_Vals))

    #         # if (-1 * S_slope) < 0:
    #         #     DynOv_Exponents_LR[indP] = 0

    #         if vImpc_Vals[-1] < 0:
    #             vImp_Exponents_LR[indP] = 0
    #         else:
    #             vI_slope, vI_intercept, vI_rvalue, vI_pvalue, vI_stderr = ss.linregress(np.log(tvImpc_Vals), np.log(vImpc_Vals))
    #             vImp_Exponents_LR[indP] = -1 * vI_slope
    #             if (-1 * vI_slope) < 0:
    #                 vImp_Exponents_LR[indP] = 0

    #     print(aIBi)
    #     print(DynOv_Exponents_LR)
    #     print('\n')
    #     print(DynOv_Pvalues)
    #     print('\n')
    #     print(DynOv_Rvalues**2)
    #     print('\n')
    #     print(DynOv_stderr)
    #     print('\n')
    #     print(DynOv_tstat)
    #     print('\n')
    #     print(DynOv_stderr / DynOv_logAve)

    #     # badFitmask = np.abs(DynOv_stderr / DynOv_logAve) > 1e-3
    #     # DynOv_Exponents_LR[badFitmask] = 0

    #     if seperate:
    #         ax.plot(Pnorm, DynOv_Exponents_LR, color=colorList[inda], linestyle='solid', marker='x', label='{:.1f}'.format(aIBi))
    #         ax1.plot(Pnorm, vImp_Exponents, color=colorList[inda], linestyle='dotted', marker='+', markerfacecolor='none', label='{:.1f}'.format(aIBi))
    #     else:
    #         # ax.plot(Pnorm, DynOv_Exponents, color=colorList[inda], linestyle='solid', marker='x', label='{:.1f}'.format(aIBi))
    #         # ax.plot(Pnorm, vImp_Exponents, color=colorList[inda], linestyle='dotted', marker='+', markerfacecolor='none', label='{:.1f}'.format(aIBi))

    #         ax.plot(Pnorm, DynOv_Exponents_LR, color=colorList[inda], linestyle='solid', marker='x', label='{:.1f}'.format(aIBi))
    #         ax.plot(Pnorm, vImp_Exponents_LR, color=colorList[inda], linestyle='dotted', marker='+', markerfacecolor='none', label='{:.1f}'.format(aIBi))

    # if seperate:
    #     ax.set_xlabel(r'$\frac{P}{m_{I}c_{BEC}}$')
    #     ax.set_ylabel(r'$\gamma$' + ' for ' + r'$|S(t)|\propto t^{-\gamma}$')
    #     ax.set_title('Long Time Power-Law Behavior of Loschmidt Echo')
    #     ax.legend(title=r'$a_{IB}^{-1}$', loc=2)

    #     ax1.set_xlabel(r'$\frac{P}{m_{I}c_{BEC}}$')
    #     ax1.set_ylabel(r'$\gamma$' + ' for ' + r'$|S(t)|\propto t^{-\gamma}$')
    #     ax1.set_title('Long Time Power-Law Behavior of Average Impurity Momentum')
    #     ax1.legend(title=r'$a_{IB}^{-1}$', loc=2)

    # else:
    #     # ax.set_xlabel(r'$\frac{P}{m_{I}c_{BEC}}$')
    #     ax.set_xlabel(r'$\frac{<v_{I}(t_{0})>}{c_{BEC}}$')
    #     ax.set_ylabel(r'$\gamma$' + ' for ' + r'$|S(t)|\propto t^{-\gamma}$')
    #     ax.set_title('Long Time Power-Law Behavior of Observables')
    #     alegend_elements = []
    #     mlegend_elements = []
    #     for inda, aIBi in enumerate(aIBi_des):
    #         alegend_elements.append(Line2D([0], [0], color=colorList[inda], linestyle='solid', label='{0}'.format(aIBi)))
    #     mlegend_elements.append(Line2D([0], [0], color='black', marker='x', label=r'$S(t)$'))
    #     mlegend_elements.append(Line2D([0], [0], color='black', marker='+', label=r'$<v_{I}(t)>$'))
    #     alegend = ax.legend(handles=alegend_elements, loc=(0.01, 0.7), title=r'$a_{IB}^{-1}$')
    #     plt.gca().add_artist(alegend)
    #     mlegend = ax.legend(handles=mlegend_elements, loc=(0.12, 0.75), title='Observable')
    #     plt.gca().add_artist(mlegend)

    # # ax.set_xlim([0, 7])

    # plt.show()

    # # # IR Cuts S(t) (SPHERICAL)

    # Pdes = 2.5
    # Pind = np.argmin(np.abs(Pnorm - Pdes))
    # aIBi = -10

    # tau = 100
    # tsVals = tVals[tVals < tau]
    # qds_aIBi_ts = qds_aIBi.sel(t=tsVals)

    # fig2, ax2 = plt.subplots()
    # DynOv_IRVals = np.zeros(IRrat_Vals.size)
    # for ind, IRrat_val in enumerate(IRrat_Vals):
    #     IRdatapath = qdatapath_Dict[IRrat_val]
    #     qds_aIBi_ts = xr.open_dataset(IRdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi)).sel(t=tsVals)
    #     St = np.abs(qds_aIBi_ts.isel(P=Pind)['Real_DynOv'].values + 1j * qds_aIBi_ts.isel(P=Pind)['Imag_DynOv'].values).real.astype(float)
    #     DynOv_IRVals[ind] = St[-1]
    #     ax2.plot(tsVals / tscale, St, label='{:.1E}'.format(IRrat_val))

    # qds_aIBi_orig = xr.open_dataset(qdatapath_Dict[1.0] + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi)).sel(t=tsVals)
    # DynOv_orig = np.abs(qds_aIBi_orig.isel(P=Pind)['Real_DynOv'].values + 1j * qds_aIBi_orig.isel(P=Pind)['Imag_DynOv'].values).real.astype(float)

    # ax2.plot(tsVals / tscale, DynOv_orig, label='Original')
    # ax2.set_title('Loschmidt Echo (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c_{BEC}}=$' + '{:.1f})'.format(Pnorm[Pind]))
    # ax2.set_ylabel(r'$|S(t)|$')
    # ax2.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.set_xlim([1e-1, 1e2])
    # ax2.legend(title='IR Cutoff Ratio')

    # fig, axes = plt.subplots(nrows=1, ncols=2)
    # axes[0].plot(tsVals / tscale, DynOv_orig, 'k-')
    # axes[0].set_title('Loschmidt Echo (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c_{BEC}}=$' + '{:.1f})'.format(Pnorm[Pind]) + ', Original IR cutoff')
    # axes[0].set_ylabel(r'$|S(t)|$')
    # axes[0].set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    # axes[0].set_xscale('log')
    # axes[0].set_yscale('log')
    # axes[0].set_xlim([1e-1, 1e2])

    # axes[1].plot(IRrat_Vals, 100 * np.abs(DynOv_IRVals - DynOv_IRVals[0]) / DynOv_IRVals[0], 'g-')
    # axes[1].set_xlabel('IR Cutoff Increase Ratio')
    # axes[1].set_ylabel('Percentage Difference in ' + r'$|S(t_{f})|$')
    # axes[1].set_title('Percentage Difference in Loschmidt Echo Final Value (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c_{BEC}}=$' + '{:.1f})'.format(Pnorm[Pind]))
    # axes[1].set_xscale('log')
    # # axes[1].set_ylim([0, 1])

    # # fig.tight_layout()
    # plt.show()

    # # # # BOGOLIUBOV DISPERSION (SPHERICAL)

    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds.coords['k'].values); kgrid.initArray_premade('th', qds.coords['th'].values)
    # kVals = kgrid.getArray('k')
    # wk_Vals = pfs.omegak(kVals, mB, n0, gBB)
    # bdiff = 100 * np.abs(wk_Vals - nu * kVals) / kVals
    # kind = np.abs(bdiff - 1).argmin().astype(int)
    # tlin = 2 * np.pi / (nu * kVals[kind])
    # print(kVals[kind], tlin / tscale)

    # fig, ax = plt.subplots()
    # # ax.plot(kVals, bdiff)
    # ax.plot(kVals, wk_Vals, 'k-', label='')
    # ax.plot(kVals, nu * kVals, 'b--', label=r'$c_{BEC}|k|$')
    # ax.plot(kVals, kVals**2 / (2 * mB), 'r--', label=r'$\frac{k^{2}}{2m_{B}}$')
    # ax.set_title('Bogoliubov Phonon Dispersion')
    # ax.set_xlabel(r'$|k|$')
    # ax.set_ylabel(r'$\omega_{|k|}$')
    # ax.set_xlim([0, 2])
    # ax.set_ylim([0, 3])
    # ax.legend(loc=2, fontsize='x-large')
    # plt.show()

    # # # # # IMPURITY VELOCITY RATIO CURVES

    # def powerfunc(t, a, b):
    #     return b * t**(-1 * a)

    # tmin = 70
    # tmax = 100
    # tfVals = tVals[(tVals <= tmax) * (tVals >= tmin)]
    # rollwin = 1

    # colorList = ['red', '#7e1e9c', 'green', 'orange', 'blue']
    # lineList = ['solid', 'dotted', 'dashed', '-.']
    # aIBi_des = np.array([-10.0, -5.0, -2.0, -1.5])
    # # aIBi_des = np.array([aIBi_des[2]])
    # massRat_des = np.array([0.5, 0.75, 1.0, 2, 5.0])
    # mdatapaths = []

    # for mR in massRat_des:
    #     if toggleDict['Old'] is True:
    #         mdatapaths.append(datapath[0:-7] + '{:.1f}_old'.format(mR))
    #     else:
    #         mdatapaths.append(datapath[0:-3] + '{:.1f}'.format(mR))
    # if toggleDict['Dynamics'] != 'real' or toggleDict['Grid'] != 'spherical' or toggleDict['Coupling'] != 'twophonon':
    #     print('SETTING ERROR')

    # fig2, ax2 = plt.subplots()
    # for inda, aIBi in enumerate(aIBi_des):
    #     for indm, mRat in enumerate(massRat_des):
    #         mds = xr.open_dataset(mdatapaths[indm] + '/redyn_spherical/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #         Plen = mds.coords['P'].values.size
    #         Pstart_ind = 0
    #         PVals = mds.coords['P'].values[Pstart_ind:Plen]
    #         n0 = mds.attrs['n0']
    #         gBB = mds.attrs['gBB']
    #         mI = mds.attrs['mI']
    #         mB = mds.attrs['mB']
    #         nu = np.sqrt(n0 * gBB / mB)

    #         vI0_Vals = (PVals - mds.isel(t=0, P=np.arange(Pstart_ind, Plen))['Pph'].values) / mI

    #         mds_ts = mds.sel(t=tfVals)
    #         vImp_Exponents = np.zeros(PVals.size)
    #         vImp_Constants = np.zeros(PVals.size)

    #         for indP, P in enumerate(PVals):
    #             Pph_ds = mds_ts.isel(P=indP)['Pph'].rolling(t=rollwin, center=True).mean().dropna('t')
    #             vImpc_Vals = (P - Pph_ds.values) / mI - nu
    #             tvImpc_Vals = Pph_ds['t'].values

    #             with warnings.catch_warnings():
    #                 warnings.simplefilter("error", OptimizeWarning)
    #                 try:
    #                     vIopt, vIcov = curve_fit(powerfunc, tvImpc_Vals, vImpc_Vals)
    #                     vImp_Exponents[indP] = vIopt[0]
    #                     vImp_Constants[indP] = vIopt[1]
    #                     if vIopt[0] < 0:
    #                         vImp_Exponents[indP] = 0
    #                     if vImpc_Vals[-1] < 0:
    #                         vImp_Exponents[indP] = 0
    #                         vImp_Constants[indP] = vImpc_Vals[-1]
    #                 except OptimizeWarning:
    #                     vImp_Exponents[indP] = 0
    #                     vImp_Constants[indP] = vImpc_Vals[-1]
    #                 except RuntimeError:
    #                     vImp_Exponents[indP] = 0
    #                     vImp_Constants[indP] = vImpc_Vals[-1]

    #         vIf_Vals = nu + powerfunc(1e1000, vImp_Exponents, vImp_Constants)
    #         # vIf_Vals = (PVals - mds['Pph'].isel(t=np.arange(-5, 0), P=np.arange(Pstart_ind, Plen)).mean(dim='t').values) / mI
    #         ax2.plot(vI0_Vals / nu, vIf_Vals / vI0_Vals, linestyle=lineList[inda], color=colorList[indm])

    # vI0_norm = vI0_Vals / nu; refMask = vI0_norm >= 1
    # ax2.plot(vI0_norm[refMask], nu / vI0_Vals[refMask], 'k-')

    # alegend_elements2 = []
    # mlegend_elements2 = []
    # for inda, aIBi in enumerate(aIBi_des):
    #     alegend_elements2.append(Line2D([0], [0], color='magenta', linestyle=lineList[inda], label='{0}'.format(aIBi)))
    # for indm, mR in enumerate(massRat_des):
    #     mlegend_elements2.append(Line2D([0], [0], color=colorList[indm], linestyle='solid', label='{0}'.format(mR)))

    # ax2.set_xlabel(r'$\frac{<v_{I}(t_{0})>}{c_{BEC}}$')
    # ax2.set_ylabel(r'$\frac{<v_{I}(t_{f})>}{<v_{I}(t_{0})>}$')
    # ax2.set_title('Average Impurity Speed')
    # alegend2 = ax2.legend(handles=alegend_elements2, loc=(0.45, 0.65), title=r'$a_{IB}^{-1}$')
    # plt.gca().add_artist(alegend2)
    # mlegend2 = ax2.legend(handles=mlegend_elements2, loc=(0.64, 0.70), ncol=2, title=r'$\frac{m_{I}}{m_{B}}$')
    # plt.gca().add_artist(mlegend2)
    # reflegend = ax2.legend(handles=[Line2D([0], [0], color='black', linestyle='solid', label=r'$<v_{I}(t_{f})>=c_{BEC}$')], loc=(0.65, 0.60))
    # plt.gca().add_artist(reflegend)

    # plt.show()

    # # # # IMPURITY FINAL VELOCITY CURVES

    # def powerfunc(t, a, b):
    #     return b * t**(-1 * a)

    # tmin = 90
    # tmax = 100
    # tfVals = tVals[(tVals <= tmax) * (tVals >= tmin)]
    # rollwin = 1

    # colorList = ['red', '#7e1e9c', 'green', 'orange', 'blue']
    # lineList = ['solid', 'dotted', 'dashed', '-.']
    # aIBi_des = np.array([-10.0, -5.0, -2.0, -1.25])
    # # aIBi_des = np.array([aIBi_des[2]])
    # massRat_des = np.array([0.5, 1.0, 2.0])
    # mdatapaths = []

    # for mR in massRat_des:
    #     if toggleDict['noCSAmp'] is True:
    #         mdatapaths.append(datapath[0:-11] + '{:.1f}_noCSAmp'.format(mR))
    #     else:
    #         mdatapaths.append(datapath[0:-3] + '{:.1f}_noCSAmp'.format(mR))
    # if toggleDict['Dynamics'] != 'real' or toggleDict['Grid'] != 'spherical' or toggleDict['Coupling'] != 'twophonon':
    #     print('SETTING ERROR')

    # fig1, ax1 = plt.subplots()
    # # fig2, ax2 = plt.subplots()
    # for inda, aIBi in enumerate(aIBi_des):
    #     for indm, mRat in enumerate(massRat_des):
    #         mds = xr.open_dataset(mdatapaths[indm] + '/redyn_spherical/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #         Plen = mds.coords['P'].values.size
    #         Pstart_ind = 0
    #         PVals = mds.coords['P'].values[Pstart_ind:Plen]
    #         n0 = mds.attrs['n0']
    #         gBB = mds.attrs['gBB']
    #         mI = mds.attrs['mI']
    #         mB = mds.attrs['mB']
    #         nu = np.sqrt(n0 * gBB / mB)

    #         vI0_Vals = (PVals - mds.isel(t=0, P=np.arange(Pstart_ind, Plen))['Pph'].values) / mI

    #         mds_ts = mds.sel(t=tfVals)
    #         vImp_Exponents = np.zeros(PVals.size)
    #         vImp_Constants = np.zeros(PVals.size)

    #         for indP, P in enumerate(PVals):
    #             Pph_ds = mds_ts.isel(P=indP)['Pph'].rolling(t=rollwin, center=True).mean().dropna('t')
    #             vImpc_Vals = (P - Pph_ds.values) / mI - nu
    #             tvImpc_Vals = Pph_ds['t'].values

    #             if vImpc_Vals[-1] < 0:
    #                 vImp_Exponents[indP] = 0
    #                 vImp_Constants[indP] = vImpc_Vals[-1]
    #             else:
    #                 vI_slope, vI_intercept, vI_rvalue, vI_pvalue, vI_stderr = ss.linregress(np.log(tvImpc_Vals), np.log(vImpc_Vals))
    #                 vImp_Exponents[indP] = -1 * vI_slope
    #                 vImp_Constants[indP] = np.exp(vI_intercept)

    #                 if (-1 * vI_slope) < 0:
    #                     vImp_Exponents[indP] = 0

    #             # with warnings.catch_warnings():
    #             #     warnings.simplefilter("error", OptimizeWarning)
    #             #     try:
    #             #         vIopt, vIcov = curve_fit(powerfunc, tvImpc_Vals, vImpc_Vals)
    #             #         vImp_Exponents[indP] = vIopt[0]
    #             #         vImp_Constants[indP] = vIopt[1]
    #             #         if vIopt[0] < 0:
    #             #             vImp_Exponents[indP] = 0
    #             #         if vImpc_Vals[-1] < 0:
    #             #             vImp_Exponents[indP] = 0
    #             #             vImp_Constants[indP] = vImpc_Vals[-1]
    #             #     except OptimizeWarning:
    #             #         vImp_Exponents[indP] = 0
    #             #         vImp_Constants[indP] = vImpc_Vals[-1]
    #             #     except RuntimeError:
    #             #         vImp_Exponents[indP] = 0
    #             #         vImp_Constants[indP] = vImpc_Vals[-1]

    #         if aIBi == -1.5 and mRat == 1.0:
    #             print(vImp_Exponents)
    #         vIf_Vals = nu + powerfunc(1e1000, vImp_Exponents, vImp_Constants)
    #         ax1.plot(vI0_Vals / nu, vIf_Vals / nu, linestyle=lineList[indm], color=colorList[inda])
    #         # ax2.plot(vI0_Vals / nu, vIf_Vals / vI0_Vals, linestyle=lineList[inda], color=colorList[indm])

    # ax1.plot(vI0_Vals / nu, np.ones(vI0_Vals.size), 'k-')
    # vI0_norm = vI0_Vals / nu; refMask = vI0_norm >= 1
    # # ax2.plot(vI0_norm[refMask], nu / vI0_Vals[refMask], 'k-')

    # alegend_elements = []
    # mlegend_elements = []
    # for inda, aIBi in enumerate(aIBi_des):
    #     alegend_elements.append(Line2D([0], [0], color=colorList[inda], linestyle='solid', label='{0}'.format(aIBi)))
    # for indm, mR in enumerate(massRat_des):
    #     mlegend_elements.append(Line2D([0], [0], color='magenta', linestyle=lineList[indm], label='{0}'.format(mR)))

    # ax1.set_xlabel(r'$\frac{<v_{I}(t_{0})>}{c_{BEC}}$')
    # ax1.set_ylabel(r'$\frac{<v_{I}(t_{\infty})>}{c_{BEC}}$')
    # ax1.set_title('Average Impurity Speed')
    # alegend = ax1.legend(handles=alegend_elements, loc=(0.45, 0.08), title=r'$a_{IB}^{-1}$')
    # plt.gca().add_artist(alegend)
    # mlegend = ax1.legend(handles=mlegend_elements, loc=(0.64, 0.15), ncol=2, title=r'$\frac{m_{I}}{m_{B}}$')
    # plt.gca().add_artist(mlegend)
    # reflegend = ax1.legend(handles=[Line2D([0], [0], color='black', linestyle='solid', label=r'$<v_{I}(t_{\infty})>=c_{BEC}$')], loc=(0.65, 0.05))
    # plt.gca().add_artist(reflegend)
    # ax1.set_ylim([0, 1.2])
    # ax1.set_xlim([0, 7])

    # # Pcrit_norm_gs = np.array([1.086, 1.146, 1.446])
    # # intersec_points = np.array([1.0, 0.925, 0.719])
    # # ax1.plot(Pcrit_norm_gs, intersec_points, 'gx', label='')

    # plt.show()

    # # # # IMPURITY FINAL LOSCHMIDT ECHO CURVES

    # def powerfunc(t, a, b):
    #     return b * t**(-1 * a)

    # # tmin = 90; tmax = 100
    # tmin = 250; tmax = 300
    # tfVals = tVals[(tVals <= tmax) * (tVals >= tmin)]
    # rollwin = 1

    # colorList = ['red', '#7e1e9c', 'green', 'orange', 'blue']
    # lineList = ['solid', 'dotted', 'dashed', '-.']
    # aIBi_des = np.array([-10.0, -5.0, -2.0, -1.25, -1.0])
    # # massRat_des = np.array([0.5, 1.0, 2])
    # massRat_des = np.array([1.0])

    # mdatapaths = []

    # for mR in massRat_des:
    #     if toggleDict['noCSAmp'] is True:
    #         mdatapaths.append(datapath[0:-11] + '{:.1f}_noCSAmp'.format(mR))
    #     else:
    #         mdatapaths.append(datapath[0:-3] + '{:.1f}_noCSAmp'.format(mR))
    # if toggleDict['Dynamics'] != 'real' or toggleDict['Grid'] != 'spherical' or toggleDict['Coupling'] != 'twophonon':
    #     print('SETTING ERROR')

    # fig1, ax1 = plt.subplots()
    # for inda, aIBi in enumerate(aIBi_des):
    #     for indm, mRat in enumerate(massRat_des):
    #         mds = xr.open_dataset(mdatapaths[indm] + '/redyn_spherical/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #         Plen = mds.coords['P'].values.size
    #         Pstart_ind = 0
    #         PVals = mds.coords['P'].values[Pstart_ind:Plen]
    #         n0 = mds.attrs['n0']
    #         gBB = mds.attrs['gBB']
    #         mI = mds.attrs['mI']
    #         mB = mds.attrs['mB']
    #         nu = np.sqrt(n0 * gBB / mB)

    #         vI0_Vals = (PVals - mds.isel(t=0, P=np.arange(Pstart_ind, Plen))['Pph'].values) / mI

    #         mds_ts = mds.sel(t=tfVals)
    #         DynOv_Exponents = np.zeros(PVals.size)
    #         DynOv_Constants = np.zeros(PVals.size)

    #         for indP, P in enumerate(PVals):
    #             DynOv_raw = np.abs(mds_ts.isel(P=indP)['Real_DynOv'].values + 1j * mds_ts.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #             DynOv_ds = xr.DataArray(DynOv_raw, coords=[tfVals], dims=['t'])
    #             DynOv_ds = DynOv_ds.rolling(t=rollwin, center=True).mean().dropna('t')
    #             DynOv_Vals = DynOv_ds.values

    #             tDynOvc_Vals = DynOv_ds['t'].values

    #             S_slope, S_intercept, S_rvalue, S_pvalue, S_stderr = ss.linregress(np.log(tDynOvc_Vals), np.log(DynOv_Vals))
    #             DynOv_Exponents[indP] = -1 * S_slope
    #             DynOv_Constants[indP] = np.exp(S_intercept)

    #             if DynOv_Exponents[indP] < 0:
    #                 DynOv_Exponents[indP] = 0

    #             if (np.abs(DynOv_Exponents[indP]) < 0.001):
    #                 DynOv_Exponents[indP] = 0

    #             # if (np.abs(DynOv_Exponents[indP]) < 0.01) and (aIBi == -1.25):
    #             #     DynOv_Exponents[indP] = 0

    #             # if (np.abs(DynOv_Exponents[indP]) < 0.05) and (aIBi == -1.5):
    #             #     DynOv_Exponents[indP] = 0
    #             #     # DynOv_Constants[indP] = DynOv_Vals[-1]
    #             # if (DynOv_Exponents[indP] < 0) and (aIBi != -1.5):
    #             #     DynOv_Exponents[indP] = 0

    #             # with warnings.catch_warnings():
    #             #     warnings.simplefilter("error", OptimizeWarning)
    #             #     try:
    #             #         Sopt, Scov = curve_fit(powerfunc, tDynOvc_Vals, DynOv_Vals)
    #             #         DynOv_Exponents[indP] = Sopt[0]
    #             #         DynOv_Constants[indP] = Sopt[1]
    #             #         if Sopt[0] < 0:
    #             #             DynOv_Exponents[indP] = 0
    #             #     except OptimizeWarning:
    #             #         DynOv_Exponents[indP] = 0
    #             #         DynOv_Constants[indP] = DynOv_Vals[-1]
    #             #     except RuntimeError:
    #             #         DynOv_Exponents[indP] = 0
    #             #         DynOv_Constants[indP] = DynOv_Vals[-1]

    #         # if aIBi == -1.5 and mRat == 1.0:
    #         #     print(mdatapaths[indm] + '/redyn_spherical/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #         #     print(DynOv_Exponents)
    #         DynOvf_Vals = powerfunc(1e1000, DynOv_Exponents, DynOv_Constants)
    #         ax1.plot(vI0_Vals / nu, DynOvf_Vals, linestyle=lineList[indm], color=colorList[inda])

    # alegend_elements = []
    # mlegend_elements = []
    # for inda, aIBi in enumerate(aIBi_des):
    #     alegend_elements.append(Line2D([0], [0], color=colorList[inda], linestyle='solid', label='{0}'.format(aIBi)))
    # for indm, mR in enumerate(massRat_des):
    #     mlegend_elements.append(Line2D([0], [0], color='magenta', linestyle=lineList[indm], label='{0}'.format(mR)))

    # ax1.set_xlabel(r'$\frac{<v_{I}(t_{0})>}{c_{BEC}}$')
    # ax1.set_ylabel(r'$S(t_{\infty})$')
    # ax1.set_title('Loschmidt Echo')
    # alegend = ax1.legend(handles=alegend_elements, loc=(0.45, 0.65), title=r'$a_{IB}^{-1}$')
    # plt.gca().add_artist(alegend)
    # mlegend = ax1.legend(handles=mlegend_elements, loc=(0.64, 0.70), ncol=2, title=r'$\frac{m_{I}}{m_{B}}$')
    # plt.gca().add_artist(mlegend)
    # ax1.set_ylim([0, 1.2])
    # # ax1.set_xlim([0, np.max(vI0_Vals / nu)])
    # # ax1.set_xlim([0, 7])

    # plt.show()

    # # # INDIVIDUAL PHONON MOMENTUM DISTRIBUTION

    # Pnorm_des = np.array([0.1, 0.5, 0.8, 1.3, 1.5, 1.8, 3.0, 3.5, 4.0, 5.0, 8.0])
    # Pinds = np.zeros(Pnorm_des.size, dtype=int)
    # for Pn_ind, Pn in enumerate(Pnorm_des):
    #     Pinds[Pn_ind] = np.abs(Pnorm - Pn).argmin().astype(int)

    # print(PVals[Pinds])

    # indP = Pinds[5]
    # P = PVals[indP]
    # print(aIBi, P)

    # vmaxAuto = False
    # FGRBool = True; FGRlim = 1e-2
    # IRpatch = False
    # shortTime = False; tau = 5

    # # tau = 100
    # # tsVals = tVals[tVals < tau]
    # if Lx == 60:
    #     qds_PaIBi = xr.open_dataset(distdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    #     tsVals = qds_PaIBi.coords['tc'].values
    # else:
    #     # qds_PaIBi = qds_aIBi.sel(t=tsVals, P=P)
    #     qds_PaIBi = qds_aIBi.sel(P=P)
    #     tsVals = qds_PaIBi.coords['t'].values

    # if shortTime is True:
    #     tsVals = tsVals[tsVals <= tau]
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds_PaIBi.coords['k'].values); kgrid.initArray_premade('th', qds_PaIBi.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # dVk = kgrid.dV()

    # axislim = 1.2
    # if shortTime is True:
    #     axislim = 1.01 * P
    # # kIRcut = 0.13
    # # axislim = 3
    # kIRcut = 0.1
    # if Lx == 60:
    #     kIRcut = 0.01
    # if vmaxAuto is True:
    #     kIRcut = -1

    # kIRmask = kg < kIRcut
    # dVk_IR = dVk.reshape((len(kVec), len(thVec)))[kIRmask]
    # axmask = (kg >= kIRcut) * (kg <= axislim)
    # dVk_ax = dVk.reshape((len(kVec), len(thVec)))[axmask]

    # Omegak_da = xr.DataArray(np.full((tsVals.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tsVals, kVec, thVec], dims=['t', 'k', 'th'])
    # PhDen_da = xr.DataArray(np.full((tsVals.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tsVals, kVec, thVec], dims=['t', 'k', 'th'])
    # Nph_Vals = np.zeros(tsVals.size)
    # Pph_Vals = np.zeros(tsVals.size)
    # Pimp_Vals = np.zeros(tsVals.size)
    # norm_IRpercent = np.zeros(tsVals.size)
    # norm_axpercent = np.zeros(tsVals.size)
    # vmax = 0
    # for tind, t in enumerate(tsVals):
    #     if Lx == 60:
    #         CSAmp_ds = (qds_PaIBi['Real_CSAmp'] + 1j * qds_PaIBi['Imag_CSAmp']).sel(tc=t)
    #     else:
    #         CSAmp_ds = (qds_PaIBi['Real_CSAmp'] + 1j * qds_PaIBi['Imag_CSAmp']).sel(t=t)
    #     CSAmp_Vals = CSAmp_ds.values
    #     Nph_Vals[tind] = qds_PaIBi['Nph'].sel(t=t).values
    #     Pph_Vals[tind] = qds_PaIBi['Pph'].sel(t=t).values
    #     Pimp_Vals[tind] = P - Pph_Vals[tind]
    #     Bk_2D_vals = CSAmp_Vals.reshape((len(kVec), len(thVec)))
    #     PhDen_da.sel(t=t)[:] = ((1 / Nph_Vals[tind]) * np.abs(Bk_2D_vals)**2).real.astype(float)
    #     norm_tot = np.dot(PhDen_da.sel(t=t).values.flatten(), dVk)

    #     PhDen_IR = PhDen_da.sel(t=t).values[kIRmask]
    #     norm_IR = np.dot(PhDen_IR.flatten(), dVk_IR.flatten())
    #     norm_IRpercent[tind] = 100 * np.abs(norm_IR / norm_tot)
    #     # print(norm_IRpercent[tind])

    #     PhDen_ax = PhDen_da.sel(t=t).values[axmask]
    #     norm_ax = np.dot(PhDen_ax.flatten(), dVk_ax.flatten())
    #     norm_axpercent[tind] = 100 * np.abs(norm_ax / norm_tot)

    #     Omegak_da.sel(t=t)[:] = pfs.Omega(kgrid, Pimp_Vals[tind], mI, mB, n0, gBB).reshape((len(kVec), len(thVec))).real.astype(float)
    #     # print(Omegak_da.sel(t=t))

    #     maxval = np.max(PhDen_da.sel(t=t).values[np.logical_not(kIRmask)])
    #     if maxval > vmax:
    #         vmax = maxval

    # # Animations

    # fig1, ax1 = plt.subplots()

    # print(vmax)
    # vmin = 0

    # if (vmaxAuto is False) and (Lx != 60):
    #     vmax = 800
    # if shortTime is True:
    #     vmax = 200
    # interpmul = 5
    # if Lx == 60:
    #     PhDen0_interp_vals = PhDen_da.isel(t=0).values
    #     kxg_interp = kg * np.sin(thg)
    #     kzg_interp = kg * np.cos(thg)
    # else:
    #     PhDen0_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(PhDen_da.isel(t=0), 'k', 'th', interpmul)
    #     kxg_interp = kg_interp * np.sin(thg_interp)
    #     kzg_interp = kg_interp * np.cos(thg_interp)

    # if vmaxAuto is True:
    #     quad1 = ax1.pcolormesh(kzg_interp, kxg_interp, PhDen0_interp_vals[:-1, :-1], norm=colors.LogNorm(vmin=1e-3, vmax=vmax), cmap='inferno')
    #     quad1m = ax1.pcolormesh(kzg_interp, -1 * kxg_interp, PhDen0_interp_vals[:-1, :-1], norm=colors.LogNorm(vmin=1e-3, vmax=vmax), cmap='inferno')
    # else:
    #     quad1 = ax1.pcolormesh(kzg_interp, kxg_interp, PhDen0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax, cmap='inferno')
    #     quad1m = ax1.pcolormesh(kzg_interp, -1 * kxg_interp, PhDen0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax, cmap='inferno')

    # curve1 = ax1.plot(Pph_Vals[0], 0, marker='x', markersize=10, zorder=11, color="xkcd:steel grey")[0]
    # curve1m = ax1.plot(Pimp_Vals[0], 0, marker='o', markersize=10, zorder=11, color="xkcd:apple green")[0]
    # curve2 = ax1.plot(mc, 0, marker='*', markersize=10, zorder=11, color="cyan")[0]
    # patch_Excitation = plt.Circle((0, 0), 1e10, edgecolor='white', facecolor='None', linewidth=2)
    # ax1.add_patch(patch_Excitation)
    # # patch_klin = plt.Circle((0, 0), klin, edgecolor='#ff7f0e', facecolor='None')
    # patch_klin = plt.Circle((0, 0), klin, edgecolor='tab:cyan', facecolor='None')
    # ax1.add_patch(patch_klin)
    # t_text = ax1.text(0.81, 0.9, r'$t$ [$\frac{\xi}{c}$]: ' + '{:1.2f}'.format(tsVals[0] / tscale), transform=ax1.transAxes, fontsize='small', color='r')
    # Nph_text = ax1.text(0.81, 0.825, r'$N_{ph}$: ' + '{:.2f}'.format(Nph_Vals[0]), transform=ax1.transAxes, fontsize='small', color='xkcd:steel grey')

    # if IRpatch is True:
    #     patch_IR = plt.Circle((0, 0), kIRcut, edgecolor='#8c564b', facecolor='#8c564b')
    #     ax1.add_patch(patch_IR)
    #     IR_text = ax1.text(0.61, 0.75, r'Weight (IR patch): ' + '{:.2f}%'.format(norm_IRpercent[0]), transform=ax1.transAxes, fontsize='small', color='#8c564b')
    #     rem_text = ax1.text(0.61, 0.675, r'Weight (Rem vis): ' + '{:.2f}%'.format(norm_axpercent[0]), transform=ax1.transAxes, fontsize='small', color='yellow')

    # if FGRBool is True:
    #     if Lx == 60:
    #         Omegak0_interp_vals = Omegak_da.isel(t=0).values
    #     else:
    #         Omegak0_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(Omegak_da.isel(t=0), 'k', 'th', interpmul)
    #     FGRmask0 = np.abs(Omegak0_interp_vals) < FGRlim
    #     Omegak0_interp_vals[FGRmask0] = 1
    #     Omegak0_interp_vals[np.logical_not(FGRmask0)] = 0
    #     p = []
    #     p.append(ax1.contour(kzg_interp, kxg_interp, Omegak0_interp_vals, zorder=10, colors='tab:gray'))
    #     p.append(ax1.contour(kzg_interp, -1 * kxg_interp, Omegak0_interp_vals, zorder=10, colors='tab:gray'))
    #     p.append(ax1.contour(Pimp_Vals[0] - kzg_interp, -1 * kxg_interp, Omegak0_interp_vals, zorder=10, colors='xkcd:military green'))
    #     p.append(ax1.contour(Pimp_Vals[0] - kzg_interp, -1 * (-1) * kxg_interp, Omegak0_interp_vals, zorder=10, colors='xkcd:military green'))

    # ax1.set_xlim([-1 * axislim, axislim])
    # ax1.set_ylim([-1 * axislim, axislim])

    # patch_FGR_ph = Patch(facecolor='none', edgecolor='tab:gray')
    # patch_FGR_imp = Patch(facecolor='none', edgecolor='xkcd:military green')

    # if IRpatch is True:
    #     handles = (curve1, curve1m, curve2, patch_Excitation, patch_IR, patch_klin, patch_FGR_ph, patch_FGR_imp)
    #     labels = (r'$P_{ph}$', r'$P_{imp}$', r'$m_{I}c$', r'$\omega_{|k|}^{-1}(\frac{2\pi}{t})$', r'Singular Region', r'Linear Excitations', 'FGR Phase Space (ph)', 'FGR Phase Space (imp)')
    # else:
    #     handles = (curve1, curve1m, curve2, patch_Excitation, patch_klin, patch_FGR_ph, patch_FGR_imp)
    #     labels = (r'$P_{ph}$', r'$P_{imp}$', r'$m_{I}c$', r'$\omega_{|k|}^{-1}(\frac{2\pi}{t})$', r'Linear Excitations', 'FGR Phase Space (ph)', 'FGR Phase Space (imp)')

    # ax1.legend(handles, labels, loc=2, fontsize='small')
    # ax1.grid(True, linewidth=0.5)
    # ax1.set_title('Individual Phonon Distribution (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c}=$' + '{:.2f})'.format(Pnorm[indP]))
    # ax1.set_xlabel(r'$k_{z}$')
    # ax1.set_ylabel(r'$k_{x}$')
    # fig1.colorbar(quad1, ax=ax1, extend='both')

    # def animate1(i):
    #     if Lx == 60:
    #         PhDen_interp_vals = PhDen_da.isel(t=i).values
    #     else:
    #         PhDen_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(PhDen_da.isel(t=i), 'k', 'th', interpmul)
    #     quad1.set_array(PhDen_interp_vals[:-1, :-1].ravel())
    #     quad1m.set_array(PhDen_interp_vals[:-1, :-1].ravel())
    #     curve1.set_xdata(Pph_Vals[i])
    #     curve1m.set_xdata(Pimp_Vals[i])
    #     t_text.set_text(r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tsVals[i] / tscale))
    #     Nph_text.set_text(r'$N_{ph}$: ' + '{:.2f}'.format(Nph_Vals[i]))
    #     if IRpatch is True:
    #         IR_text.set_text(r'Weight (IR patch): ' + '{:.2f}%'.format(norm_IRpercent[i]))
    #         rem_text.set_text(r'Weight (Rem vis): ' + '{:.2f}%'.format(norm_axpercent[i]))

    #     def rfunc(k): return (pfs.omegak(k, mB, n0, gBB) - 2 * np.pi / tsVals[i])
    #     kroot = fsolve(rfunc, 1e8); kroot = kroot[kroot >= 0]
    #     patch_Excitation.set_radius(kroot[0])

    #     if FGRBool is True:
    #         if Lx == 60:
    #             Omegak_interp_vals = Omegak_da.isel(t=i).values
    #         else:
    #             Omegak_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(Omegak_da.isel(t=i), 'k', 'th', interpmul)
    #         FGRmask = np.abs(Omegak_interp_vals) < FGRlim
    #         Omegak_interp_vals[FGRmask] = 1
    #         Omegak_interp_vals[np.logical_not(FGRmask)] = 0

    #         for tp in p[0].collections:
    #             tp.remove()
    #         for tp in p[1].collections:
    #             tp.remove()
    #         for tp in p[2].collections:
    #             tp.remove()
    #         for tp in p[3].collections:
    #             tp.remove()
    #         p[0] = ax1.contour(kzg_interp, kxg_interp, Omegak_interp_vals, zorder=10, colors='tab:gray')
    #         p[1] = ax1.contour(kzg_interp, -1 * kxg_interp, Omegak_interp_vals, zorder=10, colors='tab:gray')
    #         p[2] = ax1.contour(Pimp_Vals[i] - kzg_interp, -1 * kxg_interp, Omegak_interp_vals, zorder=10, colors='xkcd:military green')
    #         p[3] = ax1.contour(Pimp_Vals[i] - kzg_interp, -1 * (-1) * kxg_interp, Omegak_interp_vals, zorder=10, colors='xkcd:military green')

    # if Lx == 60:
    #     intanim = 300
    # else:
    #     intanim = 1e-5
    # anim1 = animation.FuncAnimation(fig1, animate1, interval=intanim, frames=range(tsVals.size), blit=False)
    # anim1_filename = '/aIBi_{:.2f}_P_{:.2f}'.format(aIBi, P) + '_indPhononDist_2D_oscBox'
    # if vmaxAuto is True:
    #     anim1_filename = anim1_filename + '_vmaxLog'
    # if FGRBool is True:
    #     anim1_filename = anim1_filename + '_FGR'
    # if shortTime is True:
    #     anim1_filename = anim1_filename + '_shortTime'
    # # anim1.save(animpath + anim1_filename + '.mp4', writer=mpegWriter)
    # # anim1.save(animpath + anim1_filename + '.gif', writer='imagemagick')

    # plt.show()

    # # # INDIVIDUAL PHONON MOMENTUM DISTRIBUTION PLOT SLICES

    # Pnorm_des = np.array([0.1, 0.5, 0.8, 1.3, 1.5, 1.8, 3.0, 3.5, 4.0, 5.0, 8.0])
    # Pinds = np.zeros(Pnorm_des.size, dtype=int)
    # for Pn_ind, Pn in enumerate(Pnorm_des):
    #     Pinds[Pn_ind] = np.abs(Pnorm - Pn).argmin().astype(int)

    # print(PVals[Pinds])

    # indP = Pinds[5]
    # P = PVals[indP]
    # print(aIBi, P)

    # vmaxAuto = False
    # FGRBool = True; FGRlim = 1e-2
    # IRpatch = False
    # shortTime = False; tau = 5

    # # tau = 100
    # # tsVals = tVals[tVals < tau]
    # if Lx == 60:
    #     qds_PaIBi = xr.open_dataset(distdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    #     tsVals = qds_PaIBi.coords['tc'].values
    # else:
    #     # qds_PaIBi = qds_aIBi.sel(t=tsVals, P=P)
    #     qds_PaIBi = qds_aIBi.sel(P=P)
    #     tsVals = qds_PaIBi.coords['t'].values

    # if shortTime is True:
    #     tsVals = tsVals[tsVals <= tau]
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds_PaIBi.coords['k'].values); kgrid.initArray_premade('th', qds_PaIBi.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # dVk = kgrid.dV()

    # axislim = 1.2
    # if shortTime is True:
    #     axislim = 1.01 * P
    # # kIRcut = 0.13
    # # axislim = 3
    # kIRcut = 0.1
    # if Lx == 60:
    #     kIRcut = 0.01
    # if vmaxAuto is True:
    #     kIRcut = -1

    # kIRmask = kg < kIRcut
    # dVk_IR = dVk.reshape((len(kVec), len(thVec)))[kIRmask]
    # axmask = (kg >= kIRcut) * (kg <= axislim)
    # dVk_ax = dVk.reshape((len(kVec), len(thVec)))[axmask]

    # Omegak_da = xr.DataArray(np.full((tsVals.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tsVals, kVec, thVec], dims=['t', 'k', 'th'])
    # PhDen_da = xr.DataArray(np.full((tsVals.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tsVals, kVec, thVec], dims=['t', 'k', 'th'])
    # Nph_Vals = np.zeros(tsVals.size)
    # Pph_Vals = np.zeros(tsVals.size)
    # Pimp_Vals = np.zeros(tsVals.size)
    # norm_IRpercent = np.zeros(tsVals.size)
    # norm_axpercent = np.zeros(tsVals.size)
    # vmax = 0
    # for tind, t in enumerate(tsVals):
    #     if Lx == 60:
    #         CSAmp_ds = (qds_PaIBi['Real_CSAmp'] + 1j * qds_PaIBi['Imag_CSAmp']).sel(tc=t)
    #     else:
    #         CSAmp_ds = (qds_PaIBi['Real_CSAmp'] + 1j * qds_PaIBi['Imag_CSAmp']).sel(t=t)
    #     CSAmp_Vals = CSAmp_ds.values
    #     Nph_Vals[tind] = qds_PaIBi['Nph'].sel(t=t).values
    #     Pph_Vals[tind] = qds_PaIBi['Pph'].sel(t=t).values
    #     Pimp_Vals[tind] = P - Pph_Vals[tind]
    #     Bk_2D_vals = CSAmp_Vals.reshape((len(kVec), len(thVec)))
    #     PhDen_da.sel(t=t)[:] = ((1 / Nph_Vals[tind]) * np.abs(Bk_2D_vals)**2).real.astype(float)
    #     norm_tot = np.dot(PhDen_da.sel(t=t).values.flatten(), dVk)

    #     PhDen_IR = PhDen_da.sel(t=t).values[kIRmask]
    #     norm_IR = np.dot(PhDen_IR.flatten(), dVk_IR.flatten())
    #     norm_IRpercent[tind] = 100 * np.abs(norm_IR / norm_tot)
    #     # print(norm_IRpercent[tind])

    #     PhDen_ax = PhDen_da.sel(t=t).values[axmask]
    #     norm_ax = np.dot(PhDen_ax.flatten(), dVk_ax.flatten())
    #     norm_axpercent[tind] = 100 * np.abs(norm_ax / norm_tot)

    #     Omegak_da.sel(t=t)[:] = pfs.Omega(kgrid, Pimp_Vals[tind], mI, mB, n0, gBB).reshape((len(kVec), len(thVec))).real.astype(float)
    #     # print(Omegak_da.sel(t=t))

    #     maxval = np.max(PhDen_da.sel(t=t).values[np.logical_not(kIRmask)])
    #     if maxval > vmax:
    #         vmax = maxval

    # # Plot slices

    # tnorm = tsVals / tscale
    # tnVals_des = np.array([0.5, 8.0, 15.0, 25.0, 40.0, 75.0])
    # tninds = np.zeros(tnVals_des.size, dtype=int)
    # for tn_ind, tn in enumerate(tnVals_des):
    #     tninds[tn_ind] = np.abs(tnorm - tn).argmin().astype(int)
    # tslices = tsVals[tninds]

    # print(vmax)
    # vmin = 0

    # if (vmaxAuto is False) and (Lx != 60):
    #     vmax = 800
    # if shortTime is True:
    #     vmax = 200
    # interpmul = 5
    # if Lx == 60:
    #     PhDen0_interp_vals = PhDen_da.isel(t=0).values
    #     kxg_interp = kg * np.sin(thg)
    #     kzg_interp = kg * np.cos(thg)
    # else:
    #     PhDen0_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(PhDen_da.isel(t=0), 'k', 'th', interpmul)
    #     kxg_interp = kg_interp * np.sin(thg_interp)
    #     kzg_interp = kg_interp * np.cos(thg_interp)

    # vmax = 3000

    # fig, axes = plt.subplots(nrows=3, ncols=2)
    # for tind, t in enumerate(tslices):
    #     if tind == 0:
    #         ax = axes[0, 0]
    #     elif tind == 1:
    #         ax = axes[0, 1]
    #     if tind == 2:
    #         ax = axes[1, 0]
    #     if tind == 3:
    #         ax = axes[1, 1]
    #     if tind == 4:
    #         ax = axes[2, 0]
    #     if tind == 5:
    #         ax = axes[2, 1]

    #     PhDen_interp_vals = PhDen_da.sel(t=t).values
    #     if vmaxAuto is True:
    #         quad1 = ax.pcolormesh(kzg_interp, kxg_interp, PhDen_interp_vals[:-1, :-1], norm=colors.LogNorm(vmin=1e-3, vmax=vmax), cmap='inferno')
    #         quad1m = ax.pcolormesh(kzg_interp, -1 * kxg_interp, PhDen_interp_vals[:-1, :-1], norm=colors.LogNorm(vmin=1e-3, vmax=vmax), cmap='inferno')
    #     else:
    #         quad1 = ax.pcolormesh(kzg_interp, kxg_interp, PhDen_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax, cmap='inferno')
    #         quad1m = ax.pcolormesh(kzg_interp, -1 * kxg_interp, PhDen_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax, cmap='inferno')

    #     curve1 = ax.plot(Pph_Vals[tninds[tind]], 0, marker='x', markersize=10, zorder=11, color="xkcd:steel grey")[0]
    #     curve1m = ax.plot(Pimp_Vals[tninds[tind]], 0, marker='o', markersize=10, zorder=11, color="xkcd:apple green")[0]
    #     curve2 = ax.plot(mc, 0, marker='*', markersize=10, zorder=11, color="cyan")[0]

    #     def rfunc(k): return (pfs.omegak(k, mB, n0, gBB) - 2 * np.pi / tsVals[tninds[tind]])
    #     kroot = fsolve(rfunc, 1e8); kroot = kroot[kroot >= 0]
    #     patch_Excitation = plt.Circle((0, 0), kroot[0], edgecolor='red', facecolor='None', linewidth=2)
    #     ax.add_patch(patch_Excitation)
    #     patch_klin = plt.Circle((0, 0), klin, edgecolor='tab:cyan', facecolor='None')
    #     ax.add_patch(patch_klin)

    #     if IRpatch is True:
    #         patch_IR = plt.Circle((0, 0), kIRcut, edgecolor='#8c564b', facecolor='#8c564b')
    #         ax.add_patch(patch_IR)
    #         IR_text = ax.text(0.61, 0.75, r'Weight (IR patch): ' + '{:.2f}%'.format(norm_IRpercent[tninds[tind]]), transform=ax.transAxes, fontsize='small', color='#8c564b')
    #         rem_text = ax.text(0.61, 0.675, r'Weight (Rem vis): ' + '{:.2f}%'.format(norm_axpercent[tninds[tind]]), transform=ax.transAxes, fontsize='small', color='yellow')

    #     if FGRBool is True:
    #         if Lx == 60:
    #             Omegak_interp_vals = Omegak_da.sel(t=t).values
    #         else:
    #             Omegak_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(Omegak_da.sel(t=t), 'k', 'th', interpmul)
    #         FGRmask0 = np.abs(Omegak_interp_vals) < FGRlim
    #         Omegak_interp_vals[FGRmask0] = 1
    #         Omegak_interp_vals[np.logical_not(FGRmask0)] = 0
    #         p = []
    #         p.append(ax.contour(kzg_interp, kxg_interp, Omegak_interp_vals, zorder=10, colors='tab:gray'))
    #         p.append(ax.contour(kzg_interp, -1 * kxg_interp, Omegak_interp_vals, zorder=10, colors='tab:gray'))
    #         p.append(ax.contour(Pimp_Vals[tind] - kzg_interp, -1 * kxg_interp, Omegak_interp_vals, zorder=10, colors='xkcd:military green'))
    #         p.append(ax.contour(Pimp_Vals[tind] - kzg_interp, -1 * (-1) * kxg_interp, Omegak_interp_vals, zorder=10, colors='xkcd:military green'))

    #     ax.set_xlim([-1 * axislim, axislim])
    #     ax.set_ylim([-1 * axislim, axislim])
    #     ax.grid(True, linewidth=0.5)
    #     ax.set_title(r'$t$ [$\frac{\xi}{c}$]: ' + '{:1.2f}'.format(tsVals[tninds[tind]] / tscale))
    #     ax.set_xlabel(r'$k_{z}$')
    #     ax.set_ylabel(r'$k_{x}$')

    # patch_FGR_ph = Patch(facecolor='none', edgecolor='tab:gray')
    # patch_FGR_imp = Patch(facecolor='none', edgecolor='xkcd:military green')

    # if IRpatch is True:
    #     handles = (curve1, curve1m, curve2, patch_Excitation, patch_IR, patch_klin, patch_FGR_ph, patch_FGR_imp)
    #     labels = (r'$P_{ph}$', r'$P_{imp}$', r'$m_{I}c$', r'$\omega_{|k|}^{-1}(\frac{2\pi}{t})$', r'Singular Region', r'Linear Excitations', 'FGR Phase Space (ph)', 'FGR Phase Space (imp)')
    # else:
    #     handles = (curve1, curve1m, curve2, patch_Excitation, patch_klin, patch_FGR_ph, patch_FGR_imp)
    #     labels = (r'$P_{ph}$', r'$P_{imp}$', r'$m_{I}c$', r'$\omega_{|k|}^{-1}(\frac{2\pi}{t})$', r'Linear Excitations', 'FGR Phase Space (ph)', 'FGR Phase Space (imp)')

    # fig.subplots_adjust(right=0.8, bottom=0.13, hspace=0.5)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    # fig.colorbar(quad1, cax=cbar_ax, extend='both')
    # fig.legend(handles, labels, ncol=4, loc='lower center')
    # # st = fig.suptitle('Individual Phonon Distribution (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c}=$' + '{:.2f})'.format(Pnorm[indP]))

    # plt.show()

    # # # # SUBSONIC POLARON STATE OVERLAP

    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds_aIBi.coords['k'].values); kgrid.initArray_premade('th', qds_aIBi.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # dVk = kgrid.dV()

    # tau = 100
    # tsVals = tVals[tVals < tau]

    # # Static Interpolation

    # Nsteps = 1e2
    # pss.createSpline_grid(Nsteps, kgrid, mI, mB, n0, gBB)

    # aSi_tck = np.load('aSi_spline_sph.npy')
    # PBint_tck = np.load('PBint_spline_sph.npy')

    # print('Created Splines')

    # Pnorm_des = np.array([0.1, 0.2, 0.5, 0.8, 0.9])

    # Pinds = np.zeros(Pnorm_des.size, dtype=int)
    # for Pn_ind, Pn in enumerate(Pnorm_des):
    #     Pinds[Pn_ind] = np.abs(Pnorm - Pn).argmin().astype(int)

    # fig, ax = plt.subplots()
    # for ip, indP in enumerate(Pinds):
    #     P = PVals[indP]
    #     DP = pss.DP_interp(0, P, aIBi, aSi_tck, PBint_tck)
    #     aSi = pss.aSi_interp(DP, aSi_tck)
    #     Bk_static = pss.BetaK(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB)
    #     Phase_static = 0

    #     qds_PaIBi = qds_aIBi.sel(t=tsVals, P=P)
    #     overlapVals = np.zeros(tsVals.size)
    #     for tind, t in enumerate(tsVals):
    #         CSAmp_ds = (qds_PaIBi['Real_CSAmp'] + 1j * qds_PaIBi['Imag_CSAmp']).sel(t=t)
    #         CSAmp_Vals = CSAmp_ds.values
    #         CSAmp_Vals = CSAmp_Vals.reshape(CSAmp_Vals.size)
    #         Phase = qds_PaIBi['Phase'].sel(t=t).values
    #         exparg = np.dot(np.abs(CSAmp_Vals)**2 + np.abs(Bk_static)**2 - 2 * Bk_static.conjugate() * CSAmp_Vals, dVk)
    #         overlapVals[tind] = np.abs(np.exp(-1j * (Phase - Phase_static)) * np.exp((-1 / 2) * exparg))

    #     ax.plot(tsVals / tscale, overlapVals, label='{:.2f}'.format(P / mc))

    # ax.legend(title=r'$\frac{P}{m_{I}c_{BEC}}$', loc=1, ncol=2)
    # # ax.set_xscale('log')
    # # ax.set_yscale('log')
    # # ax.set_xlim([1e-1, 1e2])
    # ax.set_title('Polaron State Overlap (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # ax.set_ylabel(r'$|<\psi_{pol}|\psi(t)>|$')
    # ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    # plt.show()

    # # # # PARTICIPATION RATIO CURVES (VS TIME) - SPHERICAL

    # shortTime = True; tau = 5

    # Pnorm_des = np.array([0.1, 0.5, 1.0, 1.3, 1.5, 2.1, 2.5, 3.0, 4.0, 5.0])
    # # Pnorm_des = np.array([0.1, 0.5, 1.0, 3.0])

    # Pinds = np.zeros(Pnorm_des.size, dtype=int)
    # for Pn_ind, Pn in enumerate(Pnorm_des):
    #     Pinds[Pn_ind] = np.abs(Pnorm - Pn).argmin().astype(int)

    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds_aIBi.coords['k'].values); kgrid.initArray_premade('th', qds_aIBi.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # dVk = kgrid.dV()

    # fig, ax = plt.subplots()
    # for indP in Pinds:
    #     P = PVals[indP]

    #     if Lx == 60:
    #         qds_PaIBi = xr.open_dataset(distdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    #         tsVals = qds_PaIBi.coords['tc'].values

    #     else:
    #         qds_PaIBi = qds_aIBi.sel(P=P)
    #         tsVals = qds_PaIBi.coords['t'].values

    #     CSAmp_ds = (qds_PaIBi['Real_CSAmp'] + 1j * qds_PaIBi['Imag_CSAmp'])
    #     Nph_ds = qds_PaIBi['Nph']

    #     if Lx == 60:
    #         CSAmp_ds = CSAmp_ds.rename({'tc': 't'})

    #     if shortTime is True:
    #         tsVals = tsVals[tsVals <= tau]
    #         CSAmp_ds = CSAmp_ds.sel(t=tsVals)
    #         Nph_ds = Nph_ds.sel(t=tsVals)

    #     PR_Vals = np.zeros(tsVals.size)

    #     for indt, t in enumerate(tsVals):
    #         CSAmp_Vals = CSAmp_ds.sel(t=t).values
    #         Bk_2D_vals = CSAmp_Vals.reshape((len(kVec), len(thVec)))

    #         # PhDen_Vals = ((1 / Nph_ds.sel(t=t).values) * np.abs(Bk_2D_vals)**2).real.astype(float)
    #         # # norm_tot = np.dot(PhDen_Vals.flatten(), dVk); print(norm_tot)
    #         # PR_Vals[indt] = np.dot((PhDen_Vals**2).flatten(), dVk) * ((2 * np.pi)**(-3))

    #         PhDen_Vals = ((2 * np.pi)**(-3)) * ((1 / Nph_ds.sel(t=t).values) * np.abs(Bk_2D_vals)**2).real.astype(float)
    #         dVk_n = ((2 * np.pi)**(3)) * dVk
    #         # norm_tot = np.dot(PhDen_Vals.flatten(), dVk_n); print(norm_tot)
    #         PR_Vals[indt] = np.dot((PhDen_Vals**2).flatten(), dVk_n)

    #         # # PhDen_Vals_disc = ((1 / np.sum(np.abs(Bk_2D_vals)**2)) * np.abs(Bk_2D_vals)**2).real.astype(float)
    #         # PhDen_Vals_disc = ((2 * np.pi)**(-3)) * ((1 / Nph_ds.sel(t=t).values) * np.abs(Bk_2D_vals)**2).real.astype(float)
    #         # dk = kVec[1] - kVec[0]
    #         # norm_tot = 2 * np.pi * np.sum(PhDen_Vals_disc) / (dk**2); print(norm_tot)
    #         # PR_Vals[indt] = np.sum(PhDen_Vals_disc**2)

    #     ax.plot(tsVals / tscale, PR_Vals, label='{:.2f}'.format(P / mc))

    # ax.legend(title=r'$\frac{P}{m_{I}c_{BEC}}$', loc=2, ncol=2)
    # # ax.set_xscale('log')
    # ax.set_title('Participation Ratio (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # ax.set_ylabel(r'$PR = \sum_{\vec{k}} (\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2})^{2}$')
    # ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')

    # plt.show()

# # # # PARTICIPATION RATIO CURVES (VS INITIAL VELOCITY) - SPHERICAL APPROXIMATION TO CARTESIAN INTERPOLATION

# # NOTE: We need the massRatio_1.0_old folder (or technically any of the _old folders) and the constants determined at the beginning of the script for this to run

#     inversePlot = True

#     # PRtype = 'continuous'
#     PRtype = 'discrete'; discPR_norm = True

#     Vol_fac = False

#     tau = 2.3
#     # tau = 5

#     # NOTE: The following constants are grid dependent (both on original spherical grid and interpolated cartesian grid)
#     dVk_cart = 0.0001241449577749997  # = dkx*dky*dkz from cartesian interpolation
#     Npoints_xyz = 85184000
#     Vxyz = 1984476.915083265
#     contToDisc_factor = dVk_cart / ((2 * np.pi)**3)

#     colorList = ['red', '#7e1e9c', 'green', 'orange', 'blue', '#60460f']
#     lineList = ['solid', 'dotted', 'dashed', 'dashdot']
#     # aIBi_des = np.array([-10.0, -5.0, -2.0, -1.5])
#     aIBi_des = np.array([-10.0, -5.0, -2.0, -1.5, -1.25, -1.0])
#     # massRat_des = np.array([1.0])
#     massRat_des = np.array([0.5, 1.0, 2])
#     mdatapaths = []

#     for mR in massRat_des:
#         if toggleDict['Old'] is True:
#             mdatapaths.append(datapath[0:-7] + '{:.1f}'.format(mR))
#         else:
#             mdatapaths.append(datapath[0:-3] + '{:.1f}'.format(mR))

#     if toggleDict['Dynamics'] != 'real' or toggleDict['Grid'] != 'spherical' or toggleDict['Coupling'] != 'twophonon':
#         print('SETTING ERROR')

#     kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds_aIBi.coords['k'].values); kgrid.initArray_premade('th', qds_aIBi.coords['th'].values)
#     kVec = kgrid.getArray('k')
#     thVec = kgrid.getArray('th')
#     kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
#     dVk = kgrid.dV()
#     print(kVec[-1], kVec[1] - kVec[0])

#     PRcont_Averages = np.zeros(PVals.size)
#     PRdisc_Averages = np.zeros(PVals.size)

#     P_Vals_norm = np.concatenate((np.linspace(0.1, 0.8, 5, endpoint=False), np.linspace(0.8, 1.4, 10, endpoint=False), np.linspace(1.4, 3.0, 12, endpoint=False), np.linspace(3.0, 5.0, 10, endpoint=False), np.linspace(5.0, 9.0, 20)))

#     fig1, ax1 = plt.subplots()
#     for inda, aIBi in enumerate(aIBi_des):
#         for indm, mRat in enumerate(massRat_des):

#             vI0_Vals = np.zeros(PVals.size)
#             PR_Averages = np.zeros(PVals.size)
#             PVals = mRat * mB * nu * P_Vals_norm

#             for indP, P in enumerate(PVals):
#                 qds_PaIBi = xr.open_dataset(mdatapaths[indm] + '/redyn_spherical/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
#                 CSAmp_ds = (qds_PaIBi['Real_CSAmp'] + 1j * qds_PaIBi['Imag_CSAmp'])
#                 Nph_ds = qds_PaIBi['Nph']
#                 mI = qds_PaIBi.attrs['mI']

#                 if Lx == 60:
#                     CSAmp_ds = CSAmp_ds.rename({'tc': 't'})

#                 tsVals = CSAmp_ds.coords['t'].values
#                 tsVals = tsVals[tsVals <= tau]
#                 CSAmp_ds = CSAmp_ds.sel(t=tsVals)
#                 Nph_ds = Nph_ds.sel(t=tsVals)

#                 PR_Vals = np.zeros(tsVals.size)

#                 dt = tsVals[1] - tsVals[0]

#                 for indt, t in enumerate(tsVals):
#                     CSAmp_Vals = CSAmp_ds.sel(t=t).values
#                     Bk_2D_vals = CSAmp_Vals.reshape((len(kVec), len(thVec)))

#                     PhDen_Vals = ((2 * np.pi)**(-3)) * ((1 / Nph_ds.sel(t=t).values) * np.abs(Bk_2D_vals)**2).real.astype(float)
#                     dVk_n = ((2 * np.pi)**(3)) * dVk
#                     PR_Vals[indt] = (2 * np.pi)**3 * np.dot((PhDen_Vals**2).flatten(), dVk_n)

#                 vI0_Vals[indP] = (P - qds_PaIBi.isel(t=0)['Pph'].values) / mI
#                 PR_Vals_del = np.delete(PR_Vals, 0); PR_Averages[indP] = (1 / (tsVals[-1] - tsVals[1])) * simps(y=PR_Vals_del, dx=dt)

#             #     # Cartesian reconstruct
#             #     interpds_PaIBi = xr.open_dataset(mdatapaths[indm] + '/redyn_spherical/amp3D/interp_P_{:.3f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, linDimMajor, linDimMinor))
#             #     # kx = interpds_PaIBi.coords['kx'].values; ky = interpds_PaIBi.coords['ky'].values; kz = interpds_PaIBi.coords['kz'].values
#             #     # dkx = kx[1] - kx[0]; dky = ky[1] - ky[0]; dkz = kz[1] - kz[0]
#             #     # Vconst = False
#             #     tsValsC = interpds_PaIBi.coords['t'].values
#             #     tsValsC = tsValsC[tsValsC <= tau]
#             #     dt = tsValsC[1] - tsValsC[0]
#             #     # if Vconst is True:
#             #     #     Vxyz = interpds_PaIBi.attrs['Vxyz']
#             #     # else:
#             #     #     Vxyz = 1
#             #     # Npoints_xyz = interpds_PaIBi.attrs['Npoints3D']
#             #     # PRcont_Vals = Vxyz * (2 * np.pi)**3 * interpds_PaIBi['PR_bare_cont'].sel(t=tsValsC).values
#             #     PRdisc_Vals = interpds_PaIBi['PR_bare_discrete'].sel(t=tsValsC).values
#             #     # PRcont_Averages[indP] = (1 / (tsValsC[-1] - tsValsC[0])) * simps(y=PRcont_Vals, dx=dt)
#             #     PRdisc_Averages[indP] = (1 / (tsValsC[-1] - tsValsC[0])) * simps(y=PRdisc_Vals, dx=dt)
#             # ax1.plot(vI0_Vals / nu, 1 / (PRdisc_Averages * Npoints_xyz), linestyle=lineList[inda], color=colorList[indm + 1])
#             #     # END OF CARTESIAN

#             if Vol_fac is True:
#                 PR_Averages = Vxyz * PR_Averages
#             else:
#                 PR_Averages = PR_Averages

#             if PRtype == 'continuous':
#                 PR_Averages = PR_Averages
#             elif PRtype == 'discrete':
#                 PR_Averages = PR_Averages * contToDisc_factor
#                 if discPR_norm is True:
#                     PR_Averages = PR_Averages * Npoints_xyz

#             if inversePlot is True:
#                 ax1.plot(vI0_Vals / nu, 1 / PR_Averages, linestyle=lineList[indm], color=colorList[inda])
#             else:
#                 ax1.plot(vI0_Vals / nu, PR_Averages, linestyle=lineList[indm], color=colorList[inda])

#     alegend_elements = []
#     mlegend_elements = []
#     for inda, aIBi in enumerate(aIBi_des):
#         alegend_elements.append(Line2D([0], [0], color=colorList[inda], linestyle='solid', label='{0}'.format(aIBi)))
#     for indm, mR in enumerate(massRat_des):
#         mlegend_elements.append(Line2D([0], [0], color='magenta', linestyle=lineList[indm], label='{0}'.format(mR)))

#     ax1.set_xlabel(r'$\frac{<v_{I}(t_{0})>}{c_{BEC}}$')

#     if inversePlot is True:
#         ax1.set_title('Short-Time-Averaged Inverse Participation Ratio (' + r'$t\in[0, $' + '{:.2f}'.format(tau / tscale) + r'$\frac{\xi}{c}]$)')
#         if PRtype == 'continuous':
#             ax1.set_ylabel(r'Average $IPR$ with $IPR = ((2\pi)^{3} \int d^3\vec{k} (\frac{1}{(2\pi)^3}\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2})^{2})^{-1}$')
#         elif PRtype == 'discrete':
#             if discPR_norm is True:
#                 ax1.set_ylabel(r'Average $IPR$ (Normalized by $N_{tot}$ modes in system)')
#             else:
#                 ax1.set_ylabel(r'Average $IPR$')
#     else:
#         ax1.set_title('Time-Averaged Participation Ratio (' + r'$t\in[0, $' + '{:.2f}'.format(tau / tscale) + r'$\frac{\xi}{c}]$)')
#         ax1.set_ylabel(r'Average $PR$ with $PR = (2\pi)^{3} \int d^3\vec{k} (\frac{1}{(2\pi)^3}\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2})^{2}$')
#     alegend = ax1.legend(handles=alegend_elements, loc=(0.03, 0.5), title=r'$a_{IB}^{-1}$')
#     plt.gca().add_artist(alegend)
#     mlegend = ax1.legend(handles=mlegend_elements, loc=(0.22, 0.70), ncol=2, title=r'$\frac{m_{I}}{m_{B}}$')
#     plt.gca().add_artist(mlegend)
#     ax1.set_xlim([0, np.max(vI0_Vals / nu)])

#     plt.show()

    # # # # PARTICIPATION RATIO CURVES (VS TIME) - CARTESIAN AMP RECONSTRUCT

    # Pnorm_des = np.array([0.1, 0.5, 1.0, 1.3, 1.5, 2.1, 2.5, 3.0, 4.0, 5.0])
    # # Pnorm_des = np.array([0.079, 0.190]) / (mI * nu)

    # Pinds = np.zeros(Pnorm_des.size, dtype=int)
    # for Pn_ind, Pn in enumerate(Pnorm_des):
    #     Pinds[Pn_ind] = np.abs(Pnorm - Pn).argmin().astype(int)

    # fig1, ax1 = plt.subplots()
    # fig2, ax2 = plt.subplots()
    # for indP in Pinds:
    #     P = PVals[indP]

    #     interpds_PaIBi = xr.open_dataset(distdatapath + '/amp3D/interp_P_{:.3f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, linDimMajor, linDimMinor))

    #     tsVals = interpds_PaIBi.coords['t'].values
    #     PRcont_Vals = interpds_PaIBi['PR_bare_cont'].values
    #     PRdisc_Vals = interpds_PaIBi['PR_bare_discrete'].values

    #     ax1.plot(tsVals / tscale, 1 / PRcont_Vals, label='{:.2f}'.format(P / mc))
    #     ax2.plot(tsVals / tscale, 1 / PRdisc_Vals, label='{:.2f}'.format(P / mc))

    # ax1.legend(title=r'$\frac{P}{m_{I}c_{BEC}}$', loc=2, ncol=2)
    # ax1.set_title('Inverse Participation Ratio (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # ax1.set_ylabel(r'$IPR = ((2\pi)^{3} \int d^3\vec{k} (\frac{1}{(2\pi)^3}\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2})^{2})^{-1}$')
    # ax1.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')

    # ax2.legend(title=r'$\frac{P}{m_{I}c_{BEC}}$', loc=2, ncol=2)
    # ax2.set_title('Inverse Participation Ratio (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # ax2.set_ylabel(r'$IPR = ((2\pi)^{3} \int d^3\vec{k} (\frac{1}{(2\pi)^3}\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2})^{2})^{-1}$')
    # ax2.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')

    # plt.show()

    # # # # PARTICIPATION RATIO CURVES (VS INITIAL VELOCITY) - CARTESIAN AMP RECONSTRUCT

    # inversePlot = True
    # PRconst = True; Vconst = False
    # PRtype = 'discrete'
    # tau = 2.3

    # if PRconst is True:
    #     PRcont_const = (2 * np.pi)**3
    # else:
    #     PRcont_const = 1

    # colorList = ['red', '#7e1e9c', 'green', 'orange', 'blue', '#60460f']
    # lineList = ['solid', 'dotted', 'dashed', 'dashdot']
    # aIBi_des = np.array([-10.0, -5.0, -2.0, -1.5, -1.25, -1.0])
    # massRat_des = np.array([1.0])
    # # massRat_des = np.array([0.5, 0.75, 1.0, 2, 5.0])
    # mdatapaths = []

    # for mR in massRat_des:
    #     if toggleDict['noCSAmp'] is True:
    #         mdatapaths.append(datapath[0:-11] + '{:.1f}'.format(mR))
    #     else:
    #         mdatapaths.append(datapath[0:-3] + '{:.1f}'.format(mR))

    # fig1, ax1 = plt.subplots()
    # for inda, aIBi in enumerate(aIBi_des):
    #     for indm, mRat in enumerate(massRat_des):

    #         vI0_Vals = np.zeros(PVals.size)
    #         PRcont_Averages = np.zeros(PVals.size)
    #         PRdisc_Averages = np.zeros(PVals.size)

    #         for indP, P in enumerate(PVals):
    #             qds_PaIBi = xr.open_dataset(mdatapaths[indm] + '/redyn_spherical/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    #             interpds_PaIBi = xr.open_dataset(mdatapaths[indm] + '/redyn_spherical/amp3D/interp_P_{:.3f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, linDimMajor, linDimMinor))

    #             tsVals = interpds_PaIBi.coords['t'].values
    #             tsVals = tsVals[tsVals <= tau]
    #             dt = tsVals[1] - tsVals[0]
    #             if Vconst is True:
    #                 Vxyz = interpds_PaIBi.attrs['Vxyz']
    #             else:
    #                 Vxyz = 1
    #             Npoints_xyz = interpds_PaIBi.attrs['Npoints3D']
    #             PRcont_Vals = Vxyz * PRcont_const * interpds_PaIBi['PR_bare_cont'].sel(t=tsVals).values
    #             PRdisc_Vals = interpds_PaIBi['PR_bare_discrete'].sel(t=tsVals).values

    #             vI0_Vals[indP] = (P - qds_PaIBi.isel(t=0)['Pph'].values) / mI
    #             PRcont_Averages[indP] = (1 / (tsVals[-1] - tsVals[1])) * simps(y=PRcont_Vals, dx=dt)
    #             PRdisc_Averages[indP] = (1 / (tsVals[-1] - tsVals[1])) * simps(y=PRdisc_Vals, dx=dt)
    #             if PRtype == 'continuous':
    #                 PR_Averages = PRcont_Averages
    #             elif PRtype == 'discrete':
    #                 PR_Averages = PRdisc_Averages * Npoints_xyz
    #                 # print(Npoints_xyz)
    #             else:
    #                 print('PR ERROR')
    #         if inversePlot is True:
    #             ax1.plot(vI0_Vals / nu, 1 / PR_Averages, linestyle=lineList[indm], color=colorList[inda])
    #         else:
    #             ax1.plot(vI0_Vals / nu, PR_Averages, linestyle=lineList[indm], color=colorList[inda])

    # alegend_elements = []
    # mlegend_elements = []
    # for inda, aIBi in enumerate(aIBi_des):
    #     alegend_elements.append(Line2D([0], [0], color=colorList[inda], linestyle='solid', label='{0}'.format(aIBi)))
    # for indm, mR in enumerate(massRat_des):
    #     mlegend_elements.append(Line2D([0], [0], color='magenta', linestyle=lineList[indm], label='{0}'.format(mR)))

    # ax1.set_xlabel(r'$\frac{<v_{I}(t_{0})>}{c_{BEC}}$')

    # if inversePlot is True:
    #     ax1.set_title('Short-Time-Averaged Inverse Participation Ratio (' + r'$t\in[0, $' + '{:.2f}'.format(tau / tscale) + r'$\frac{\xi}{c}]$)')
    #     if PRtype == 'continuous':
    #         ax1.set_ylabel(r'Average $IPR$ with $IPR = ((2\pi)^{3} \int d^3\vec{k} (\frac{1}{(2\pi)^3}\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2})^{2})^{-1}$')
    #     elif PRtype == 'discrete':
    #         ax1.set_ylabel(r'Average $IPR$ (Normalized by $N_{tot}$ modes in system)')
    # else:
    #     ax1.set_title('Time-Averaged Participation Ratio (' + r'$t\in[0, $' + '{:.2f}'.format(tau / tscale) + r'$\frac{\xi}{c}]$)')
    #     ax1.set_ylabel(r'Average $PR$ with $PR = (2\pi)^{3} \int d^3\vec{k} (\frac{1}{(2\pi)^3}\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2})^{2}$')
    # alegend = ax1.legend(handles=alegend_elements, loc=(0.03, 0.55), title=r'$a_{IB}^{-1}$')
    # plt.gca().add_artist(alegend)
    # mlegend = ax1.legend(handles=mlegend_elements, loc=(0.22, 0.70), ncol=2, title=r'$\frac{m_{I}}{m_{B}}$')
    # plt.gca().add_artist(mlegend)
    # ax1.set_xlim([0, np.max(vI0_Vals / nu)])
    # # ax1.set_xlim([0, 7])

    # plt.show()
