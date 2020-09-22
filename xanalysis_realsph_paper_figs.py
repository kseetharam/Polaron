import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Ellipse, Circle
from matplotlib.legend_handler import HandlerPatch
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
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
from matplotlib.ticker import NullFormatter

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    mpegWriter = animation.writers['ffmpeg'](fps=2, bitrate=1800)
    # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    # Writer = animation.writers['ffmpeg']
    # mpegWriter = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    matplotlib.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman', 'text.usetex': True, 'mathtext.fontset': 'dejavuserif'})

    higherCutoff = False
    cutoffRat = 1.0
    betterResolution = False
    resRat = 1.0

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (60, 60, 60)
    (dx, dy, dz) = (0.25, 0.25, 0.25)
    higherCutoff = False
    cutoffRat = 1.5
    betterResolution = True
    resRat = 0.5

    # (Lx, Ly, Lz) = (40, 40, 40)
    # (dx, dy, dz) = (0.25, 0.25, 0.25)

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.375, 0.375, 0.375)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    # NGridPoints_cart = 1.37e5

    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)
    linDimMajor = 0.99 * (k_max * np.sqrt(2) / 2)
    linDimMinor = linDimMajor

    massRat = 1.0
    IRrat = 1

    # git test

    # Toggle parameters

    toggleDict = {'Dynamics': 'real', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon', 'noCSAmp': True}

    # ---- SET OUTPUT DATA FOLDER ----

    datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    animpath = '/Users/kis/Dropbox/VariationalResearch/DataAnalysis/figs'
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

    # figdatapath = '/Users/kis/Dropbox/Apps/Overleaf/Quantum Cherenkov Transition in Bose Polaron Systems/figures/figdump'
    figdatapath = '/Users/kis/Dropbox/Apps/Overleaf/Cherenkov Polaron Paper pt1/figures/figdump'

    # # Analysis of Total Dataset

    aIBi = -10

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

    print(aIBi * xi)
    print(mI / mB, IRrat)
    IR_lengthscale = 1 / (k0 / (2 * np.pi)) / xi
    UV_lengthscale = 1 / (kf / (2 * np.pi)) / xi

    print(k0, 1 / IR_lengthscale, IR_lengthscale)
    print(kf, 1 / UV_lengthscale, UV_lengthscale)

    # aIBi_Vals = np.array([-10.0, -5.0, -2.0, -1.0, -0.75, -0.5])
    aIBi_Vals = np.array([-10.0, -5.0, -2.0])

    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray_premade('k', qds.coords['k'].values)
    kgrid.initArray_premade('th', qds.coords['th'].values)
    kVals = kgrid.getArray('k')
    wk_Vals = pfs.omegak(kVals, mB, n0, gBB)
    bdiff = 100 * np.abs(wk_Vals - nu * kVals) / (nu * kVals)
    kind = np.abs(bdiff - 1).argmin().astype(int)
    klin = kVals[kind]
    tlin = 2 * np.pi / (nu * kVals[kind])
    tlin_norm = tlin / tscale
    print(klin, tlin_norm)
    print(90 / tscale, 100 / tscale)

    print(kVals[-1], kVals[1] - kVals[0])
    print(qds.attrs['k_mag_cutoff'] * xi)
    print('Np: {0}'.format(qds.coords['k'].values.size * qds.coords['th'].values.size))

    # # FIG 3 - S(t) CURVES - PRL

    colorList = ['r', 'g', 'b']

    matplotlib.rcParams.update({'font.size': 12})

    tailFit = True
    logScale = True
    PimpData_roll = False; PimpData_rollwin = 2
    longTime = True
    # tau = 100; tfCutoff = 90; tfstart = 10
    tau = 300; tfCutoff = 200; tfstart = 10

    aIBi_weak = -10.0
    print(aIBi_weak * xi)

    if longTime:
        innerdatapath_longtime = datapath + '_longtime/redyn_spherical'
        qds_w = xr.open_dataset(innerdatapath_longtime + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi_weak))
    else:
        qds_w = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi_weak))

    tVals = qds_w['t'].values
    tsVals = tVals[tVals < tau]

    qds_aIBi_ts_w = qds_w.sel(t=tsVals)

    Pnorm_des = np.array([0.5, 2.2])

    Pinds = np.zeros(Pnorm_des.size, dtype=int)
    for Pn_ind, Pn in enumerate(Pnorm_des):
        Pinds[Pn_ind] = np.abs(Pnorm - Pn).argmin().astype(int)

    fig, ax = plt.subplots()
    for ip, indP in enumerate(Pinds):
        P = PVals[indP]
        DynOv_w = np.abs(qds_aIBi_ts_w.isel(P=indP)['Real_DynOv'].values + 1j * qds_aIBi_ts_w.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
        Pph_ds_w = xr.DataArray(qds_aIBi_ts_w.isel(P=indP)['Pph'].values, coords=[tsVals], dims=['t'])
        if PimpData_roll:
            Pph_ds_w = Pph_ds_w.rolling(t=PimpData_rollwin, center=True).mean().dropna('t')
        vImp_Vals_w = (P - Pph_ds_w.values) / mI
        tvImp_Vals_w = Pph_ds_w['t'].values

        if tailFit is True:
            tfmask = tsVals > tfCutoff
            tfVals = tsVals[tfmask]
            tfLin = tsVals[tsVals > tfstart]
            zD = np.polyfit(np.log(tfVals), np.log(DynOv_w[tfmask]), deg=1)
            if longTime:
                tfLin_plot = tVals[tVals > tfstart]
            else:
                tfLin_plot = tfLin
            fLinD_plot = np.exp(zD[1]) * tfLin_plot**(zD[0])
            ax.plot(tfLin_plot / tscale, fLinD_plot, 'k--', label='')

        if longTime:
            DynOv_w_plot = np.abs(qds_w.isel(P=indP)['Real_DynOv'].values + 1j * qds_w.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
            ax.plot(tVals / tscale, DynOv_w_plot, label='{:.2f}'.format(P / mc), lw=3, color=colorList[ip])
        else:
            ax.plot(tsVals / tscale, DynOv_w, label='{:.2f}'.format(P / mc))

    ax.set_ylabel(r'$|S(t)|$', fontsize=18)
    ax.set_xlabel(r'$t/(\xi c^{-1})$', fontsize=18)

    if logScale is True:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.tick_params(which='both', direction='in', right=True, top=True)
    ax.tick_params(which='major', length=6, width=1)
    ax.tick_params(which='minor', length=3, width=1)
    ax.tick_params(axis='x', which='major', pad=10)
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.tick_params(axis='both', which='minor', labelsize=17)

    # ax.legend(title=r'$v_{\rm imp}(t_{0}) / c$')

    handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, title=r'$\langle v_{\rm imp}(t_{0})\rangle / c$', ncol=1, loc='center right', bbox_to_anchor=(0.11, 0.38)))
    fig.subplots_adjust(left=0.2, bottom=0.175, top=0.98, right=0.98)

    fig.legend(handles, labels, title=r'$v_{\rm imp}(t_{0}) / c$', loc=3, bbox_to_anchor=(0.25, 0.25), fontsize=18, title_fontsize=18)

    fig.set_size_inches(6, 3.9)
    filename = '/Fig3_PRL.pdf'
    fig.savefig(figdatapath + filename)

    # # # # FIG 4 - S(t) AND v_Imp CURVES (WEAK AND STRONG INTERACTIONS)

    # colorList = ['b', 'orange', 'g', 'r']

    # matplotlib.rcParams.update({'font.size': 20})

    # tailFit = True
    # logScale = True
    # PimpData_roll = False; PimpData_rollwin = 2
    # longTime = True
    # # tau = 100; tfCutoff = 90; tfstart = 10
    # tau = 300; tfCutoff = 200; tfstart = 10

    # aIBi_weak = -10.0
    # aIBi_strong = -2
    # print(aIBi_weak * xi, aIBi_strong * xi)

    # if longTime:
    #     innerdatapath_longtime = datapath + '_longtime/redyn_spherical'
    #     qds_w = xr.open_dataset(innerdatapath_longtime + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi_weak))
    #     qds_s = xr.open_dataset(innerdatapath_longtime + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi_strong))
    # else:
    #     qds_w = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi_weak))
    #     qds_s = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi_strong))

    # tVals = qds_w['t'].values
    # tsVals = tVals[tVals < tau]

    # qds_aIBi_ts_w = qds_w.sel(t=tsVals)
    # qds_aIBi_ts_s = qds_s.sel(t=tsVals)

    # # Pnorm_des = np.array([0.1, 0.5, 0.9, 1.4, 2.2, 3.0, 5.0])
    # Pnorm_des = np.array([0.5, 0.98, 2.2, 3.0])

    # Pinds = np.zeros(Pnorm_des.size, dtype=int)
    # for Pn_ind, Pn in enumerate(Pnorm_des):
    #     Pinds[Pn_ind] = np.abs(Pnorm - Pn).argmin().astype(int)

    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # for ip, indP in enumerate(Pinds):
    #     P = PVals[indP]
    #     DynOv_w = np.abs(qds_aIBi_ts_w.isel(P=indP)['Real_DynOv'].values + 1j * qds_aIBi_ts_w.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #     Pph_ds_w = xr.DataArray(qds_aIBi_ts_w.isel(P=indP)['Pph'].values, coords=[tsVals], dims=['t'])
    #     if PimpData_roll:
    #         Pph_ds_w = Pph_ds_w.rolling(t=PimpData_rollwin, center=True).mean().dropna('t')
    #     vImp_Vals_w = (P - Pph_ds_w.values) / mI
    #     tvImp_Vals_w = Pph_ds_w['t'].values

    #     if tailFit is True:
    #         tfmask = tsVals > tfCutoff
    #         tfVals = tsVals[tfmask]
    #         tfLin = tsVals[tsVals > tfstart]
    #         zD = np.polyfit(np.log(tfVals), np.log(DynOv_w[tfmask]), deg=1)
    #         if longTime:
    #             tfLin_plot = tVals[tVals > tfstart]
    #         else:
    #             tfLin_plot = tfLin
    #         fLinD_plot = np.exp(zD[1]) * tfLin_plot**(zD[0])
    #         axes[0, 0].plot(tfLin_plot / tscale, fLinD_plot, 'k--', label='')

    #     if longTime:
    #         DynOv_w_plot = np.abs(qds_w.isel(P=indP)['Real_DynOv'].values + 1j * qds_w.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #         axes[0, 0].plot(tVals / tscale, DynOv_w_plot, label='{:.2f}'.format(P / mc), lw=3, color=colorList[ip])
    #     else:
    #         axes[0, 0].plot(tsVals / tscale, DynOv_w, label='{:.2f}'.format(P / mc))
    #     axes[1, 0].plot(tvImp_Vals_w / tscale, vImp_Vals_w / nu, label='{:.2f}'.format(P / mc), lw=3, color=colorList[ip])

    #     DynOv_s = np.abs(qds_aIBi_ts_s.isel(P=indP)['Real_DynOv'].values + 1j * qds_aIBi_ts_s.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #     Pph_ds_s = xr.DataArray(qds_aIBi_ts_s.isel(P=indP)['Pph'].values, coords=[tsVals], dims=['t'])
    #     if PimpData_roll:
    #         Pph_ds_s = Pph_ds_s.rolling(t=PimpData_rollwin, center=True).mean().dropna('t')
    #     vImp_Vals_s = (P - Pph_ds_s.values) / mI
    #     tvImp_Vals_s = Pph_ds_s['t'].values

    #     if tailFit is True:
    #         tfmask = tsVals > tfCutoff
    #         tfVals = tsVals[tfmask]
    #         tfLin = tsVals[tsVals > tfstart]
    #         zD = np.polyfit(np.log(tfVals), np.log(DynOv_s[tfmask]), deg=1)
    #         if longTime:
    #             tfLin_plot = tVals[tVals > tfstart]
    #         else:
    #             tfLin_plot = tfLin
    #         fLinD_plot = np.exp(zD[1]) * tfLin_plot**(zD[0])
    #         axes[0, 1].plot(tfLin_plot / tscale, fLinD_plot, 'k--', label='')

    #     if longTime:
    #         DynOv_s_plot = np.abs(qds_s.isel(P=indP)['Real_DynOv'].values + 1j * qds_s.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #         axes[0, 1].plot(tVals / tscale, DynOv_s_plot, label='{:.2f}'.format(P / mc), lw=3, color=colorList[ip])
    #     else:
    #         axes[0, 1].plot(tsVals / tscale, DynOv_s, label='{:.2f}'.format(P / mc))
    #     axes[1, 1].plot(tvImp_Vals_s / tscale, vImp_Vals_s / nu, label='{:.2f}'.format(P / mc), lw=3, color=colorList[ip])

    # axes[0, 0].set_ylabel(r'$|S(t)|$', fontsize=27)
    # # axes[0, 0].set_xlabel(r'$t/(\xi c^{-1})$', fontsize=27)

    # axes[1, 0].plot(tsVals / tscale, np.ones(tsVals.size), 'k--', label='$c$')
    # axes[1, 0].set_ylabel(r'$v_{\rm imp}(t) / c$', fontsize=27)
    # axes[1, 0].set_xlabel(r'$t/(\xi c^{-1})$', fontsize=27)

    # if logScale is True:
    #     # axes[0, 0].plot(tlin_norm * np.ones(DynOv_w.size), np.linspace(np.min(DynOv_w), np.max(DynOv_w), DynOv_w.size), 'k-')
    #     axes[0, 0].set_xscale('log')
    #     axes[0, 0].set_yscale('log')
    #     # axes[0, 0].set_ylim([7e-2, 1e0])
    #     axes[1, 0].set_xscale('log')

    # # axes[0, 1].set_ylabel(r'$|S(t)|$', fontsize=27)
    # # axes[0, 1].set_xlabel(r'$t/(\xi c^{-1})$', fontsize=27)

    # axes[1, 1].plot(tsVals / tscale, np.ones(tsVals.size), 'k--', label='$c$')
    # # axes[1, 1].set_ylabel(r'$\langle v_{\rm imp}\rangle / c$', fontsize=27)
    # axes[1, 1].set_xlabel(r'$t/(\xi c^{-1})$', fontsize=27)

    # if logScale is True:
    #     # axes[0, 1].plot(tlin_norm * np.ones(DynOv_s.size), np.linspace(np.min(DynOv_s), np.max(DynOv_s), DynOv_s.size), 'k-')
    #     axes[0, 1].set_xscale('log')
    #     axes[0, 1].set_yscale('log')
    #     # axes[0, 1].set_ylim([7e-2, 1e0])
    #     axes[1, 1].set_xscale('log')

    # fig.text(0.06, 0.95, '(a)', fontsize=30)
    # fig.text(0.52, 0.95, '(b)', fontsize=30)
    # fig.text(0.06, 0.55, '(c)', fontsize=30)
    # fig.text(0.52, 0.55, '(d)', fontsize=30)

    # axes[0, 0].tick_params(which='both', direction='in', right=True, top=True)
    # axes[0, 1].tick_params(which='both', direction='in', right=True, top=True)
    # axes[1, 0].tick_params(which='both', direction='in', right=True, top=True)
    # axes[1, 1].tick_params(which='both', direction='in', right=True, top=True)

    # axes[0, 0].tick_params(which='major', length=6, width=1)
    # axes[0, 1].tick_params(which='major', length=6, width=1)
    # axes[1, 0].tick_params(which='major', length=6, width=1)
    # axes[1, 1].tick_params(which='major', length=6, width=1)

    # axes[0, 0].tick_params(which='minor', length=3, width=1)
    # axes[0, 1].tick_params(which='minor', length=3, width=1)
    # axes[1, 0].tick_params(which='minor', length=3, width=1)
    # axes[1, 1].tick_params(which='minor', length=3, width=1)

    # axes[0, 0].tick_params(axis='x', which='major', pad=10)
    # axes[0, 1].tick_params(axis='x', which='major', pad=10)
    # axes[1, 0].tick_params(axis='x', which='major', pad=10)
    # axes[1, 1].tick_params(axis='x', which='major', pad=10)

    # axes[0, 0].tick_params(axis='both', which='major', labelsize=20)
    # axes[0, 1].tick_params(axis='both', which='major', labelsize=20)
    # axes[1, 0].tick_params(axis='both', which='major', labelsize=20)
    # axes[1, 1].tick_params(axis='both', which='major', labelsize=20)

    # axes[0, 1].yaxis.set_major_formatter(NullFormatter())
    # axes[0, 1].yaxis.set_minor_formatter(NullFormatter())
    # # axes[0, 1].set_yticks([])
    # axes[0, 1].yaxis.set_ticklabels([])
    # axes[1, 1].yaxis.set_ticklabels([])

    # handles, labels = axes[0, 0].get_legend_handles_labels()
    # # fig.legend(handles, labels, title=r'$\langle v_{\rm imp}(t_{0})\rangle / c$', ncol=1, loc='center right', bbox_to_anchor=(0.11, 0.38))
    # # fig.subplots_adjust(left=0.16, bottom=0.1, top=0.925, right=0.95, wspace=0.25, hspace=0.32)

    # fig.legend(handles, labels, title=r'$v_{\rm imp}(t_{0}) / c$', ncol=Pnorm_des.size, loc='lower center', bbox_to_anchor=(0.55, 0.01), fontsize=25, title_fontsize=25)
    # fig.subplots_adjust(left=0.1, bottom=0.22, top=0.925, right=0.95, wspace=0.1, hspace=0.32)

    # fig.set_size_inches(16.9, 12)
    # filename = '/Fig4.pdf'
    # fig.savefig(figdatapath + filename)

    # # # # FIG 5 - LOSCHMIDT ECHO EXPONENTS + FINAL LOSCHMIDT ECHO + FINAL IMPURITY VELOCITY

    # DynOvData_roll = False
    # DynOvData_rollwin = 2
    # PimpData_roll = False
    # PimpData_rollwin = 2
    # DynOvExp_roll = False
    # DynOvExp_rollwin = 2
    # DynOvExp_NegMask = False
    # DynOvExp_Cut = False
    # cut = 1e-4
    # consecDetection = True
    # consecSamples = 10
    # flattenAboveC = True

    # aIBi_des = np.array([-10.0, -5.0, -3.5, -2.5, -2.0, -1.75])

    # Pnorm = PVals / mc

    # tmin = 90; tmax = 100

    # tfVals = tVals[(tVals <= tmax) * (tVals >= tmin)]

    # colorList = ['red', '#7e1e9c', 'green', 'orange', '#60460f', 'blue', 'magenta']
    # lineList = ['solid', 'dashed', 'dotted', '-.']

    # def powerfunc(t, a, b):
    #     return b * t**(-1 * a)

    # Pcrit_da = xr.DataArray(np.full(aIBi_des.size, np.nan, dtype=float), coords=[aIBi_des], dims=['aIBi'])

    # fig, axes = plt.subplots(nrows=3, ncols=1)
    # for inda, aIBi in enumerate(aIBi_des):
    #     qds_aIBi = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #     # print(qds_aIBi['t'].values)
    #     qds_aIBi_ts = qds_aIBi.sel(t=tfVals)
    #     PVals = qds_aIBi['P'].values
    #     Pnorm = PVals / mc
    #     DynOv_Exponents = np.zeros(PVals.size)
    #     DynOv_Cov = np.full(PVals.size, np.nan)
    #     vImp_Exponents = np.zeros(PVals.size)
    #     vImp_Cov = np.full(PVals.size, np.nan)

    #     Plen = PVals.size
    #     Pstart_ind = 0
    #     vI0_Vals = (PVals - qds_aIBi.isel(t=0, P=np.arange(Pstart_ind, Plen))['Pph'].values) / mI

    #     DynOv_Exponents = np.zeros(PVals.size)
    #     DynOv_Constants = np.zeros(PVals.size)

    #     vImp_Exponents = np.zeros(PVals.size)
    #     vImp_Constants = np.zeros(PVals.size)

    #     DynOv_Rvalues = np.zeros(PVals.size)
    #     DynOv_Pvalues = np.zeros(PVals.size)
    #     DynOv_stderr = np.zeros(PVals.size)
    #     DynOv_tstat = np.zeros(PVals.size)
    #     DynOv_logAve = np.zeros(PVals.size)

    #     for indP, P in enumerate(PVals):
    #         DynOv_raw = np.abs(qds_aIBi_ts.isel(P=indP)['Real_DynOv'].values + 1j * qds_aIBi_ts.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #         DynOv_ds = xr.DataArray(DynOv_raw, coords=[tfVals], dims=['t'])
    #         Pph_ds = xr.DataArray(qds_aIBi_ts.isel(P=indP)['Pph'].values, coords=[tfVals], dims=['t'])

    #         if DynOvData_roll:
    #             DynOv_ds = DynOv_ds.rolling(t=DynOvData_rollwin, center=True).mean().dropna('t')
    #         if PimpData_roll:
    #             Pph_ds = Pph_ds.rolling(t=PimpData_rollwin, center=True).mean().dropna('t')

    #         DynOv_Vals = DynOv_ds.values
    #         tDynOv_Vals = DynOv_ds['t'].values

    #         vImpc_Vals = (P - Pph_ds.values) / mI - nu
    #         tvImpc_Vals = Pph_ds['t'].values

    #         S_slope, S_intercept, S_rvalue, S_pvalue, S_stderr = ss.linregress(np.log(tDynOv_Vals), np.log(DynOv_Vals))
    #         DynOv_Exponents[indP] = -1 * S_slope
    #         DynOv_Constants[indP] = np.exp(S_intercept)

    #         DynOv_Rvalues[indP] = S_rvalue
    #         DynOv_Pvalues[indP] = S_pvalue
    #         DynOv_stderr[indP] = S_stderr
    #         DynOv_tstat[indP] = S_slope / S_stderr
    #         DynOv_logAve[indP] = np.average(np.log(DynOv_Vals))

    #         # if (-1 * S_slope) < 0:
    #         #     DynOv_Exponents[indP] = 0

    #         if vImpc_Vals[-1] < 0:
    #             vImp_Exponents[indP] = 0
    #             vImp_Constants[indP] = vImpc_Vals[-1]
    #         else:
    #             vI_slope, vI_intercept, vI_rvalue, vI_pvalue, vI_stderr = ss.linregress(np.log(tvImpc_Vals), np.log(vImpc_Vals))
    #             vImp_Exponents[indP] = -1 * vI_slope
    #             vImp_Constants[indP] = np.exp(vI_intercept)
    #             if (-1 * vI_slope) < 0:
    #                 vImp_Exponents[indP] = 0

    #     DynOvExponents_da = xr.DataArray(DynOv_Exponents, coords=[PVals], dims=['P'])
    #     if DynOvExp_roll:
    #         DynOvExponents_da = DynOvExponents_da.rolling(P=DynOvExp_rollwin, center=True).mean().dropna('P')
    #     if DynOvExp_NegMask:
    #         ExpMask = DynOvExponents_da.values < 0
    #         DynOvExponents_da[ExpMask] = 0
    #     if DynOvExp_Cut:
    #         ExpMask = np.abs(DynOvExponents_da.values) < cut
    #         DynOvExponents_da[ExpMask] = 0
    #     DynOv_Exponents = DynOvExponents_da.values
    #     if consecDetection:
    #         crit_ind = 0
    #         for indE, exp in enumerate(DynOv_Exponents):
    #             if indE > DynOv_Exponents.size - consecDetection:
    #                 break
    #             expSlice = DynOv_Exponents[indE:(indE + consecSamples)]
    #             if np.all(expSlice > 0):
    #                 crit_ind = indE
    #                 break
    #         DynOvExponents_da[0:crit_ind] = 0

    #     DynOv_Exponents = DynOvExponents_da.values
    #     Pnorm_dynov = DynOvExponents_da['P'].values / mc
    #     DynOvf_Vals = powerfunc(1e1000, DynOv_Exponents, DynOv_Constants)
    #     Pcrit_da[inda] = PVals[crit_ind] / (mI * nu)

    #     vIf_Vals = nu + powerfunc(1e1000, vImp_Exponents, vImp_Constants)
    #     if flattenAboveC:
    #         vIf_Vals[vIf_Vals > nu] = nu

    #     axes[0].plot(Pnorm_dynov, DynOv_Exponents, color=colorList[inda], linestyle='solid', label='{:.2f}'.format(aIBi * xi), marker='D')
    #     # ax1.plot(Pnorm, vImp_Exponents, color=colorList[inda], linestyle='dotted', marker='+', markerfacecolor='none', label='{:.2f}'.format(aIBi))
    #     axes[1].plot(vI0_Vals / nu, DynOvf_Vals, color=colorList[inda], linestyle='solid', marker='D')
    #     axes[2].plot(vI0_Vals / nu, vIf_Vals / nu, color=colorList[inda], linestyle='solid', marker='D')

    # # axes[0].set_xlabel(r'$\langle v_{I}(t_{0})\rangle/c$', fontsize=20))
    # axes[0].set_ylabel(r'$\gamma$' + ' for ' + r'$|S(t)|\propto t^{-\gamma}$', fontsize=20)
    # axes[0].set_xlim([0, 4])
    # axes[0].set_ylim([-.02, 0.25])

    # # axes[1].set_xlabel(r'$\langle v_{I}(t_{0})\rangle/c$', fontsize=20))
    # axes[1].set_ylabel(r'$S(t_{\infty})$', fontsize=20)
    # axes[1].set_xlim([0, 4])
    # axes[1].set_ylim([-.05, 1.1])

    # axes[2].plot(vI0_Vals / nu, np.ones(vI0_Vals.size), 'k:')
    # axes[2].set_xlabel(r'$v_{\rm imp}(t_{0})/c$', fontsize=20)
    # axes[2].set_ylabel(r'$v_{\rm imp}(t_{\infty})/c$', fontsize=20)
    # axes[2].set_xlim([0, 4])
    # axes[2].set_ylim([-.03, 1.1])

    # fig.text(0.03, 0.97, '(a)', fontsize=20)
    # fig.text(0.03, 0.7, '(b)', fontsize=20)
    # fig.text(0.03, 0.42, '(c)', fontsize=20)

    # axes[0].xaxis.set_ticklabels([])
    # axes[1].xaxis.set_ticklabels([])

    # axes[0].tick_params(which='both', direction='in', right=True, top=True)
    # axes[1].tick_params(which='both', direction='in', right=True, top=True)
    # axes[2].tick_params(which='both', direction='in', right=True, top=True)

    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, title=r'$a_{\rm IB}^{-1}/\xi^{-1}$', ncol=aIBi_des.size // 2, loc='lower center', bbox_to_anchor=(0.55, 0.01))
    # fig.subplots_adjust(left=0.2, bottom=0.17, top=0.97, right=0.97, hspace=0.15)
    # fig.set_size_inches(6, 12)
    # fig.savefig(figdatapath + '/Fig5.pdf')

    # # # # FIG DPT - NESS + GS PHASE DIAGRAM

    # DynOvExp_NegMask = False
    # DynOvExp_Cut = False
    # cut = 1e-4
    # consecDetection = True
    # consecSamples = 10

    # def powerfunc(t, a, b):
    #     return b * t**(-1 * a)

    # tmin = 90
    # tmax = 100
    # tfVals = tVals[(tVals <= tmax) * (tVals >= tmin)]
    # rollwin = 1

    # colorList = ['red', '#7e1e9c', 'green', 'orange', '#60460f', 'blue', 'magenta']
    # lineList = ['solid', 'dashed', 'dotted', '-.']
    # aIBi_des = np.array([-10.0, -5.0, -3.5, -2.5, -2.0, -1.75])
    # massRat_des = np.array([1.0])

    # mdatapaths = []

    # for mR in massRat_des:
    #     if toggleDict['noCSAmp'] is True:
    #         mdatapaths.append(datapath[0:-11] + '{:.1f}_noCSAmp'.format(mR))
    #     else:
    #         mdatapaths.append(datapath[0:-3] + '{:.1f}_noCSAmp'.format(mR))
    # if toggleDict['Dynamics'] != 'real' or toggleDict['Grid'] != 'spherical' or toggleDict['Coupling'] != 'twophonon':
    #     print('SETTING ERROR')

    # Pcrit_da = xr.DataArray(np.full((massRat_des.size, aIBi_des.size), np.nan, dtype=float), coords=[massRat_des, aIBi_des], dims=['mRatio', 'aIBi'])
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
    #             # DynOv_ds = DynOv_ds.rolling(t=rollwin, center=True).mean().dropna('t')
    #             DynOv_Vals = DynOv_ds.values

    #             tDynOvc_Vals = DynOv_ds['t'].values

    #             S_slope, S_intercept, S_rvalue, S_pvalue, S_stderr = ss.linregress(np.log(tDynOvc_Vals), np.log(DynOv_Vals))
    #             DynOv_Exponents[indP] = -1 * S_slope
    #             DynOv_Constants[indP] = np.exp(S_intercept)

    #         if DynOvExp_NegMask:
    #             DynOv_Exponents[DynOv_Exponents < 0] = 0

    #         if DynOvExp_Cut:
    #             DynOv_Exponents[np.abs(DynOv_Exponents) < cut] = 0

    #         if consecDetection:
    #             crit_ind = 0
    #             for indE, exp in enumerate(DynOv_Exponents):
    #                 if indE > DynOv_Exponents.size - consecDetection:
    #                     break
    #                 expSlice = DynOv_Exponents[indE:(indE + consecSamples)]
    #                 if np.all(expSlice > 0):
    #                     crit_ind = indE
    #                     break
    #             DynOv_Exponents[0:crit_ind] = 0
    #         Pcrit_da[indm, inda] = PVals[crit_ind] / (mI * nu)
    #         DynOvf_Vals = powerfunc(1e1000, DynOv_Exponents, DynOv_Constants)

    # PcritInterp = False
    # plotGS = True

    # Pcrit_interpVals_mRat1 = 0
    # fig2, ax2 = plt.subplots()
    # for indm, massRat in enumerate(massRat_des):
    #     if PcritInterp is True:
    #         Pcrit_norm = Pcrit_da.sel(mRatio=massRat).values
    #         Pcrit_tck = interpolate.splrep(aIBi_des, Pcrit_norm, s=0, k=1)
    #         aIBi_interpVals = np.linspace(np.min(aIBi_des), np.max(aIBi_des), 2 * aIBi_des.size)
    #         Pcrit_interpVals = 1 * interpolate.splev(aIBi_interpVals, Pcrit_tck, der=0)
    #     else:
    #         aIBi_interpVals = aIBi_des
    #         Pcrit_interpVals = Pcrit_da.sel(mRatio=massRat).values

    #     if massRat == 1.0:
    #         Pcrit_interpVals_mRat1 = Pcrit_interpVals

    #     # ax2.plot(aIBi_interpVals /xi, Pcrit_interpVals, color='k', linestyle=lineList[indm], label='{0}'.format(massRat))
    #     # ax2.plot(aIBi_interpVals / xi, Pcrit_interpVals, color='k', linestyle=lineList[indm], label='NESS')
    #     ax2.plot(aIBi_des / xi, Pcrit_da.sel(mRatio=massRat).values, 'kx', mew=2, ms=12, label='NESS')

    # xmin = np.min(aIBi_interpVals / xi)
    # xmax = 1.01 * np.max(aIBi_interpVals / xi)
    # ymin = 0
    # ymax = 1.01 * np.max(Pcrit_da.values)
    # font = {'family': 'serif', 'color': 'black', 'size': 16}
    # sfont = {'family': 'serif', 'color': 'black', 'size': 15}

    # if massRat_des.size > 1:
    #     ax2.legend(title=r'$m_{I}/{m_{B}$', loc=2)
    # ax2.set_xlabel(r'$a_{IB}^{-1}$ [$\xi$]', fontsize=20)
    # ax2.set_ylabel(r'$\langle v_{I}(t_{0})\rangle/c$', fontsize=20)
    # ax2.text(-4.5, ymin + 0.175 * (ymax - ymin), 'Polaron', fontdict=font)
    # ax2.text(-4.4, ymin + 0.1 * (ymax - ymin), '(' + r'$S(t_{\infty})>0$' + ')', fontdict=sfont)
    # ax2.text(-7.0, ymin + 0.63 * (ymax - ymin), 'Cherenkov', fontdict=font)
    # ax2.text(-6.85, ymin + 0.555 * (ymax - ymin), '(' + r'$S(t_{\infty})=0$' + ')', fontdict=sfont)
    # # ax2.fill_between(aIBi_interpVals / xi, Pcrit_interpVals_mRat1, ymax, facecolor='b', alpha=0.25)
    # # ax2.fill_between(aIBi_interpVals / xi, ymin, Pcrit_interpVals_mRat1, facecolor='g', alpha=0.25)
    # ax2.set_xlim([xmin, xmax])
    # ax2.set_ylim([ymin, ymax])

    # if plotGS is True:
    #     gs_datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_1.44E+06/massRatio=1.0/imdyn_spherical'
    #     aIBi_Vals = np.array([-10.0, -9.0, -8.0, -7.0, -5.0, -3.5, -2.0, -1.0])  # used by many plots (spherical)
    #     Pcrit = np.zeros(aIBi_Vals.size)
    #     for aind, aIBi in enumerate(aIBi_Vals):
    #         qds_aIBi = xr.open_dataset(gs_datapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #         PVals = qds_aIBi['P'].values
    #         CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    #         kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)

    #         Energy_Vals_inf = np.zeros(PVals.size)
    #         for Pind, P in enumerate(PVals):
    #             CSAmp = CSAmp_ds.sel(P=P).isel(t=-1).values
    #             Energy_Vals_inf[Pind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)

    #         Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)
    #         Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 2 * PVals.size)
    #         Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
    #         Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)
    #         Pcrit[aind] = Pinf_Vals[np.argmin(np.gradient(Einf_2ndderiv_Vals)) - 0]

    #     Pcrit_norm = Pcrit / (mI * nu)
    #     Pcrit_tck = interpolate.splrep(aIBi_Vals, Pcrit_norm, s=0, k=3)
    #     aIBi_interpVals = np.linspace(np.min(aIBi_Vals), np.max(aIBi_Vals), 5 * aIBi_Vals.size)
    #     Pcrit_interpVals = 1 * interpolate.splev(aIBi_interpVals, Pcrit_tck, der=0)
    #     ax2.plot(aIBi_interpVals / xi, Pcrit_interpVals, color='k', linestyle='solid', label='Ground State')
    #     ax2.fill_between(aIBi_interpVals / xi, Pcrit_interpVals, ymax, facecolor='b', alpha=0.25)
    #     ax2.fill_between(aIBi_interpVals / xi, ymin, Pcrit_interpVals, facecolor='g', alpha=0.25)
    #     ax2.legend(loc=2)

    # fig2.set_size_inches(6, 4.5)
    # fig2.subplots_adjust(bottom=0.17, top=0.97, left=0.15, right=0.97)
    # # fig2.savefig(figdatapath + '/FigDPT.pdf')

    # # # # FIG 6 - PARTICIPATION RATIO CURVES VS INITIAL VELOCITY (SPHERICAL APPROXIMATION TO CARTESIAN INTERPOLATION)

    # # NOTE: We need the massRatio_1.0_old folder (or technically any of the _old folders) and the constants determined at the beginning of the script for this to run

    # inversePlot = True

    # # PRtype = 'continuous'
    # PRtype = 'discrete'; discPR_norm = True

    # Vol_fac = False

    # tau = 2.3
    # # tau = 5

    # # NOTE: The following constants are grid dependent (both on original spherical grid and interpolated cartesian grid)
    # dVk_cart = 0.0001241449577749997  # = dkx*dky*dkz from cartesian interpolation
    # Npoints_xyz = 85184000
    # Vxyz = 1984476.915083265
    # contToDisc_factor = dVk_cart / ((2 * np.pi)**3)

    # colorList = ['red', '#7e1e9c', '#60460f', '#658b38']
    # # colorList = ['red', '#7e1e9c', 'green', 'orange', '#60460f', 'blue', 'magenta']
    # lineList = ['solid', 'dotted', 'dashed', 'dashdot']
    # aIBi_des = np.array([-10.0, -5.0, -2.0, -1.5])
    # # aIBi_des = np.array([-10.0, -5.0, -2.0, -1.5, -1.25, -1.0])
    # massRat_des = np.array([1.0])
    # # massRat_des = np.array([0.5, 1.0, 2])
    # mdatapaths = []

    # for mR in massRat_des:
    #     mdatapaths.append('/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}_resRat_{:.2f}/massRatio={:.1f}'.format(NGridPoints_cart, resRat, mR))

    # if toggleDict['Dynamics'] != 'real' or toggleDict['Grid'] != 'spherical' or toggleDict['Coupling'] != 'twophonon':
    #     print('SETTING ERROR')

    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds_aIBi.coords['k'].values); kgrid.initArray_premade('th', qds_aIBi.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # dVk = kgrid.dV()
    # print(kVec[-1], kVec[1] - kVec[0])

    # PRcont_Averages = np.zeros(PVals.size)
    # PRdisc_Averages = np.zeros(PVals.size)

    # P_Vals_norm = np.concatenate((np.linspace(0.1, 0.8, 5, endpoint=False), np.linspace(0.8, 1.4, 10, endpoint=False), np.linspace(1.4, 3.0, 12, endpoint=False), np.linspace(3.0, 5.0, 10, endpoint=False), np.linspace(5.0, 9.0, 20)))

    # fig1, ax1 = plt.subplots()
    # for inda, aIBi in enumerate(aIBi_des):
    #     for indm, mRat in enumerate(massRat_des):

    #         vI0_Vals = np.zeros(PVals.size)
    #         PR_Averages = np.zeros(PVals.size)
    #         PVals = mRat * mB * nu * P_Vals_norm

    #         for indP, P in enumerate(PVals):
    #             qds_PaIBi = xr.open_dataset(mdatapaths[indm] + '/redyn_spherical/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    #             CSAmp_ds = (qds_PaIBi['Real_CSAmp'] + 1j * qds_PaIBi['Imag_CSAmp'])
    #             Nph_ds = qds_PaIBi['Nph']
    #             mI = qds_PaIBi.attrs['mI']

    #             if Lx == 60:
    #                 CSAmp_ds = CSAmp_ds.rename({'tc': 't'})

    #             tsVals = CSAmp_ds.coords['t'].values
    #             tsVals = tsVals[tsVals <= tau]
    #             CSAmp_ds = CSAmp_ds.sel(t=tsVals)
    #             Nph_ds = Nph_ds.sel(t=tsVals)

    #             PR_Vals = np.zeros(tsVals.size)

    #             dt = tsVals[1] - tsVals[0]

    #             for indt, t in enumerate(tsVals):
    #                 CSAmp_Vals = CSAmp_ds.sel(t=t).values
    #                 Bk_2D_vals = CSAmp_Vals.reshape((len(kVec), len(thVec)))

    #                 PhDen_Vals = ((2 * np.pi)**(-3)) * ((1 / Nph_ds.sel(t=t).values) * np.abs(Bk_2D_vals)**2).real.astype(float)
    #                 dVk_n = ((2 * np.pi)**(3)) * dVk
    #                 PR_Vals[indt] = (2 * np.pi)**3 * np.dot((PhDen_Vals**2).flatten(), dVk_n)

    #             vI0_Vals[indP] = (P - qds_PaIBi.isel(t=0)['Pph'].values) / mI
    #             PR_Vals_del = np.delete(PR_Vals, 0); PR_Averages[indP] = (1 / (tsVals[-1] - tsVals[1])) * simps(y=PR_Vals_del, dx=dt)

    #         if Vol_fac is True:
    #             PR_Averages = Vxyz * PR_Averages
    #         else:
    #             PR_Averages = PR_Averages

    #         if PRtype == 'continuous':
    #             PR_Averages = PR_Averages
    #         elif PRtype == 'discrete':
    #             PR_Averages = PR_Averages * contToDisc_factor
    #             if discPR_norm is True:
    #                 PR_Averages = PR_Averages * Npoints_xyz

    #         if inversePlot is True:
    #             ax1.plot(vI0_Vals / nu, 1 / PR_Averages, linestyle=lineList[indm], color=colorList[inda])
    #         else:
    #             ax1.plot(vI0_Vals / nu, PR_Averages, linestyle=lineList[indm], color=colorList[inda])

    # alegend_elements = []
    # mlegend_elements = []
    # for inda, aIBi in enumerate(aIBi_des):
    #     alegend_elements.append(Line2D([0], [0], color=colorList[inda], linestyle='solid', label='{:.2f}'.format(aIBi / xi)))
    # for indm, mR in enumerate(massRat_des):
    #     mlegend_elements.append(Line2D([0], [0], color='magenta', linestyle=lineList[indm], label='{0}'.format(mR)))

    # ax1.set_xlabel(r'$\langle v_{I}(t_{0})\rangle /c$', fontsize=20)

    # if inversePlot is True:
    #     # ax1.set_title('Short-Time-Averaged Inverse Participation Ratio (' + r'$t\in[0, $' + '{:.2f}'.format(tau / tscale) + r'$\frac{\xi}{c}]$)')
    #     if PRtype == 'continuous':
    #         ax1.set_ylabel(r'Average $IPR$ with $IPR = ((2\pi)^{3} \int d^3\vec{k} (\frac{1}{(2\pi)^3}\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2})^{2})^{-1}$')
    #     elif PRtype == 'discrete':
    #         if discPR_norm is True:
    #             # ax1.set_ylabel(r'Average $IPR$ (Normalized by $N_{tot}$ modes in system)')
    #             ax1.set_ylabel(r'$\overline{IPR}/N_{tot}$', fontsize=20)
    #         else:
    #             ax1.set_ylabel(r'Average $IPR$')
    # else:
    #     # ax1.set_title('Time-Averaged Participation Ratio (' + r'$t\in[0, $' + '{:.2f}'.format(tau / tscale) + r'$\frac{\xi}{c}]$)')
    #     ax1.set_ylabel(r'Average $PR$ with $PR = (2\pi)^{3} \int d^3\vec{k} (\frac{1}{(2\pi)^3}\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2})^{2}$')
    # alegend = ax1.legend(handles=alegend_elements, loc=2, title=r'$a_{IB}^{-1}$ [$\xi$]', ncol=2)
    # plt.gca().add_artist(alegend)
    # # mlegend = ax1.legend(handles=mlegend_elements, loc=(0.22, 0.70), ncol=2, title=r'$m_{I}/m_{B}$')
    # # plt.gca().add_artist(mlegend)
    # # ax1.set_xlim([0, np.max(vI0_Vals / nu)])
    # ax1.set_xlim([0, 4])
    # ax1.set_ylim([0.004, 0.009])
    # # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax1.xaxis.set_major_locator(plt.MaxNLocator(4))

    # fig1.set_size_inches(6, 3.9)
    # fig1.subplots_adjust(bottom=0.17, top=0.94, right=0.97)
    # # fig1.savefig(figdatapath + '/Fig6.pdf')

    # # # FIG 7 - INDIVIDUAL PHONON MOMENTUM DISTRIBUTION PLOT SLICES

    # matplotlib.rcParams.update({'font.size': 18})

    # class HandlerEllipse(HandlerPatch):
    #     def create_artists(self, legend, orig_handle,
    #                        xdescent, ydescent, width, height, fontsize, trans):
    #         center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
    #         p = Ellipse(xy=center, width=width + xdescent,
    #                     height=height + ydescent)
    #         self.update_prop(p, orig_handle, legend)
    #         p.set_transform(trans)
    #         return [p]

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
    #     # patch_klin = plt.Circle((0, 0), klin, edgecolor='tab:cyan', facecolor='None')
    #     # ax.add_patch(patch_klin)

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
    #     ax.set_title(r'$t$ [$\xi/c$]: ' + '{:1.2f}'.format(tsVals[tninds[tind]] / tscale))
    #     ax.set_xlabel(r'$k_{z}$')
    #     ax.set_ylabel(r'$k_{x}$')

    # curve1_LE = Line2D([0], [0], color='none', lw=0, marker='x', markerfacecolor='xkcd:steel grey', markeredgecolor='xkcd:steel grey', markersize=10)
    # curve1m_LE = Line2D([0], [0], color='none', lw=0, marker='o', markerfacecolor='xkcd:apple green', markeredgecolor='xkcd:apple green', markersize=10)
    # curve2_LE = Line2D([0], [0], color='none', lw=0, marker='*', markerfacecolor='cyan', markeredgecolor='cyan', markersize=10)
    # patch_Excitation_LE = Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='red', markersize=20, mew=2)
    # # patch_klin_LE = Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='tab:cyan', markersize=20, mew=2)
    # patch_FGR_ph_LE = Ellipse(xy=(0, 0), width=0.2, height=0.1, angle=0, edgecolor='tab:gray', facecolor='none', lw=3)
    # patch_FGR_imp_LE = Ellipse(xy=(0, 0), width=0.2, height=0.1, angle=0, edgecolor='xkcd:military green', facecolor='none', lw=3)

    # if IRpatch is True:
    #     handles = (curve1_LE, curve1m_LE, curve2_LE, patch_Excitation_LE, patch_IR, patch_FGR_ph_LE, patch_FGR_imp_LE)
    #     labels = (r'$\langle P_{ph} \rangle$', r'$\langle P_{I} \rangle$', r'$(m_{I}c)\vec{e}_{k_{z}}$', r'$\omega_{|k|}^{-1}(\frac{2\pi}{t})$', r'Singular Region', 'FGR Phase Space (ph)', 'FGR Phase Space (imp)')
    # else:
    #     handles = (curve1_LE, curve1m_LE, curve2_LE, patch_Excitation_LE, patch_FGR_ph_LE, patch_FGR_imp_LE)
    #     labels = (r'$\langle \mathbf{P}_{\rm ph} \rangle$', r'$\langle \mathbf{P}_{\rm imp} \rangle$', r'$(m_{I}c)\mathbf{e}_{k_{z}}$', r'$\omega_{\mathbf{k}}^{-1}(\frac{2\pi}{t})$', 'FGR Phase Space (ph)', 'FGR Phase Space (imp)')

    # cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.7])
    # fig.colorbar(quad1, cax=cbar_ax, extend='both')
    # fig.legend(handles, labels, ncol=3, loc='lower center', handler_map={Ellipse: HandlerEllipse()})

    # fig.text(0.05, 0.97, '(a)', fontsize=20)
    # fig.text(0.05, 0.68, '(c)', fontsize=20)
    # fig.text(0.05, 0.38, '(e)', fontsize=20)
    # fig.text(0.47, 0.97, '(b)', fontsize=20)
    # fig.text(0.47, 0.68, '(d)', fontsize=20)
    # fig.text(0.47, 0.38, '(f)', fontsize=20)

    # fig.set_size_inches(12, 12)
    # fig.subplots_adjust(bottom=0.17, top=0.95, right=0.85, hspace=0.6, wspace=0.4)
    # # fig.savefig(figdatapath + '/Fig7.pdf', dpi=20)
    # fig.savefig(figdatapath + '/Fig7.jpg', quality=100)

    # # # FIG 7 - INDIVIDUAL PHONON MOMENTUM DISTRIBUTION PLOT SLICES (OLD)

    # matplotlib.rcParams.update({'font.size': 18})

    # class HandlerEllipse(HandlerPatch):
    #     def create_artists(self, legend, orig_handle,
    #                        xdescent, ydescent, width, height, fontsize, trans):
    #         center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
    #         p = Ellipse(xy=center, width=width + xdescent,
    #                     height=height + ydescent)
    #         self.update_prop(p, orig_handle, legend)
    #         p.set_transform(trans)
    #         return [p]

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
    #     ax.set_title(r'$t$ [$\xi/c$]: ' + '{:1.2f}'.format(tsVals[tninds[tind]] / tscale))
    #     ax.set_xlabel(r'$k_{z}$')
    #     ax.set_ylabel(r'$k_{x}$')

    # curve1_LE = Line2D([0], [0], color='none', lw=0, marker='x', markerfacecolor='xkcd:steel grey', markeredgecolor='xkcd:steel grey', markersize=10)
    # curve1m_LE = Line2D([0], [0], color='none', lw=0, marker='o', markerfacecolor='xkcd:apple green', markeredgecolor='xkcd:apple green', markersize=10)
    # curve2_LE = Line2D([0], [0], color='none', lw=0, marker='*', markerfacecolor='cyan', markeredgecolor='cyan', markersize=10)
    # patch_Excitation_LE = Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='red', markersize=20, mew=2)
    # patch_klin_LE = Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='tab:cyan', markersize=20, mew=2)
    # patch_FGR_ph_LE = Ellipse(xy=(0, 0), width=0.2, height=0.1, angle=0, edgecolor='tab:gray', facecolor='none', lw=3)
    # patch_FGR_imp_LE = Ellipse(xy=(0, 0), width=0.2, height=0.1, angle=0, edgecolor='xkcd:military green', facecolor='none', lw=3)

    # if IRpatch is True:
    #     handles = (curve1_LE, curve1m_LE, curve2_LE, patch_Excitation_LE, patch_IR, patch_klin_LE, patch_FGR_ph_LE, patch_FGR_imp_LE)
    #     labels = (r'$\langle P_{ph} \rangle$', r'$\langle P_{I} \rangle$', r'$(m_{I}c)\vec{e}_{k_{z}}$', r'$\omega_{|k|}^{-1}(\frac{2\pi}{t})$', r'Singular Region', r'Linear Excitations', 'FGR Phase Space (ph)', 'FGR Phase Space (imp)')
    # else:
    #     handles = (curve1_LE, curve1m_LE, curve2_LE, patch_Excitation_LE, patch_klin_LE, patch_FGR_ph_LE, patch_FGR_imp_LE)
    #     labels = (r'$\langle P_{ph} \rangle$', r'$\langle P_{I} \rangle$', r'$(m_{I}c)\vec{e}_{k_{z}}$', r'$\omega_{|k|}^{-1}(\frac{2\pi}{t})$', r'Linear Excitations', 'FGR Phase Space (ph)', 'FGR Phase Space (imp)')

    # cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.7])
    # fig.colorbar(quad1, cax=cbar_ax, extend='both')
    # fig.legend(handles, labels, ncol=4, loc='lower center', handler_map={Ellipse: HandlerEllipse()})

    # fig.text(0.05, 0.97, '(a)', fontsize=20)
    # fig.text(0.05, 0.68, '(c)', fontsize=20)
    # fig.text(0.05, 0.38, '(e)', fontsize=20)
    # fig.text(0.47, 0.97, '(b)', fontsize=20)
    # fig.text(0.47, 0.68, '(d)', fontsize=20)
    # fig.text(0.47, 0.38, '(f)', fontsize=20)

    # fig.set_size_inches(12, 12)
    # fig.subplots_adjust(bottom=0.17, top=0.95, right=0.85, hspace=0.6, wspace=0.4)
    # # fig.savefig(figdatapath + '/Fig7.pdf', dpi=20)
    # fig.savefig(figdatapath + '/Fig7.jpg', quality=20)

    # plt.show()
