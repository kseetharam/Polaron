import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from matplotlib.animation import writers
from matplotlib.patches import ConnectionPatch
import matplotlib.image as mpimg
import os
import itertools
import pf_dynamic_cart as pfc
import pf_dynamic_sph as pfs
import Grid
from scipy import interpolate
from timeit import default_timer as timer
import scipy.stats as ss
import colors as col

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})
    mpegWriter = writers['ffmpeg'](fps=0.75, bitrate=1800)
    matplotlib.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman', 'text.usetex': True, 'mathtext.fontset': 'dejavuserif'})

    axl = matplotlib.rcParams['axes.linewidth']

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    # (Lx, Ly, Lz) = (105, 105, 105)
    # (dx, dy, dz) = (0.375, 0.375, 0.375)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # Toggle parameters

    toggleDict = {'Dynamics': 'imaginary', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon', 'IRcuts': 'false', 'ReducedInterp': 'false', 'kGrid_ext': 'false'}

    # ---- SET OUTPUT DATA FOLDER ----

    datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, 1)
    animpath = '/Users/kis/Dropbox/VariationalResearch/DataAnalysis/figs'

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

    if toggleDict['IRcuts'] == 'true':
        innerdatapath = innerdatapath + '_IRcuts'
    elif toggleDict['IRcuts'] == 'false':
        innerdatapath = innerdatapath

    print(innerdatapath)

    # figdatapath = '/Users/kis/Dropbox/Apps/Overleaf/Quantum Cherenkov Transition in Bose Polaron Systems/figures/figdump'
    figdatapath = '/Users/kis/Dropbox/Apps/Overleaf/Cherenkov Polaron Paper pt1/figures/figdump'
    innerdatapath_cart = innerdatapath[0:-10] + '_cart'

    # # Analysis of Total Dataset

    base02 = col.base02.ashexstring()
    base2 = col.base2.ashexstring()
    red = col.red.ashexstring()
    green = col.green.ashexstring()
    cyan = col.cyan.ashexstring()
    blue = col.blue.ashexstring()

    aIBi = -2
    # qds = xr.open_dataset(innerdatapath + '/quench_Dataset.nc')
    # qds_aIBi = qds.sel(aIBi=aIBi)
    qds = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    qds_aIBi = qds

    PVals = qds['P'].values
    tVals = qds['t'].values
    n0 = qds.attrs['n0']
    gBB = qds.attrs['gBB']
    mI = qds.attrs['mI']
    mB = qds.attrs['mB']
    nu = np.sqrt(n0 * gBB / mB)
    aBB = (mB / (4 * np.pi)) * gBB
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)
    print(qds.attrs['k_mag_cutoff'] * xi)

    aIBi_Vals = np.array([-12.5, -10.0, -9.0, -8.0, -7.0, -5.0, -3.5, -2.0, -1.0, -0.75, -0.5, -0.1])  # used by many plots (spherical)

    # # # # FIG SCHEMATIC - POLARON GRAPHIC + BOGO DISPERSION + POLARON DISPERSION

    # matplotlib.rcParams.update({'font.size': 12})
    # labelsize = 13
    # legendsize = 12

    # fig1 = plt.figure(constrained_layout=False)
    # # gs1 = fig1.add_gridspec(nrows=1, ncols=1, bottom=0.1, top=0.95, left=0.05, right=0.2)
    # # gs2 = fig1.add_gridspec(nrows=1, ncols=1, bottom=0.15, top=0.91, left=0.32, right=0.58)
    # # gs3 = fig1.add_gridspec(nrows=1, ncols=1, bottom=0.15, top=0.91, left=0.7, right=0.97)

    # gs2 = fig1.add_gridspec(nrows=1, ncols=1, bottom=0.15, top=0.91, left=0.08, right=0.45)
    # gs3 = fig1.add_gridspec(nrows=1, ncols=1, bottom=0.15, top=0.91, left=0.57, right=0.97)

    # # ax_pol = fig1.add_subplot(gs1[0], frame_on=False); ax_pol.get_xaxis().set_visible(False); ax_pol.get_yaxis().set_visible(False)
    # ax_bogo = fig1.add_subplot(gs2[0])
    # ax_gsE = fig1.add_subplot(gs3[0])

    # # fig1.text(0.01, 0.95, '(a)', fontsize=labelsize)
    # # fig1.text(0.24, 0.95, '(b)', fontsize=labelsize)
    # # fig1.text(0.65, 0.95, '(c)', fontsize=labelsize)
    # fig1.text(0.01, 0.95, '(a)', fontsize=labelsize)
    # fig1.text(0.52, 0.95, '(b)', fontsize=labelsize)
    # fig1.set_size_inches(7.8, 3.5)

    # # # POLARON GRAPHIC

    # # polimg = mpimg.imread('images/PolaronGraphic.png')
    # # imgplot = ax_pol.imshow(polimg)

    # # BOGOLIUBOV DISPERSION (SPHERICAL)

    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds.coords['k'].values); kgrid.initArray_premade('th', qds.coords['th'].values)
    # kVals = kgrid.getArray('k')
    # wk_Vals = pfs.omegak(kVals, mB, n0, gBB)
    # mask = (wk_Vals < 2) * (wk_Vals > 0)
    # ax_bogo.plot(kVals[mask], wk_Vals[mask], 'k-', label='')
    # ax_bogo.plot(kVals[mask], nu * kVals[mask], color=red, linestyle='--', label=r'$c|\mathbf{k}|$')
    # ax_bogo.set_xlabel(r'$|\mathbf{k}|$', fontsize=labelsize)
    # ax_bogo.set_ylabel(r'$\omega_{|\mathbf{k}|}$', fontsize=labelsize)
    # ax_bogo.set_xlim([0 - 0.09, np.max(kVals[mask]) + 0.09])
    # ax_bogo.xaxis.set_major_locator(plt.MaxNLocator(2))
    # ax_bogo.set_ylim([0 - 0.09, 2 + 0.09])
    # ax_bogo.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax_bogo.legend(loc=2, fontsize=legendsize)

    # # # GROUND STATE ENERGY (SPHERICAL)

    # aIBi = -5
    # print(aIBi * xi)
    # qds_aIBi = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    # PVals = qds_aIBi['P'].values

    # CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)

    # Energy_Vals = np.zeros((PVals.size, tVals.size))
    # for Pind, P in enumerate(PVals):
    #     for tind, t in enumerate(tVals):
    #         CSAmp = CSAmp_ds.sel(P=P, t=t).values
    #         Energy_Vals[Pind, tind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)

    # Energy_Vals_inf = Energy_Vals[:, -1]
    # Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)

    # # Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 5 * PVals.size)
    # Pinf_Vals = np.linspace(0, np.max(PVals), 5 * PVals.size)
    # Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
    # Einf_1stderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=1)
    # Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)

    # sound_mask = np.abs(Einf_2ndderiv_Vals) <= 5e-3
    # Einf_sound = Einf_Vals[sound_mask]
    # Pinf_sound = Pinf_Vals[sound_mask]
    # [vsound, vs_const] = np.polyfit(Pinf_sound, Einf_sound, deg=1)

    # ms_mask = Pinf_Vals <= 0.5
    # Einf_1stderiv_ms = Einf_1stderiv_Vals[ms_mask]
    # Pinf_ms = Pinf_Vals[ms_mask]
    # [ms, ms_const] = np.polyfit(Pinf_ms, Einf_1stderiv_ms, deg=1)

    # mask = (Pinf_Vals / (mI * nu)) < 2.2
    # Ecrit = Einf_Vals[np.argmin(np.gradient(Einf_2ndderiv_Vals)) - 0]
    # ax_gsE.plot(Pinf_Vals[mask][1:-1] / (mI * nu), Einf_Vals[mask][1:-1] / np.abs(Ecrit), 'k-')
    # ax_gsE.set_xlabel(r'$P/(m_{I}c)$', fontsize=labelsize)
    # ax_gsE.set_ylabel(r'$E/E_{\rm crit}$', fontsize=labelsize)
    # # ymin = -2.1 / np.abs(Ecrit); ymax = -1 / np.abs(Ecrit)
    # ymin = -1.3; ymax = -0.6
    # # ax_gsE.set_ylim([ymin, ymax]); ax_gsE.set_xlim([0, 2.2])
    # ax_gsE.set_ylim([ymin - 0.02, ymax + 0.02]); ax_gsE.set_xlim([-0.05, 2.2 + 0.05])
    # ax_gsE.yaxis.set_major_locator(plt.MaxNLocator(2))

    # Pcrit = Pinf_Vals[np.argmin(np.gradient(Einf_2ndderiv_Vals)) - 0]
    # ax_gsE.axvline(x=Pcrit / (mI * nu), ymin=0.03, ymax=0.975, linestyle=':', color=green, lw=2)

    # ax_bogo.tick_params(direction='in', right=True, top=True)
    # ax_gsE.tick_params(direction='in', right=True, top=True)
    # ax_bogo.set_title('BEC without Impurity')
    # ax_gsE.set_title('BEC with Impurity')
    # subVals = np.linspace(0, Pcrit / (mI * nu), 100)
    # supVals = np.linspace(Pcrit / (mI * nu), np.max(Pinf_Vals[mask] / (mI * nu)), 100)
    # ax_gsE.fill_between(supVals, ymin, ymax, facecolor=base2, alpha=0.75)
    # ax_gsE.fill_between(subVals, ymin, ymax, facecolor=base02, alpha=0.3)

    # font = {'family': 'serif', 'color': 'black', 'size': legendsize}
    # sfont = {'family': 'serif', 'color': 'black', 'size': legendsize - 1}
    # ax_gsE.text(0.16, -1.3 / np.abs(Ecrit), 'Polaron', fontdict=font)
    # ax_gsE.text(0.16, -1.4 / np.abs(Ecrit), '(quadratic)', fontdict=sfont)
    # ax_gsE.text(1.3, -1.8 / np.abs(Ecrit), 'Cherenkov', fontdict=font)
    # ax_gsE.text(1.3, -1.9 / np.abs(Ecrit), '(linear)', fontdict=sfont)

    # # ax_gsE.margins(1.05, 1.05)

    # fig1.savefig(figdatapath + '/FigSchematic.pdf')

    # # # # FIG 1 - PHASE DIAGRAM + DISTRIBUTION PLOTS

    # matplotlib.rcParams['axes.linewidth'] = 0.5 * axl
    # matplotlib.rcParams.update({'font.size': 12})
    # labelsize = 13
    # legendsize = 12

    # fig1 = plt.figure(constrained_layout=False)
    # gs1 = fig1.add_gridspec(nrows=1, ncols=1, bottom=0.13, top=0.94, left=0.08, right=0.55)
    # gs2 = fig1.add_gridspec(nrows=2, ncols=1, bottom=0.13, top=0.94, left=0.67, right=0.99, height_ratios=[1, 1], hspace=0.2)

    # ax_PD = fig1.add_subplot(gs1[0])
    # ax_supDist = fig1.add_subplot(gs2[0])
    # ax_subDist = fig1.add_subplot(gs2[1])

    # fig1.text(0.01, 0.95, '(a)', fontsize=labelsize)
    # fig1.text(0.575, 0.95, '(b)', fontsize=labelsize)
    # fig1.text(0.575, 0.52, '(c)', fontsize=labelsize)
    # fig1.set_size_inches(7.8, 4.5)

    # # PHASE DIAGRAM (SPHERICAL)

    # Pcrit = np.zeros(aIBi_Vals.size)
    # ms_Vals = np.zeros(aIBi_Vals.size)
    # for aind, aIBi in enumerate(aIBi_Vals):
    #     qds_aIBi = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #     CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    #     kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)

    #     Energy_Vals_inf = np.zeros(PVals.size)
    #     for Pind, P in enumerate(PVals):
    #         CSAmp = CSAmp_ds.sel(P=P).isel(t=-1).values
    #         Energy_Vals_inf[Pind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)

    #     Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)
    #     Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 2 * PVals.size)
    #     Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
    #     Einf_1stderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=1)
    #     Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)
    #     # Pcrit[aind] = Pinf_Vals[np.argwhere(Einf_2ndderiv_Vals < 0)[-2][0] + 3]
    #     Pcrit[aind] = Pinf_Vals[np.argmin(np.gradient(Einf_2ndderiv_Vals)) - 0]  # there is a little bit of fudging with the -3 here so that aIBi=-10 gives me Pcrit/(mI*c) = 1 -> I can also just generate data for weaker interactions and see if it's better

    #     ms_mask = Pinf_Vals < 0.3
    #     Einf_1stderiv_ms = Einf_1stderiv_Vals[ms_mask]
    #     Pinf_ms = Pinf_Vals[ms_mask]
    #     [ms_Vals[aind], ms_const] = np.polyfit(Pinf_ms, Einf_1stderiv_ms, deg=1)

    # Pcrit_norm = Pcrit / (mI * nu)
    # Pcrit_tck = interpolate.splrep(aIBi_Vals, Pcrit_norm, s=0, k=3)
    # aIBi_interpVals = np.linspace(np.min(aIBi_Vals), np.max(aIBi_Vals), 5 * aIBi_Vals.size)
    # Pcrit_interpVals = 1 * interpolate.splev(aIBi_interpVals, Pcrit_tck, der=0)

    # print(Pcrit_norm)
    # print(Pcrit_norm[1], Pcrit_norm[5], Pcrit_norm[-5])

    # massEnhancement_Vals = (1 / ms_Vals) / mI
    # mE_tck = interpolate.splrep(aIBi_Vals, massEnhancement_Vals, s=0)
    # aIBi_interpVals = np.linspace(np.min(aIBi_Vals), np.max(aIBi_Vals), 5 * aIBi_Vals.size)
    # mE_interpVals = 1 * interpolate.splev(aIBi_interpVals, mE_tck, der=0)

    # # scalefac = 1.0
    # scalefac = 0.95  # just to align weakly interacting case slightly to 1 (it's pretty much there, would just need higher resolution data)
    # Pcrit_norm = scalefac * Pcrit_norm
    # Pcrit_interpVals = scalefac * Pcrit_interpVals

    # # xmin = np.min(aIBi_interpVals / xi); xmax = 1.01 * np.max(aIBi_interpVals / xi)
    # # ymin = 0; ymax = 1.01 * np.max(Pcrit_interpVals)

    # xmin = -11.45; xmax = 0.25
    # ymin = -0.1; ymax = 4.0

    # font = {'family': 'serif', 'color': 'black', 'size': legendsize}
    # sfont = {'family': 'serif', 'color': 'black', 'size': legendsize - 1}

    # ax_PD.plot(aIBi_Vals * xi, Pcrit_norm, marker='s', linestyle='None', mec='k', mfc='None', ms=5)
    # ax_PD.plot(aIBi_interpVals * xi, Pcrit_interpVals, 'k-')
    # # f1 = interpolate.interp1d(aIBi_Vals, Pcrit_norm, kind='cubic')
    # # ax_PD.plot(aIBi_interpVals, f1(aIBi_interpVals), 'k-')
    # ax_PD.set_xlabel(r'$a_{\rm IB}^{-1}/\xi^{-1}$', fontsize=labelsize)
    # ax_PD.set_ylabel(r'Total Momentum $P/(m_{I}c)$', fontsize=labelsize)
    # ax_PD.set_xlim([xmin, xmax]); ax_PD.set_ylim([ymin, ymax])
    # ax_PD.fill_between(aIBi_interpVals * xi, Pcrit_interpVals, ymax - 0.1, facecolor=base2, alpha=0.75)
    # ax_PD.fill_between(aIBi_interpVals * xi, ymin + 0.1, Pcrit_interpVals, facecolor=base02, alpha=0.3)
    # # ax_PD.text(-3.2, ymin + 0.155 * (ymax - ymin), 'Polaron', fontdict=font)
    # # ax_PD.text(-3.1, ymin + 0.08 * (ymax - ymin), '(' + r'$Z>0$' + ')', fontdict=sfont)
    # ax_PD.text(-10.5, ymin + 0.155 * (ymax - ymin), 'Polaron', fontdict=font)
    # ax_PD.text(-10.2, ymin + 0.08 * (ymax - ymin), r'$Z>0$', fontdict=sfont)
    # ax_PD.text(-10.5, ymin + 0.86 * (ymax - ymin), 'Cherenkov', fontdict=font)
    # ax_PD.text(-10.2, ymin + 0.785 * (ymax - ymin), r'$Z=0$', fontdict=sfont)

    # ax_PD.text(-5.7, ymin + 0.5 * (ymax - ymin), 'Dynamical', fontdict=font, color=red)
    # ax_PD.text(-5.6, ymin + 0.44 * (ymax - ymin), 'Transition', fontdict=font, color=red)

    # # # POLARON EFFECTIVE MASS (SPHERICAL)

    # # ax_PD.plot(aIBi_Vals * xi, massEnhancement_Vals, color='#ba9e88', marker='D', linestyle='None', markerfacecolor='None', mew=1, ms=5)
    # ax_PD.plot(aIBi_interpVals * xi, mE_interpVals, color='k', linestyle='dashed')

    # # CONNECTING LINES TO DISTRIBUTION FUNCTIONS

    # supDist_coords = [-5.0 * xi, 3.0]  # is [aIBi/xi, P/(mI*c)]
    # subDist_coords = [-5.0 * xi, 0.5]  # is [aIBi/xi, P/(mI*c)]

    # ax_PD.plot(supDist_coords[0], supDist_coords[1], linestyle='', marker='8', mec='k', mfc='k', ms=10)
    # ax_PD.plot(subDist_coords[0], subDist_coords[1], linestyle='', marker='8', mec='k', mfc='k', ms=10)

    # con_sup = ConnectionPatch(xyA=(supDist_coords[0], supDist_coords[1]), xyB=(0, 0.49), coordsA="data", coordsB="data", axesA=ax_PD, axesB=ax_supDist, color='k', linestyle='dotted', lw=0.5)
    # con_sub = ConnectionPatch(xyA=(subDist_coords[0], subDist_coords[1]), xyB=(0, 0.34), coordsA="data", coordsB="data", axesA=ax_PD, axesB=ax_subDist, color='k', linestyle='dotted', lw=0.5)
    # ax_PD.add_artist(con_sup)
    # ax_PD.add_artist(con_sub)

    # # IMPURITY DISTRIBUTION (CARTESIAN)

    # GaussianBroadening = True; sigma = 0.0168
    # incoh_color = green
    # delta_color = base02

    # def GPDF(xVals, mean, stdev):
    #     return (1 / (stdev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xVals - mean) / stdev)**2)
    #     # return (1 / (1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xVals - mean) / stdev)**2)

    # aIBi = -5
    # qds_aIBi = xr.open_dataset(innerdatapath_cart + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    # PVals = qds_aIBi['P'].values

    # nPIm_FWHM_indices = []
    # nPIm_distPeak_index = np.zeros(PVals.size, dtype=int)
    # nPIm_FWHM_Vals = np.zeros(PVals.size)
    # nPIm_distPeak_Vals = np.zeros(PVals.size)
    # nPIm_deltaPeak_Vals = np.zeros(PVals.size)
    # nPIm_Tot_Vals = np.zeros(PVals.size)
    # nPIm_Vec = np.empty(PVals.size, dtype=np.object)
    # PIm_Vec = np.empty(PVals.size, dtype=np.object)

    # for ind, P in enumerate(PVals):
    #     qds_nPIm_inf = qds_aIBi['nPI_mag'].sel(P=P).isel(t=-1).dropna('PI_mag')
    #     PIm_Vals = qds_nPIm_inf.coords['PI_mag'].values
    #     dPIm = PIm_Vals[1] - PIm_Vals[0]

    #     nPIm_Vec[ind] = qds_nPIm_inf.values
    #     PIm_Vec[ind] = PIm_Vals

    #     # # Calculate nPIm(t=inf) normalization
    #     nPIm_Tot_Vals[ind] = np.sum(qds_nPIm_inf.values * dPIm) + qds_aIBi.sel(P=P).isel(t=-1)['mom_deltapeak'].values

    #     # Calculate FWHM, distribution peak, and delta peak
    #     nPIm_FWHM_Vals[ind] = pfc.FWHM(PIm_Vals, qds_nPIm_inf.values)
    #     nPIm_distPeak_Vals[ind] = np.max(qds_nPIm_inf.values)
    #     nPIm_deltaPeak_Vals[ind] = qds_aIBi.sel(P=P).isel(t=-1)['mom_deltapeak'].values

    #     D = qds_nPIm_inf.values - np.max(qds_nPIm_inf.values) / 2
    #     indices = np.where(D > 0)[0]
    #     nPIm_FWHM_indices.append((indices[0], indices[-1]))
    #     nPIm_distPeak_index[ind] = np.argmax(qds_nPIm_inf.values)

    # Pnorm = PVals / (mI * nu)
    # Pratio_sup = 3.0; Pind_sup = np.abs(Pnorm - Pratio_sup).argmin()
    # Pratio_sub = 0.5; Pind_sub = np.abs(Pnorm - Pratio_sub).argmin()

    # print(Pnorm[Pind_sup], Pnorm[Pind_sub])
    # print(nPIm_deltaPeak_Vals[Pind_sup], nPIm_deltaPeak_Vals[Pind_sub])

    # ax_supDist.plot(PIm_Vec[Pind_sup] / (mI * nu), nPIm_Vec[Pind_sup], color=incoh_color, lw=1.0, label='Incoherent Part')
    # ax_supDist.set_xlim([-0.01, 5])
    # ax_supDist.set_ylim([0, 1.05])
    # ax_supDist.set_ylabel(r'$n_{|\mathbf{P}_{\rm imp}|}$', fontsize=labelsize)
    # # ax_supDist.set_xlabel(r'$|\vec{P_{I}}|/(m_{I}c)$', fontsize=labelsize)
    # ax_supDist.fill_between(PIm_Vec[Pind_sup] / (mI * nu), np.zeros(PIm_Vals.size), nPIm_Vec[Pind_sup], facecolor=incoh_color, alpha=0.25)
    # if GaussianBroadening:
    #     Pnorm_sup = PVals[Pind_sup] / (mI * nu)
    #     deltaPeak_sup = nPIm_deltaPeak_Vals[Pind_sup]
    #     PIm_norm_sup = PIm_Vec[Pind_sup] / (mI * nu)
    #     delta_GB_sup = deltaPeak_sup * GPDF(PIm_norm_sup, Pnorm_sup, sigma)
    #     # ax_supDist.plot(PIm_norm_sup, delta_GB_sup, linestyle='-', color=delta_color, linewidth=1, label=r'$\delta$-Peak')
    #     ax_supDist.plot(PIm_norm_sup, delta_GB_sup, linestyle='-', color=delta_color, linewidth=1.0, label='')
    #     ax_supDist.fill_between(PIm_norm_sup, np.zeros(PIm_norm_sup.size), delta_GB_sup, facecolor=delta_color, alpha=0.25)
    # else:
    #     ax_supDist.plot((PVals[Pind_sup] / (mI * nu)) * np.ones(PIm_Vals.size), np.linspace(0, nPIm_deltaPeak_Vals[Pind_sup], PIm_Vals.size), linestyle='-', color=delta_color, linewidth=1.5, label='Delta Peak (Z-factor)')
    # ax_supDist.legend(loc=1, fontsize=legendsize, frameon=False)

    # ax_subDist.plot(PIm_Vec[Pind_sub] / (mI * nu), nPIm_Vec[Pind_sub], color=incoh_color, lw=1.0, label='Incoherent Part')
    # # ax_subDist.set_xlim([-0.01, np.max(PIm_Vec[Pind_sub] / (mI*nu))])
    # ax_subDist.set_xlim([-0.01, 5])
    # ax_subDist.set_ylim([0, 1.05])
    # ax_subDist.set_ylabel(r'$n_{|\mathbf{P}_{\rm imp}|}$', fontsize=labelsize)
    # ax_subDist.set_xlabel(r'$|\mathbf{P}_{\rm imp}|/(m_{I}c)$', fontsize=labelsize)
    # ax_subDist.fill_between(PIm_Vec[Pind_sub] / (mI * nu), np.zeros(PIm_Vals.size), nPIm_Vec[Pind_sub], facecolor=incoh_color, alpha=0.25)
    # if GaussianBroadening:
    #     Pnorm_sub = PVals[Pind_sub] / (mI * nu)
    #     deltaPeak_sub = nPIm_deltaPeak_Vals[Pind_sub]
    #     PIm_norm_sub = PIm_Vec[Pind_sub] / (mI * nu)
    #     delta_GB_sub = deltaPeak_sub * GPDF(PIm_norm_sub, Pnorm_sub, sigma)
    #     ax_subDist.plot(PIm_norm_sub, delta_GB_sub, linestyle='-', color=delta_color, linewidth=1.0, label=r'$\delta$-Peak')
    #     ax_subDist.fill_between(PIm_norm_sub, np.zeros(PIm_norm_sub.size), delta_GB_sub, facecolor=delta_color, alpha=0.25)
    # else:
    #     ax_subDist.plot((PVals[Pind_sub] / (mI * nu)) * np.ones(PIm_Vals.size), np.linspace(0, nPIm_deltaPeak_Vals[Pind_sub], PIm_Vals.size), linestyle='-', color=delta_color, linewidth=1, label='Delta Peak (Z-factor)')
    # ax_subDist.legend(loc=1, fontsize=legendsize, frameon=False)

    # ax_PD.tick_params(direction='in', right=True, top=True)
    # ax_subDist.tick_params(direction='in', right=True, top=True)
    # ax_supDist.tick_params(direction='in', right=True, top=True)
    # ax_supDist.xaxis.set_ticklabels([])

    # # # # DPT

    # qds = xr.open_dataset('/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_1.11E+08_resRat_0.50/massRatio=1.0_noCSAmp/redyn_spherical' + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    # tVals = qds['t'].values

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

    # datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_1.11E+08_resRat_0.50/massRatio=1.0_noCSAmp'

    # Pcrit_da = xr.DataArray(np.full(aIBi_des.size, np.nan, dtype=float), coords=[aIBi_des], dims=['aIBi'])
    # for inda, aIBi in enumerate(aIBi_des):
    #     mds = xr.open_dataset(datapath + '/redyn_spherical/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #     Plen = mds.coords['P'].values.size
    #     Pstart_ind = 0
    #     PVals = mds.coords['P'].values[Pstart_ind:Plen]
    #     n0 = mds.attrs['n0']
    #     gBB = mds.attrs['gBB']
    #     mI = mds.attrs['mI']
    #     mB = mds.attrs['mB']
    #     nu = np.sqrt(n0 * gBB / mB)

    #     vI0_Vals = (PVals - mds.isel(t=0, P=np.arange(Pstart_ind, Plen))['Pph'].values) / mI

    #     mds_ts = mds.sel(t=tfVals)
    #     DynOv_Exponents = np.zeros(PVals.size)
    #     DynOv_Constants = np.zeros(PVals.size)

    #     for indP, P in enumerate(PVals):
    #         DynOv_raw = np.abs(mds_ts.isel(P=indP)['Real_DynOv'].values + 1j * mds_ts.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
    #         DynOv_ds = xr.DataArray(DynOv_raw, coords=[tfVals], dims=['t'])
    #         # DynOv_ds = DynOv_ds.rolling(t=rollwin, center=True).mean().dropna('t')
    #         DynOv_Vals = DynOv_ds.values
    #         tDynOvc_Vals = DynOv_ds['t'].values

    #         S_slope, S_intercept, S_rvalue, S_pvalue, S_stderr = ss.linregress(np.log(tDynOvc_Vals), np.log(DynOv_Vals))
    #         DynOv_Exponents[indP] = -1 * S_slope
    #         DynOv_Constants[indP] = np.exp(S_intercept)

    #     if DynOvExp_NegMask:
    #         DynOv_Exponents[DynOv_Exponents < 0] = 0

    #     if DynOvExp_Cut:
    #         DynOv_Exponents[np.abs(DynOv_Exponents) < cut] = 0

    #     if consecDetection:
    #         crit_ind = 0
    #         for indE, exp in enumerate(DynOv_Exponents):
    #             if indE > DynOv_Exponents.size - consecDetection:
    #                 break
    #             expSlice = DynOv_Exponents[indE:(indE + consecSamples)]
    #             if np.all(expSlice > 0):
    #                 crit_ind = indE
    #                 break
    #         DynOv_Exponents[0:crit_ind] = 0
    #     Pcrit_da[inda] = PVals[crit_ind] / (mI * nu)
    #     DynOvf_Vals = powerfunc(1e1000, DynOv_Exponents, DynOv_Constants)

    # ax_PD.plot(aIBi_des * xi, Pcrit_da.values, linestyle='None', marker='D', mec=red, mfc=red, mew=2, ms=5)

    # fig1.savefig(figdatapath + '/Fig1.pdf')

    # matplotlib.rcParams['axes.linewidth'] = axl

    # # # FIG 2 - PRL

    matplotlib.rcParams.update({'font.size': 12})
    labelsize = 13
    legendsize = 12

    fig2 = plt.figure(constrained_layout=False)
    gs1 = fig2.add_gridspec(nrows=2, ncols=1, bottom=0.23, top=0.95, left=0.12, right=0.48, hspace=0.1)
    gs2 = fig2.add_gridspec(nrows=2, ncols=1, bottom=0.23, top=0.95, left=0.61, right=0.98, hspace=0.1)

    ax_gsZ = fig2.add_subplot(gs1[0])
    ax_gsVel = fig2.add_subplot(gs1[1])
    ax_dynS = fig2.add_subplot(gs2[0])
    ax_dynVel = fig2.add_subplot(gs2[1])

    fig2.text(0.02, 0.95, '(a)', fontsize=labelsize)
    fig2.text(0.02, 0.55, '(b)', fontsize=labelsize)
    fig2.text(0.52, 0.95, '(c)', fontsize=labelsize)
    fig2.text(0.52, 0.55, '(d)', fontsize=labelsize)

    colorList = ['red', '#7e1e9c', 'green', 'orange', '#60460f', 'blue', 'magenta']
    lineList = ['solid', 'dashed', 'dotted', '-.']

    dyndatapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_1.11E+08_resRat_0.50/massRatio=1.0_noCSAmp/redyn_spherical'

    # ax_GSE1.set_ylim([0, 1.2 * np.max(Einf_1stderiv_Vals / np.abs(Ecrit))])

    # aIBi_des = np.array([-10.0, -5.0, -3.5, -2.0, -1.0])
    aIBi_Vals = np.array([-10.0, -5.0, -3.5, -2.0])  # used by many plots (spherical)

    # # POLARON SOUND VELOCITY (SPHERICAL)

    # Check to see if linear part of polaron (total system) energy spectrum has slope equal to sound velocity

    vsound_Vals = np.zeros(aIBi_Vals.size)
    vI_Vals = np.zeros(aIBi_Vals.size)
    for aind, aIBi in enumerate(aIBi_Vals):
        qds = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
        qds_aIBi = qds.isel(t=-1)
        ZVals = np.exp(-1 * qds_aIBi['Nph'].values)
        CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
        kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
        Energy_Vals_inf = np.zeros(PVals.size)
        PI_Vals = np.zeros(PVals.size)
        for Pind, P in enumerate(PVals):
            CSAmp = CSAmp_ds.sel(P=P).values
            Energy_Vals_inf[Pind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)
            PI_Vals[Pind] = P - qds_aIBi.sel(P=P)['Pph'].values

        Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)
        Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 2 * PVals.size)
        Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
        Einf_1stderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=1)
        Einf_1stderiv_Vals_subsamp = 1 * interpolate.splev(PVals, Einf_tck, der=1)
        xmask = (PVals / (mI * nu)) <= 4
        ax_gsZ.plot(PVals[xmask] / (mI * nu), ZVals[xmask], color=colorList[aind], linestyle='solid', marker='D', ms=4)
        # ax_gsVel.plot(Pinf_Vals / (mI * nu), Einf_1stderiv_Vals / nu, color=colorList[aind], linestyle='solid', marker='D', ms=4)
        ax_gsVel.plot(PVals[xmask] / (mI * nu), Einf_1stderiv_Vals_subsamp[xmask] / nu, color=colorList[aind], linestyle='solid', marker='D', ms=4)

    ax_gsVel.plot(Pinf_Vals / (mI * nu), np.ones(Pinf_Vals.size), 'k:')
    ax_gsVel.set_xlabel(r'$P/(m_{I}c)$', fontsize=14)
    ax_gsVel.set_ylabel(r'$v_{\rm pol}/c$', fontsize=14)
    ax_gsZ.set_ylabel(r'$Z$', fontsize=14)

    # DYN S(t) AND VELOCITY

    qds = xr.open_dataset('/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_1.11E+08_resRat_0.50/massRatio=1.0_noCSAmp/redyn_spherical' + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    tVals = qds['t'].values

    mc = mI * nu
    DynOvData_roll = False
    DynOvData_rollwin = 2
    PimpData_roll = False
    PimpData_rollwin = 2
    DynOvExp_roll = False
    DynOvExp_rollwin = 2
    DynOvExp_NegMask = False
    DynOvExp_Cut = False
    cut = 1e-4
    consecDetection = True
    consecSamples = 10
    flattenAboveC = True

    # aIBi_des = np.array([-10.0, -5.0, -3.5, -2.5, -2.0, -1.75])

    Pnorm = PVals / mc

    tmin = 90; tmax = 100

    tfVals = tVals[(tVals <= tmax) * (tVals >= tmin)]

    colorList = ['red', '#7e1e9c', 'green', 'orange', '#60460f', 'blue', 'magenta']
    lineList = ['solid', 'dashed', 'dotted', '-.']

    def powerfunc(t, a, b):
        return b * t**(-1 * a)

    Pcrit_da = xr.DataArray(np.full(aIBi_Vals.size, np.nan, dtype=float), coords=[aIBi_Vals], dims=['aIBi'])

    for inda, aIBi in enumerate(aIBi_Vals):
        qds_aIBi = xr.open_dataset(dyndatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
        # print(qds_aIBi['t'].values)
        qds_aIBi_ts = qds_aIBi.sel(t=tfVals)
        PVals = qds_aIBi['P'].values
        Pnorm = PVals / mc
        DynOv_Exponents = np.zeros(PVals.size)
        DynOv_Cov = np.full(PVals.size, np.nan)
        vImp_Exponents = np.zeros(PVals.size)
        vImp_Cov = np.full(PVals.size, np.nan)

        Plen = PVals.size
        Pstart_ind = 0
        vI0_Vals = (PVals - qds_aIBi.isel(t=0, P=np.arange(Pstart_ind, Plen))['Pph'].values) / mI

        DynOv_Exponents = np.zeros(PVals.size)
        DynOv_Constants = np.zeros(PVals.size)

        vImp_Exponents = np.zeros(PVals.size)
        vImp_Constants = np.zeros(PVals.size)

        DynOv_Rvalues = np.zeros(PVals.size)
        DynOv_Pvalues = np.zeros(PVals.size)
        DynOv_stderr = np.zeros(PVals.size)
        DynOv_tstat = np.zeros(PVals.size)
        DynOv_logAve = np.zeros(PVals.size)

        for indP, P in enumerate(PVals):
            DynOv_raw = np.abs(qds_aIBi_ts.isel(P=indP)['Real_DynOv'].values + 1j * qds_aIBi_ts.isel(P=indP)['Imag_DynOv'].values).real.astype(float)
            DynOv_ds = xr.DataArray(DynOv_raw, coords=[tfVals], dims=['t'])
            Pph_ds = xr.DataArray(qds_aIBi_ts.isel(P=indP)['Pph'].values, coords=[tfVals], dims=['t'])

            if DynOvData_roll:
                DynOv_ds = DynOv_ds.rolling(t=DynOvData_rollwin, center=True).mean().dropna('t')
            if PimpData_roll:
                Pph_ds = Pph_ds.rolling(t=PimpData_rollwin, center=True).mean().dropna('t')

            DynOv_Vals = DynOv_ds.values
            tDynOv_Vals = DynOv_ds['t'].values

            vImpc_Vals = (P - Pph_ds.values) / mI - nu
            tvImpc_Vals = Pph_ds['t'].values

            S_slope, S_intercept, S_rvalue, S_pvalue, S_stderr = ss.linregress(np.log(tDynOv_Vals), np.log(DynOv_Vals))
            DynOv_Exponents[indP] = -1 * S_slope
            DynOv_Constants[indP] = np.exp(S_intercept)

            DynOv_Rvalues[indP] = S_rvalue
            DynOv_Pvalues[indP] = S_pvalue
            DynOv_stderr[indP] = S_stderr
            DynOv_tstat[indP] = S_slope / S_stderr
            DynOv_logAve[indP] = np.average(np.log(DynOv_Vals))

            # if (-1 * S_slope) < 0:
            #     DynOv_Exponents[indP] = 0

            if vImpc_Vals[-1] < 0:
                vImp_Exponents[indP] = 0
                vImp_Constants[indP] = vImpc_Vals[-1]
            else:
                vI_slope, vI_intercept, vI_rvalue, vI_pvalue, vI_stderr = ss.linregress(np.log(tvImpc_Vals), np.log(vImpc_Vals))
                vImp_Exponents[indP] = -1 * vI_slope
                vImp_Constants[indP] = np.exp(vI_intercept)
                if (-1 * vI_slope) < 0:
                    vImp_Exponents[indP] = 0

        DynOvExponents_da = xr.DataArray(DynOv_Exponents, coords=[PVals], dims=['P'])
        if DynOvExp_roll:
            DynOvExponents_da = DynOvExponents_da.rolling(P=DynOvExp_rollwin, center=True).mean().dropna('P')
        if DynOvExp_NegMask:
            ExpMask = DynOvExponents_da.values < 0
            DynOvExponents_da[ExpMask] = 0
        if DynOvExp_Cut:
            ExpMask = np.abs(DynOvExponents_da.values) < cut
            DynOvExponents_da[ExpMask] = 0
        DynOv_Exponents = DynOvExponents_da.values
        if consecDetection:
            crit_ind = 0
            for indE, exp in enumerate(DynOv_Exponents):
                if indE > DynOv_Exponents.size - consecDetection:
                    break
                expSlice = DynOv_Exponents[indE:(indE + consecSamples)]
                if np.all(expSlice > 0):
                    crit_ind = indE
                    break
            DynOvExponents_da[0:crit_ind] = 0

        DynOv_Exponents = DynOvExponents_da.values
        Pnorm_dynov = DynOvExponents_da['P'].values / mc
        DynOvf_Vals = powerfunc(1e1000, DynOv_Exponents, DynOv_Constants)
        Pcrit_da[inda] = PVals[crit_ind] / (mI * nu)

        vIf_Vals = nu + powerfunc(1e1000, vImp_Exponents, vImp_Constants)
        if flattenAboveC:
            vIf_Vals[vIf_Vals > nu] = nu

        xmask = (vI0_Vals / nu) <= 4
        ax_dynS.plot(vI0_Vals[xmask] / nu, DynOvf_Vals[xmask], color=colorList[inda], linestyle='solid', marker='D', ms=4)
        ax_dynVel.plot(vI0_Vals[xmask] / nu, vIf_Vals[xmask] / nu, label='{:.2f}'.format(aIBi * xi), color=colorList[inda], linestyle='solid', marker='D', ms=4)

    ax_dynS.set_ylabel(r'$S(t_{\infty})$', fontsize=14)

    ax_dynVel.plot(vI0_Vals / nu, np.ones(vI0_Vals.size), 'k:')
    ax_dynVel.set_xlabel(r'$v_{\rm imp}(t_{0})/c$', fontsize=14)
    ax_dynVel.set_ylabel(r'$v_{\rm imp}(t_{\infty})/c$', fontsize=14)

    ax_dynS.xaxis.set_ticklabels([])

    ax_dynS.tick_params(which='both', direction='in', right=True, top=True)
    ax_dynVel.tick_params(which='both', direction='in', right=True, top=True)

    # GENERAL

    handles, labels = ax_dynVel.get_legend_handles_labels()
    plt.rcParams['legend.title_fontsize'] = 14
    fig2.legend(handles, labels, title=r'$a_{\rm IB}^{-1}/\xi^{-1}$', ncol=aIBi_Vals.size, loc='lower center', bbox_to_anchor=(0.55, 0.001), fontsize=13)

    ax_gsZ.xaxis.set_ticklabels([])
    ax_dynS.xaxis.set_ticklabels([])
    # ax_gsVel.set_xticks([0.0, 1.0, 2.0])
    ax_gsZ.tick_params(direction='in', right=True, top=True)
    ax_gsVel.tick_params(direction='in', right=True, top=True)
    ax_dynS.tick_params(direction='in', right=True, top=True)
    ax_dynVel.tick_params(direction='in', right=True, top=True)

    ax_gsZ.set_xlim([0, 4.14]); ax_gsZ.set_ylim([-0.05, 1.1])
    ax_gsVel.set_xlim([0, 4.14]); ax_gsVel.set_ylim([-0.05, 1.2])
    ax_dynS.set_xlim([-0.05, 4.14]); ax_dynS.set_ylim([-0.05, 1.1])
    ax_dynVel.set_xlim([-0.05, 4.14]); ax_dynVel.set_ylim([-0.05, 1.2])

    fig2.set_size_inches(7.8, 6.0)
    fig2.savefig(figdatapath + '/Fig2_PRL.pdf')

    # # # # FIG 1 (OLD) - POLARON GRAPHIC + BOGO DISPERSION + PHASE DIAGRAM + DISTRIBUTION PLOTS

    # matplotlib.rcParams.update({'font.size': 12})
    # labelsize = 13
    # legendsize = 12

    # fig1 = plt.figure(constrained_layout=False)
    # gs1 = fig1.add_gridspec(nrows=2, ncols=1, bottom=0.55, top=0.95, left=0.12, right=0.35, height_ratios=[1, 1])
    # gs2 = fig1.add_gridspec(nrows=1, ncols=1, bottom=0.55, top=0.95, left=0.5, right=0.98)
    # gs3 = fig1.add_gridspec(nrows=1, ncols=2, bottom=0.08, top=0.4, left=0.12, right=0.96, wspace=0.3)

    # ax_pol = fig1.add_subplot(gs1[0], frame_on=False); ax_pol.get_xaxis().set_visible(False); ax_pol.get_yaxis().set_visible(False)
    # ax_bogo = fig1.add_subplot(gs1[1])
    # ax_PD = fig1.add_subplot(gs2[0])
    # ax_supDist = fig1.add_subplot(gs3[0])
    # ax_subDist = fig1.add_subplot(gs3[1])

    # fig1.text(0.01, 0.97, '(a)', fontsize=labelsize)
    # fig1.text(0.01, 0.75, '(b)', fontsize=labelsize)
    # fig1.text(0.43, 0.97, '(c)', fontsize=labelsize)
    # fig1.text(0.01, 0.42, '(d)', fontsize=labelsize)
    # fig1.text(0.51, 0.42, '(e)', fontsize=labelsize)

    # # POLARON GRAPHIC

    # polimg = mpimg.imread('images/PolaronGraphic.png')
    # imgplot = ax_pol.imshow(polimg)

    # # BOGOLIUBOV DISPERSION (SPHERICAL)

    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds.coords['k'].values); kgrid.initArray_premade('th', qds.coords['th'].values)
    # kVals = kgrid.getArray('k')
    # wk_Vals = pfs.omegak(kVals, mB, n0, gBB)
    # ax_bogo.plot(kVals, wk_Vals, 'k-', label='')
    # ax_bogo.plot(kVals, nu * kVals, 'b--', label=r'$c|k|$')
    # ax_bogo.set_xlabel(r'$|k|$', fontsize=labelsize)
    # ax_bogo.set_ylabel(r'$\omega_{|k|}$', fontsize=labelsize)
    # ax_bogo.set_xlim([0, 2])
    # ax_bogo.xaxis.set_major_locator(plt.MaxNLocator(2))
    # ax_bogo.set_ylim([0, 3])
    # ax_bogo.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax_bogo.legend(loc=2, fontsize=legendsize)

    # # PHASE DIAGRAM (SPHERICAL)

    # Pcrit = np.zeros(aIBi_Vals.size)
    # for aind, aIBi in enumerate(aIBi_Vals):
    #     qds_aIBi = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #     CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    #     kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)

    #     Energy_Vals_inf = np.zeros(PVals.size)
    #     for Pind, P in enumerate(PVals):
    #         CSAmp = CSAmp_ds.sel(P=P).isel(t=-1).values
    #         Energy_Vals_inf[Pind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)

    #     Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)
    #     Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 2 * PVals.size)
    #     Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
    #     Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)
    #     # Pcrit[aind] = Pinf_Vals[np.argwhere(Einf_2ndderiv_Vals < 0)[-2][0] + 3]
    #     Pcrit[aind] = Pinf_Vals[np.argmin(np.gradient(Einf_2ndderiv_Vals)) - 0]  # there is a little bit of fudging with the -3 here so that aIBi=-10 gives me Pcrit/(mI*c) = 1 -> I can also just generate data for weaker interactions and see if it's better

    # Pcrit_norm = Pcrit / (mI * nu)
    # Pcrit_tck = interpolate.splrep(aIBi_Vals, Pcrit_norm, s=0, k=3)
    # aIBi_interpVals = np.linspace(np.min(aIBi_Vals), np.max(aIBi_Vals), 5 * aIBi_Vals.size)
    # Pcrit_interpVals = 1 * interpolate.splev(aIBi_interpVals, Pcrit_tck, der=0)

    # print(Pcrit_norm)
    # print(Pcrit_norm[1], Pcrit_norm[5], Pcrit_norm[-5])

    # scalefac = 1.0
    # # scalefac = 0.95  # just to align weakly interacting case slightly to 1 (it's pretty much there, would just need higher resolution data)
    # Pcrit_norm = scalefac * Pcrit_norm
    # Pcrit_interpVals = scalefac * Pcrit_interpVals

    # xmin = np.min(aIBi_interpVals / xi)
    # xmax = 1.01 * np.max(aIBi_interpVals / xi)
    # ymin = 0
    # ymax = 1.01 * np.max(Pcrit_interpVals)

    # font = {'family': 'serif', 'color': 'black', 'size': legendsize}
    # sfont = {'family': 'serif', 'color': 'black', 'size': legendsize - 1}

    # ax_PD.plot(aIBi_Vals / xi, Pcrit_norm, 'kx')
    # ax_PD.plot(aIBi_interpVals / xi, Pcrit_interpVals, 'k-')
    # # f1 = interpolate.interp1d(aIBi_Vals, Pcrit_norm, kind='cubic')
    # # ax_PD.plot(aIBi_interpVals, f1(aIBi_interpVals), 'k-')
    # ax_PD.set_xlabel(r'$a_{IB}^{-1}$ [$\xi$]', fontsize=labelsize)
    # ax_PD.set_ylabel(r'Total Momentum $P$ [$m_{I}c$]', fontsize=labelsize)
    # ax_PD.set_xlim([xmin, xmax])
    # ax_PD.set_ylim([ymin, ymax])
    # ax_PD.fill_between(aIBi_interpVals / xi, Pcrit_interpVals, ymax, facecolor='b', alpha=0.25)
    # ax_PD.fill_between(aIBi_interpVals / xi, ymin, Pcrit_interpVals, facecolor='g', alpha=0.25)
    # ax_PD.text(-3.2, ymin + 0.175 * (ymax - ymin), 'Polaron', fontdict=font)
    # ax_PD.text(-3.1, ymin + 0.1 * (ymax - ymin), '(' + r'$Z>0$' + ')', fontdict=sfont)
    # # ax_PD.text(-6.5, ymin + 0.6 * (ymax - ymin), 'Cherenkov', fontdict=font)
    # # ax_PD.text(-6.35, ymin + 0.525 * (ymax - ymin), '(' + r'$Z=0$' + ')', fontdict=sfont)
    # ax_PD.text(-12.8, ymin + 0.86 * (ymax - ymin), 'Cherenkov', fontdict=font)
    # ax_PD.text(-12.65, ymin + 0.785 * (ymax - ymin), '(' + r'$Z=0$' + ')', fontdict=sfont)

    # supDist_coords = [-5.0 / xi, 3.0]  # is [aIBi/xi, P/(mI*c)]
    # subDist_coords = [-5.0 / xi, 0.5]  # is [aIBi/xi, P/(mI*c)]

    # ax_PD.plot(supDist_coords[0], supDist_coords[1], linestyle='', marker='8', mec='#8f1402', mfc='#8f1402', ms=10)
    # ax_PD.plot(subDist_coords[0], subDist_coords[1], linestyle='', marker='8', mec='#8f1402', mfc='#8f1402', ms=10)

    # # IMPURITY DISTRIBUTION (CARTESIAN)

    # GaussianBroadening = True; sigma = 0.1
    # incoh_color = '#8f1402'
    # delta_color = '#bf9005'

    # def GPDF(xVals, mean, stdev):
    #     return (1 / (stdev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xVals - mean) / stdev)**2)
    #     # return (1 / (1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xVals - mean) / stdev)**2)

    # aIBi = -5
    # qds_aIBi = xr.open_dataset(innerdatapath_cart + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    # PVals = qds_aIBi['P'].values

    # nPIm_FWHM_indices = []
    # nPIm_distPeak_index = np.zeros(PVals.size, dtype=int)
    # nPIm_FWHM_Vals = np.zeros(PVals.size)
    # nPIm_distPeak_Vals = np.zeros(PVals.size)
    # nPIm_deltaPeak_Vals = np.zeros(PVals.size)
    # nPIm_Tot_Vals = np.zeros(PVals.size)
    # nPIm_Vec = np.empty(PVals.size, dtype=np.object)
    # PIm_Vec = np.empty(PVals.size, dtype=np.object)

    # for ind, P in enumerate(PVals):
    #     qds_nPIm_inf = qds_aIBi['nPI_mag'].sel(P=P).isel(t=-1).dropna('PI_mag')
    #     PIm_Vals = qds_nPIm_inf.coords['PI_mag'].values
    #     dPIm = PIm_Vals[1] - PIm_Vals[0]

    #     nPIm_Vec[ind] = qds_nPIm_inf.values
    #     PIm_Vec[ind] = PIm_Vals

    #     # # Calculate nPIm(t=inf) normalization
    #     nPIm_Tot_Vals[ind] = np.sum(qds_nPIm_inf.values * dPIm) + qds_aIBi.sel(P=P).isel(t=-1)['mom_deltapeak'].values

    #     # Calculate FWHM, distribution peak, and delta peak
    #     nPIm_FWHM_Vals[ind] = pfc.FWHM(PIm_Vals, qds_nPIm_inf.values)
    #     nPIm_distPeak_Vals[ind] = np.max(qds_nPIm_inf.values)
    #     nPIm_deltaPeak_Vals[ind] = qds_aIBi.sel(P=P).isel(t=-1)['mom_deltapeak'].values

    #     D = qds_nPIm_inf.values - np.max(qds_nPIm_inf.values) / 2
    #     indices = np.where(D > 0)[0]
    #     nPIm_FWHM_indices.append((indices[0], indices[-1]))
    #     nPIm_distPeak_index[ind] = np.argmax(qds_nPIm_inf.values)

    # Pnorm = PVals / (mI * nu)
    # Pratio_sup = 3.0; Pind_sup = np.abs(Pnorm - Pratio_sup).argmin()
    # Pratio_sub = 0.5; Pind_sub = np.abs(Pnorm - Pratio_sub).argmin()

    # print(Pnorm[Pind_sup], Pnorm[Pind_sub])
    # print(nPIm_deltaPeak_Vals[Pind_sup], nPIm_deltaPeak_Vals[Pind_sub])

    # ax_supDist.plot(PIm_Vec[Pind_sup] / (mI * nu), nPIm_Vec[Pind_sup], color=incoh_color, lw=0.5, label='Incoherent Part')
    # ax_supDist.set_xlim([-0.01, 10])
    # ax_supDist.set_ylim([0, 1.05])
    # ax_supDist.set_ylabel(r'$n_{|\vec{P_{I}}|}$', fontsize=labelsize)
    # ax_supDist.set_xlabel(r'$|\vec{P_{I}}|/(m_{I}c)$', fontsize=labelsize)
    # ax_supDist.fill_between(PIm_Vec[Pind_sup] / (mI * nu), np.zeros(PIm_Vals.size), nPIm_Vec[Pind_sup], facecolor=incoh_color, alpha=0.25)
    # if GaussianBroadening:
    #     Pnorm_sup = PVals[Pind_sup] / (mI * nu)
    #     deltaPeak_sup = nPIm_deltaPeak_Vals[Pind_sup]
    #     PIm_norm_sup = PIm_Vec[Pind_sup] / (mI * nu)
    #     delta_GB_sup = deltaPeak_sup * GPDF(PIm_norm_sup, Pnorm_sup, sigma)
    #     # ax_supDist.plot(PIm_norm_sup, delta_GB_sup, linestyle='-', color=delta_color, linewidth=1, label=r'$\delta$-Peak')
    #     ax_supDist.plot(PIm_norm_sup, delta_GB_sup, linestyle='-', color=delta_color, linewidth=1, label='')
    #     ax_supDist.fill_between(PIm_norm_sup, np.zeros(PIm_norm_sup.size), delta_GB_sup, facecolor=delta_color, alpha=0.25)
    # else:
    #     ax_supDist.plot((PVals[Pind_sup] / (mI * nu)) * np.ones(PIm_Vals.size), np.linspace(0, nPIm_deltaPeak_Vals[Pind_sup], PIm_Vals.size), linestyle='-', color=delta_color, linewidth=1, label='Delta Peak (Z-factor)')
    # ax_supDist.legend(loc=1, fontsize=legendsize)

    # ax_subDist.plot(PIm_Vec[Pind_sub] / (mI * nu), nPIm_Vec[Pind_sub], color=incoh_color, lw=0.5, label='Incoherent Part')
    # # ax_subDist.set_xlim([-0.01, np.max(PIm_Vec[Pind_sub] / (mI*nu))])
    # ax_subDist.set_xlim([-0.01, 10])
    # ax_subDist.set_ylim([0, 1.05])
    # ax_subDist.set_ylabel(r'$n_{|\vec{P_{I}}|}$', fontsize=labelsize)
    # ax_subDist.set_xlabel(r'$|\vec{P_{I}}|/(m_{I}c)$', fontsize=labelsize)
    # ax_subDist.fill_between(PIm_Vec[Pind_sub] / (mI * nu), np.zeros(PIm_Vals.size), nPIm_Vec[Pind_sub], facecolor=incoh_color, alpha=0.25)
    # if GaussianBroadening:
    #     Pnorm_sub = PVals[Pind_sub] / (mI * nu)
    #     deltaPeak_sub = nPIm_deltaPeak_Vals[Pind_sub]
    #     PIm_norm_sub = PIm_Vec[Pind_sub] / (mI * nu)
    #     delta_GB_sub = deltaPeak_sub * GPDF(PIm_norm_sub, Pnorm_sub, sigma)
    #     ax_subDist.plot(PIm_norm_sub, delta_GB_sub, linestyle='-', color=delta_color, linewidth=1, label=r'$\delta$-Peak')
    #     ax_subDist.fill_between(PIm_norm_sub, np.zeros(PIm_norm_sub.size), delta_GB_sub, facecolor=delta_color, alpha=0.25)
    # else:
    #     ax_subDist.plot((PVals[Pind_sub] / (mI * nu)) * np.ones(PIm_Vals.size), np.linspace(0, nPIm_deltaPeak_Vals[Pind_sub], PIm_Vals.size), linestyle='-', color=delta_color, linewidth=1, label='Delta Peak (Z-factor)')
    # ax_subDist.legend(loc=1, fontsize=legendsize)

    # fig1.set_size_inches(7.8, 9)
    # fig1.savefig(figdatapath + '/Fig1.pdf')

    # # # # FIG 2 - ENERGY DERIVATIVES + SOUND VELOCITY + EFFECTIVE MASS

    # matplotlib.rcParams.update({'font.size': 12})
    # labelsize = 13
    # legendsize = 12

    # fig2 = plt.figure(constrained_layout=False)
    # # gs1 = fig2.add_gridspec(nrows=3, ncols=1, bottom=0.12, top=0.925, left=0.12, right=0.40, hspace=1.0)
    # # gs2 = fig2.add_gridspec(nrows=2, ncols=1, bottom=0.12, top=0.925, left=0.58, right=0.98, hspace=0.7)
    # gs1 = fig2.add_gridspec(nrows=3, ncols=1, bottom=0.12, top=0.95, left=0.12, right=0.40, hspace=0.2)
    # gs2 = fig2.add_gridspec(nrows=2, ncols=1, bottom=0.12, top=0.95, left=0.58, right=0.98, hspace=0.1)

    # ax_GSE0 = fig2.add_subplot(gs1[0])
    # ax_GSE1 = fig2.add_subplot(gs1[1])
    # ax_GSE2 = fig2.add_subplot(gs1[2])
    # ax_Vel = fig2.add_subplot(gs2[0])
    # ax_Mass = fig2.add_subplot(gs2[1])

    # # fig2.text(0.01, 0.95, '(a)', fontsize=labelsize)
    # # fig2.text(0.01, 0.65, '(b)', fontsize=labelsize)
    # # fig2.text(0.01, 0.32, '(c)', fontsize=labelsize)
    # # fig2.text(0.47, 0.95, '(d)', fontsize=labelsize)
    # # fig2.text(0.47, 0.47, '(e)', fontsize=labelsize)

    # fig2.text(0.01, 0.95, '(a)', fontsize=labelsize)
    # fig2.text(0.01, 0.65, '(b)', fontsize=labelsize)
    # fig2.text(0.01, 0.37, '(c)', fontsize=labelsize)
    # fig2.text(0.47, 0.95, '(d)', fontsize=labelsize)
    # fig2.text(0.47, 0.52, '(e)', fontsize=labelsize)

    # # # ENERGY DERIVATIVES (SPHERICAL)

    # aIBi = -5
    # qds_aIBi = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    # PVals = qds_aIBi['P'].values
    # print(aIBi * xi)

    # CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)

    # Energy_Vals = np.zeros((PVals.size, tVals.size))
    # for Pind, P in enumerate(PVals):
    #     for tind, t in enumerate(tVals):
    #         CSAmp = CSAmp_ds.sel(P=P, t=t).values
    #         Energy_Vals[Pind, tind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)

    # Energy_Vals_inf = Energy_Vals[:, -1]
    # Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)

    # Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 5 * PVals.size)
    # Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
    # Einf_1stderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=1)
    # Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)

    # sound_mask = np.abs(Einf_2ndderiv_Vals) <= 5e-3
    # Einf_sound = Einf_Vals[sound_mask]
    # Pinf_sound = Pinf_Vals[sound_mask]
    # [vsound, vs_const] = np.polyfit(Pinf_sound, Einf_sound, deg=1)

    # ms_mask = Pinf_Vals <= 0.5
    # Einf_1stderiv_ms = Einf_1stderiv_Vals[ms_mask]
    # Pinf_ms = Pinf_Vals[ms_mask]
    # [ms, ms_const] = np.polyfit(Pinf_ms, Einf_1stderiv_ms, deg=1)

    # Ecrit = Einf_Vals[np.argmin(np.gradient(Einf_2ndderiv_Vals))]
    # # ax_GSE0.plot(Pinf_Vals / (mI * nu), Einf_Vals / np.abs(Ecrit), 'k-', lw=1.5)
    # ax_GSE0.plot(Pinf_Vals[::2] / (mI * nu), Einf_Vals[::2] / np.abs(Ecrit), 'ko', ms=6)
    # # ax_GSE0.set_title('Ground State Energy (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # # ax_GSE0.set_xlabel(r'$P$ [$m_{I}c$]', fontsize=labelsize)
    # ax_GSE0.set_ylabel(r'$E$', fontsize=labelsize)
    # ax_GSE0.set_ylim([1.1 * np.min(Einf_Vals / np.abs(Ecrit)), -0.5 / np.abs(Ecrit)])
    # ax_GSE0.set_xlim([0, 2.0])

    # # ax_GSE1.plot(Pinf_Vals / (mI * nu), Einf_1stderiv_Vals / np.abs(Ecrit), 'k-', lw=1.5)
    # ax_GSE1.plot(Pinf_Vals[::2] / (mI * nu), Einf_1stderiv_Vals[::2] / np.abs(Ecrit), 'ko', ms=6)
    # # ax_GSE1.set_title('First Derivative of Energy')
    # # ax_GSE1.set_xlabel(r'$P$ [$m_{I}c$]', fontsize=labelsize)
    # ax_GSE1.set_ylabel(r'$dE/dP$', fontsize=labelsize)
    # ax_GSE1.plot(Pinf_Vals / (mI * nu), vsound * np.ones(Pinf_Vals.size) / np.abs(Ecrit), color=red, linestyle='--', linewidth=2.0)
    # ax_GSE1.set_ylim([0, 1.2 * np.max(Einf_1stderiv_Vals / np.abs(Ecrit))])
    # ax_GSE1.set_xlim([0, 2.0])

    # # ax_GSE2.plot(Pinf_Vals / (mI * nu), Einf_2ndderiv_Vals / np.abs(Ecrit), 'k-', lw=1.5)
    # ax_GSE2.plot(Pinf_Vals[::2] / (mI * nu), Einf_2ndderiv_Vals[::2] / np.abs(Ecrit), 'ko', ms=6)
    # # ax_GSE2.set_title('Second Derivative of Energy')
    # ax_GSE2.set_xlabel(r'$P/(m_{I}c)$', fontsize=labelsize)
    # ax_GSE2.set_ylabel(r'$d^{2}E/dP^{2}$', fontsize=labelsize)
    # ax_GSE2.plot(Pinf_Vals / (mI * nu), ms * np.ones(Pinf_Vals.size) / np.abs(Ecrit), color=blue, linestyle='--', linewidth=2.0)
    # ax_GSE2.set_ylim([-.12, 1.2 * np.max(Einf_2ndderiv_Vals / np.abs(Ecrit))])
    # ax_GSE2.set_xlim([0, 2.0])

    # # including a Pcrit line
    # Pcrit = Pinf_Vals[np.argmin(np.gradient(Einf_2ndderiv_Vals)) - 0]
    # # Pcrit_2 = Pinf_Vals[sound_mask][0]; print(Pcrit, Pcrit_2)
    # ax_GSE0.axvline(x=Pcrit / (mI * nu), linestyle=':', color=green, lw=2)
    # ax_GSE1.axvline(x=Pcrit / (mI * nu), linestyle=':', color=green, lw=2)
    # ax_GSE2.axvline(x=Pcrit / (mI * nu), linestyle=':', color=green, lw=2)

    # # # POLARON SOUND VELOCITY (SPHERICAL)

    # # Check to see if linear part of polaron (total system) energy spectrum has slope equal to sound velocity

    # vsound_Vals = np.zeros(aIBi_Vals.size)
    # vI_Vals = np.zeros(aIBi_Vals.size)
    # for aind, aIBi in enumerate(aIBi_Vals):
    #     qds = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #     qds_aIBi = qds.isel(t=-1)
    #     CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    #     kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    #     Energy_Vals_inf = np.zeros(PVals.size)
    #     PI_Vals = np.zeros(PVals.size)
    #     for Pind, P in enumerate(PVals):
    #         CSAmp = CSAmp_ds.sel(P=P).values
    #         Energy_Vals_inf[Pind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)
    #         PI_Vals[Pind] = P - qds_aIBi.sel(P=P)['Pph'].values

    #     Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)
    #     Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 2 * PVals.size)
    #     Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
    #     Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)

    #     sound_mask = np.abs(Einf_2ndderiv_Vals) <= 5e-3
    #     Einf_sound = Einf_Vals[sound_mask]
    #     Pinf_sound = Pinf_Vals[sound_mask]
    #     [vsound_Vals[aind], vs_const] = np.polyfit(Pinf_sound, Einf_sound, deg=1)

    #     vI_inf_tck = interpolate.splrep(PVals, PI_Vals / mI, s=0)
    #     vI_inf_Vals = 1 * interpolate.splev(Pinf_Vals, vI_inf_tck, der=0)
    #     vI_Vals[aind] = np.polyfit(Pinf_sound, vI_inf_Vals[sound_mask], deg=0)

    # print(vsound_Vals)
    # print(100 * (vsound_Vals - nu) / nu)
    # ax_Vel.plot(aIBi_Vals * xi, vsound_Vals / nu, linestyle='None', mec=red, mfc=red, marker='x', mew=1, ms=10, label='Polaron')
    # ax_Vel.plot(aIBi_Vals * xi, vI_Vals / nu, 'ko', mew=1, ms=10, markerfacecolor='none', label='Impurity')
    # ax_Vel.plot(aIBi_Vals * xi, np.ones(aIBi_Vals.size), color='grey', linestyle='dashdot', linewidth=2.0, label='$c$')
    # ax_Vel.set_ylim([0.5, 1.25])
    # # ax_Vel.set_ylim([0.8, 1.25])
    # ax_Vel.legend(loc=(0.25, 0.1), fontsize=legendsize)
    # # ax_Vel.set_xlabel(r'$a_{IB}^{-1}$ [$\xi$]', fontsize=labelsize)
    # ax_Vel.set_ylabel(r'Velocity', fontsize=labelsize)

    # # # POLARON EFFECTIVE MASS (SPHERICAL)

    # ms_Vals = np.zeros(aIBi_Vals.size)
    # for aind, aIBi in enumerate(aIBi_Vals):
    #     qds = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #     qds_aIBi = qds.isel(t=-1)
    #     CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    #     kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    #     Energy_Vals_inf = np.zeros(PVals.size)
    #     PI_Vals = np.zeros(PVals.size)
    #     for Pind, P in enumerate(PVals):
    #         CSAmp = CSAmp_ds.sel(P=P).values
    #         Energy_Vals_inf[Pind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)
    #         PI_Vals[Pind] = P - qds_aIBi.sel(P=P)['Pph'].values

    #     Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)
    #     Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 2 * PVals.size)
    #     Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
    #     Einf_1stderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=1)
    #     Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)

    #     ms_mask = Pinf_Vals < 0.3
    #     Einf_1stderiv_ms = Einf_1stderiv_Vals[ms_mask]
    #     Pinf_ms = Pinf_Vals[ms_mask]
    #     [ms_Vals[aind], ms_const] = np.polyfit(Pinf_ms, Einf_1stderiv_ms, deg=1)

    # massEnhancement_Vals = (1 / ms_Vals) / mI

    # mE_tck = interpolate.splrep(aIBi_Vals, massEnhancement_Vals, s=0)
    # aIBi_interpVals = np.linspace(np.min(aIBi_Vals), np.max(aIBi_Vals), 5 * aIBi_Vals.size)
    # mE_interpVals = 1 * interpolate.splev(aIBi_interpVals, mE_tck, der=0)

    # ax_Mass.plot(aIBi_Vals * xi, massEnhancement_Vals, linestyle='None', marker='D', mec=blue, mfc=blue, mew=1, ms=5)
    # ax_Mass.plot(aIBi_interpVals * xi, mE_interpVals, color=blue, linestyle='-')
    # ax_Mass.set_xlabel(r'$a_{\rm IB}^{-1}/\xi^{-1}$', fontsize=labelsize)
    # # ax_Mass.set_ylabel(r'$\frac{m^{*}}{m_{I}} = \frac{1}{m_{I}}\frac{\partial^{2} E}{\partial P^{2}}$')
    # ax_Mass.set_ylabel(r'Effective Mass', fontsize=labelsize)

    # ax_GSE0.xaxis.set_ticklabels([])
    # ax_GSE1.xaxis.set_ticklabels([])
    # ax_Vel.xaxis.set_ticklabels([])
    # ax_GSE0.set_xticks([0.0, 1.0, 2.0])
    # ax_GSE1.set_xticks([0.0, 1.0, 2.0])
    # ax_GSE2.set_xticks([0.0, 1.0, 2.0])
    # ax_GSE0.tick_params(direction='in', right=True, top=True)
    # ax_GSE1.tick_params(direction='in', right=True, top=True)
    # ax_GSE2.tick_params(direction='in', right=True, top=True)
    # ax_Vel.tick_params(direction='in', right=True, top=True)
    # ax_Mass.tick_params(direction='in', right=True, top=True)
    # vel_coords = [2, vsound / np.abs(Ecrit)]
    # effM_coords = [2, ms / np.abs(Ecrit)]
    # con_vel = ConnectionPatch(xyA=(vel_coords[0], vel_coords[1]), xyB=(-11, 1.0), coordsA="data", coordsB="data", axesA=ax_GSE1, axesB=ax_Vel, color=red, linestyle='dashed', lw=0.5)
    # con_effM = ConnectionPatch(xyA=(effM_coords[0], effM_coords[1]), xyB=(-11, 1.92), coordsA="data", coordsB="data", axesA=ax_GSE2, axesB=ax_Mass, color=blue, linestyle='dashed', lw=0.5)
    # ax_GSE1.add_artist(con_vel)
    # ax_GSE2.add_artist(con_effM)

    # fig2.set_size_inches(7.8, 5.0)
    # fig2.savefig(figdatapath + '/Fig2.pdf')

    # # FIG 3 - IMPURITY DISTRIBUTION WITH CHARACTERIZATION (CARTESIAN)

    # matplotlib.rcParams.update({'font.size': 12})
    # labelsize = 13
    # legendsize = 12

    # GaussianBroadening = True; sigma = 0.1
    # incoh_color = green
    # delta_color = base02
    # fwhm_color = red

    # def GPDF(xVals, mean, stdev):
    #     return (1 / (stdev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xVals - mean) / stdev)**2)
    #     # return (1 / (1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xVals - mean) / stdev)**2)

    # aIBi = -2
    # print('int: {0}'.format(aIBi * xi))
    # qds_aIBi = xr.open_dataset(innerdatapath_cart + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    # PVals = qds_aIBi['P'].values

    # nPIm_FWHM_indices = []
    # nPIm_distPeak_index = np.zeros(PVals.size, dtype=int)
    # nPIm_FWHM_Vals = np.zeros(PVals.size)
    # nPIm_distPeak_Vals = np.zeros(PVals.size)
    # nPIm_deltaPeak_Vals = np.zeros(PVals.size)
    # nPIm_Tot_Vals = np.zeros(PVals.size)
    # nPIm_Vec = np.empty(PVals.size, dtype=np.object)
    # PIm_Vec = np.empty(PVals.size, dtype=np.object)

    # for ind, P in enumerate(PVals):
    #     qds_nPIm_inf = qds_aIBi['nPI_mag'].sel(P=P).isel(t=-1).dropna('PI_mag')
    #     PIm_Vals = qds_nPIm_inf.coords['PI_mag'].values
    #     dPIm = PIm_Vals[1] - PIm_Vals[0]

    #     # # Plot nPIm(t=inf)
    #     # qds_nPIm_inf.plot(ax=ax, label='P: {:.1f}'.format(P))
    #     nPIm_Vec[ind] = qds_nPIm_inf.values
    #     PIm_Vec[ind] = PIm_Vals

    #     # # Calculate nPIm(t=inf) normalization
    #     nPIm_Tot_Vals[ind] = np.sum(qds_nPIm_inf.values * dPIm) + qds_aIBi.sel(P=P).isel(t=-1)['mom_deltapeak'].values

    #     # Calculate FWHM, distribution peak, and delta peak
    #     nPIm_FWHM_Vals[ind] = pfc.FWHM(PIm_Vals, qds_nPIm_inf.values)
    #     nPIm_distPeak_Vals[ind] = np.max(qds_nPIm_inf.values)
    #     nPIm_deltaPeak_Vals[ind] = qds_aIBi.sel(P=P).isel(t=-1)['mom_deltapeak'].values

    #     D = qds_nPIm_inf.values - np.max(qds_nPIm_inf.values) / 2
    #     indices = np.where(D > 0)[0]
    #     nPIm_FWHM_indices.append((indices[0], indices[-1]))
    #     nPIm_distPeak_index[ind] = np.argmax(qds_nPIm_inf.values)

    # Pratio = 1.4
    # Pnorm = PVals / (mI * nu)
    # Pind = np.abs(Pnorm - Pratio).argmin()
    # print(Pnorm[Pind], aIBi / xi)
    # print(nPIm_deltaPeak_Vals[Pind])

    # fig3, axes3 = plt.subplots(nrows=1, ncols=3)
    # ind_s, ind_f = nPIm_FWHM_indices[Pind]
    # ind_f = ind_f - 1  # this is just to make the FWHM marker on the plot look a little cleaner

    # axes3[0].plot(PIm_Vec[Pind] / (mI * nu), nPIm_Vec[Pind], color=incoh_color, lw=1.0, label='Incoherent Part')
    # axes3[0].set_xlim([-0.01, 10])
    # axes3[0].set_ylim([0, 1.05])
    # axes3[0].set_ylabel(r'$n_{|\mathbf{P}_{\rm imp}|}$', fontsize=labelsize)
    # axes3[0].set_xlabel(r'$|\mathbf{P}_{\rm imp}|/(m_{I}c)$', fontsize=labelsize)
    # axes3[0].fill_between(PIm_Vec[Pind] / (mI * nu), np.zeros(PIm_Vals.size), nPIm_Vec[Pind], facecolor=incoh_color, alpha=0.25)
    # if GaussianBroadening:
    #     Pnorm = PVals[Pind] / (mI * nu)
    #     deltaPeak = nPIm_deltaPeak_Vals[Pind]
    #     PIm_norm = PIm_Vec[Pind] / (mI * nu)
    #     delta_GB = deltaPeak * GPDF(PIm_norm, Pnorm, sigma)
    #     axes3[0].plot(PIm_norm, delta_GB, linestyle='-', color=delta_color, linewidth=1, label='Delta Peak')
    #     axes3[0].fill_between(PIm_norm, np.zeros(PIm_norm.size), delta_GB, facecolor=delta_color, alpha=0.25)
    # else:
    #     axes3[0].plot((PVals[Pind] / (mI * nu)) * np.ones(PIm_Vals.size), np.linspace(0, nPIm_deltaPeak_Vals[Pind], PIm_Vals.size), linestyle='-', color=delta_color, linewidth=1, label='Delta Peak Weight (Z-factor)')
    # # axes3[0].legend(loc=1, fontsize=legendsize)

    # axes3[0].plot(np.linspace(PIm_Vec[Pind][ind_s] / (mI * nu), PIm_Vec[Pind][ind_f] / (mI * nu), 100), nPIm_Vec[Pind][ind_s] * np.ones(100), linestyle='-', color=fwhm_color, linewidth=2.0, label='Incoherent Part FWHM')
    # axes3[0].plot(np.linspace(PIm_Vec[Pind][ind_s] / (mI * nu), PIm_Vec[Pind][ind_f] / (mI * nu), 2), nPIm_Vec[Pind][ind_s] * np.ones(2), marker='D', color=fwhm_color, mew=0.5, ms=4, label='')

    # axes3[1].plot(PVals / (mI * nu), nPIm_deltaPeak_Vals, linestyle='-', color=delta_color)
    # axes3[1].set_xlabel(r'$P/(m_{I}c)$', fontsize=labelsize)
    # axes3[1].set_ylabel(r'Quasiparticle Residue ($Z$)', fontsize=labelsize)

    # axes3[2].plot(PVals / (mI * nu), nPIm_FWHM_Vals, linestyle='-', color=fwhm_color)
    # axes3[2].set_xlabel(r'$P/(m_{I}c)$', fontsize=labelsize)
    # axes3[2].set_ylabel('Incoherent Part FWHM', fontsize=labelsize)
    # axes3[2].set_ylim([0.5, 2.5])
    # axes3[2].yaxis.set_major_locator(plt.MaxNLocator(4))

    # axes3[0].tick_params(direction='in', right=True, top=True)
    # axes3[1].tick_params(direction='in', right=True, top=True)
    # axes3[2].tick_params(direction='in', right=True, top=True)

    # fig3.text(0.01, 0.95, '(a)', fontsize=labelsize)
    # fig3.text(0.33, 0.95, '(b)', fontsize=labelsize)
    # fig3.text(0.66, 0.95, '(c)', fontsize=labelsize)

    # fig3.subplots_adjust(left=0.1, bottom=0.17, top=0.91, right=0.98, wspace=0.6)
    # fig3.set_size_inches(7.8, 3.5)
    # fig3.savefig(figdatapath + '/Fig3.pdf')

    # # plt.show()
