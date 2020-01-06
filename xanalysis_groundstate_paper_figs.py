import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from matplotlib.animation import writers
import matplotlib.image as mpimg
import os
import itertools
import pf_dynamic_cart as pfc
import pf_dynamic_sph as pfs
import Grid
from scipy import interpolate
from timeit import default_timer as timer


if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})
    mpegWriter = writers['ffmpeg'](fps=0.75, bitrate=1800)
    matplotlib.rcParams.update({'font.size': 16})

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

    figdatapath = '/Users/kis/Dropbox/Apps/Overleaf/Quantum Cherenkov Transition in Bose Polaron Systems/figures/figdump'
    innerdatapath_cart = innerdatapath[0:-10] + '_cart'

    # # Analysis of Total Dataset

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

    # # # # FIG 1 - POLARON GRAPHIC + BOGO DISPERSION + PHASE DIAGRAM + DISTRIBUTION PLOTS

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
    #     ax_supDist.plot(PIm_norm_sup, delta_GB_sup, linestyle='-', color=delta_color, linewidth=1, label=r'$\delta$-Peak')
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
    # # fig1.savefig(figdatapath + '/Fig1.pdf')

    # # # FIG 2 - ENERGY DERIVATIVES + SOUND VELOCITY + EFFECTIVE MASS

    # # ENERGY DERIVATIVES

    aIBi = -5
    qds_aIBi = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    PVals = qds_aIBi['P'].values

    CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)

    Energy_Vals = np.zeros((PVals.size, tVals.size))
    for Pind, P in enumerate(PVals):
        for tind, t in enumerate(tVals):
            CSAmp = CSAmp_ds.sel(P=P, t=t).values
            Energy_Vals[Pind, tind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)

    Energy_Vals_inf = Energy_Vals[:, -1]
    Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)

    Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 5 * PVals.size)
    Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
    Einf_1stderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=1)
    Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)

    sound_mask = np.abs(Einf_2ndderiv_Vals) <= 5e-3
    Einf_sound = Einf_Vals[sound_mask]
    Pinf_sound = Pinf_Vals[sound_mask]
    [vsound, vs_const] = np.polyfit(Pinf_sound, Einf_sound, deg=1)

    ms_mask = Pinf_Vals <= 0.5
    Einf_1stderiv_ms = Einf_1stderiv_Vals[ms_mask]
    Pinf_ms = Pinf_Vals[ms_mask]
    [ms, ms_const] = np.polyfit(Pinf_ms, Einf_1stderiv_ms, deg=1)

    fig, axes = plt.subplots(nrows=3, ncols=1)
    axes[0].plot(Pinf_Vals, Einf_Vals, 'k-')
    axes[0].set_title('Ground State Energy (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    axes[0].set_xlabel('P')
    axes[0].set_ylim([1.1 * np.min(Einf_Vals), -0.5])
    axes[0].set_xlim([0, 2.0])

    axes[1].plot(Pinf_Vals, Einf_1stderiv_Vals, 'k-')
    axes[1].set_title('First Derivative of Energy')
    axes[1].set_xlabel('P')
    axes[1].plot(Pinf_Vals, vsound * np.ones(Pinf_Vals.size), 'r--', linewidth=2.0)
    axes[1].set_ylim([0, 1.2 * np.max(Einf_1stderiv_Vals)])
    axes[1].set_xlim([0, 2.0])

    axes[2].plot(Pinf_Vals[::2], Einf_2ndderiv_Vals[::2], 'ko')
    axes[2].set_title('Second Derivative of Energy')
    axes[2].set_xlabel('P')
    axes[2].plot(Pinf_Vals, ms * np.ones(Pinf_Vals.size), 'c--', linewidth=2.0)
    axes[2].set_ylim([0, 1.2 * np.max(Einf_2ndderiv_Vals)])
    axes[2].set_xlim([0, 2.0])

    # # This plot below is for saturation/convergence of the energy with imaginary time
    # fig3, ax3 = plt.subplots()
    # Pind = 8
    # ax3.plot(tVals, np.abs(Energy_Vals[Pind, :]), 'k-')
    # ax3.set_yscale('log')
    # ax3.set_xscale('log')
    # ax3.set_title('Ground State Energy (' + r'$a_{IB}^{-1}=$' + '{0}, '.format(aIBi) + r'$P=$' + '{:.2f})'.format(PVals[Pind]))
    # ax3.set_xlabel('Imaginary time')

    fig.tight_layout()

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
    # fig, ax = plt.subplots()
    # ax.plot(aIBi_Vals, vsound_Vals, 'rx', mew=1, ms=10, label='Post-Transition Polaron Sound Velocity (' + r'$\frac{\partial E}{\partial P}$' + ')')
    # ax.plot(aIBi_Vals, vI_Vals, 'ko', mew=1, ms=10, markerfacecolor='none', label='Post-Transition Impurity Velocity (' + r'$\frac{P-<P_{ph}>}{m_{I}}$' + ')')
    # ax.plot(aIBi_Vals, nu * np.ones(aIBi_Vals.size), 'g--', linewidth=3.0, label='BEC Sound Speed')
    # ax.set_ylim([0, 1.2])
    # ax.legend(loc=(0.25, 0.1))
    # ax.set_title('Velocity Comparison')
    # ax.set_xlabel(r'$a_{IB}^{-1}$')
    # ax.set_ylabel('Velocity')

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

    # fig, ax = plt.subplots()
    # ax.plot(aIBi_Vals, massEnhancement_Vals, 'cD', mew=1, ms=10)
    # ax.plot(aIBi_interpVals, mE_interpVals, 'c-')
    # ax.set_title('Mass Enhancement')
    # ax.set_xlabel(r'$a_{IB}^{-1}$')
    # ax.set_ylabel(r'$\frac{m^{*}}{m_{I}} = \frac{1}{m_{I}}\frac{\partial^{2} E}{\partial P^{2}}$')

    # # # POLARON EFFECTIVE MASS VS CRITICAL MOMENTUM (SPHERICAL)

    # ms_Vals = np.zeros(aIBi_Vals.size)
    # Pcrit = np.zeros(aIBi_Vals.size)
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

    #     Pcrit[aind] = Pinf_Vals[np.argmin(np.gradient(Einf_2ndderiv_Vals)) - 0]

    # massEnhancement_Vals = (1 / ms_Vals) / mI
    # Pcrit_norm = Pcrit / (mI * nu)
    # print(massEnhancement_Vals)
    # print(Pcrit_norm)
    # print(100 * np.abs(massEnhancement_Vals - Pcrit_norm) / Pcrit_norm)

    # fig, ax = plt.subplots()
    # ax.plot(aIBi_Vals, massEnhancement_Vals, 'co', mew=1, ms=10, markerfacecolor='none', label='Mass Enhancement (' + r'$\frac{m^{*}}{m_{I}}$' + ')')
    # ax.plot(aIBi_Vals, Pcrit_norm, 'kx', mew=1, ms=10, label='Normalized Critical Momentum (' + r'$\frac{P_{crit}}{m_{I}c_{BEC}}$' + ')')
    # ax.legend(loc=2)
    # ax.set_title('Mass Enhancement vs Critical Momentum')
    # ax.set_xlabel(r'$a_{IB}^{-1}$')

    # # FIG 3 - IMPURITY DISTRIBUTION WITH CHARACTERIZATION (CARTESIAN)

    # matplotlib.rcParams.update({'font.size': 12})
    # labelsize = 13
    # legendsize = 12

    # GaussianBroadening = True; sigma = 0.1
    # incoh_color = '#8f1402'
    # delta_color = '#bf9005'
    # fwhm_color = '#06470c'

    # def GPDF(xVals, mean, stdev):
    #     return (1 / (stdev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xVals - mean) / stdev)**2)
    #     # return (1 / (1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xVals - mean) / stdev)**2)

    # aIBi = -2
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

    # axes3[0].plot(PIm_Vec[Pind] / (mI * nu), nPIm_Vec[Pind], color=incoh_color, lw=0.5, label='Incoherent Part')
    # axes3[0].set_xlim([-0.01, 10])
    # axes3[0].set_ylim([0, 1.05])
    # axes3[0].set_ylabel(r'$n_{|\vec{P_{I}}|}$', fontsize=labelsize)
    # axes3[0].set_xlabel(r'$|\vec{P_{I}}|/(m_{I}c)$', fontsize=labelsize)
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
    # axes3[1].set_xlabel(r'$P$ [$m_{I}c$]', fontsize=labelsize)
    # axes3[1].set_ylabel(r'Quasiparticle Residue', fontsize=labelsize)

    # axes3[2].plot(PVals / (mI * nu), nPIm_FWHM_Vals, linestyle='-', color=fwhm_color)
    # axes3[2].set_xlabel(r'$P$ [$m_{I}c$]', fontsize=labelsize)
    # axes3[2].set_ylabel('Incoherent Part FWHM', fontsize=labelsize)
    # axes3[2].set_ylim([0.5, 2.5])
    # axes3[2].yaxis.set_major_locator(plt.MaxNLocator(4))

    # fig3.text(0.01, 0.95, '(a)', fontsize=labelsize)
    # fig3.text(0.33, 0.95, '(b)', fontsize=labelsize)
    # fig3.text(0.66, 0.95, '(c)', fontsize=labelsize)

    # fig3.subplots_adjust(left=0.1, bottom=0.17, top=0.91, right=0.98, wspace=0.6)
    # fig3.set_size_inches(7.8, 3.5)
    # # fig3.savefig(figdatapath + '/Fig3.pdf')

    plt.show()
