import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import writers
import os
import itertools
from scipy.interpolate import griddata
import pf_dynamic_cart as pfc

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})
    mpegWriter = writers['ffmpeg'](fps=20, bitrate=1800)

    # gParams

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    # (Lx, Ly, Lz) = (20, 20, 20)
    # (dx, dy, dz) = (0.5, 0.5, 0.5)

    NGridPoints = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'real', 'Grid': 'cartesian', 'Coupling': 'twophonon'}

    if toggleDict['Dynamics'] == 'real':
        innerdatapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}/redyn_cart'.format(NGridPoints, 1)
        animpath = '/media/kis/Storage/Dropbox/VariationalResearch/DataAnalysis/figs/rdyn_twophonon'
    if toggleDict['Dynamics'] == 'imaginary':
        innerdatapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}/imdyn_cart'.format(NGridPoints, 1)
        animpath = '/media/kis/Storage/Dropbox/VariationalResearch/DataAnalysis/figs/idyn_twophonon'

    # Analysis of Total Dataset

    aIBi = -2
    qds = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))

    # qds['nPI_mag'].sel(aIBi=-10, P=1.5, t=99).dropna('PI_mag').plot()
    # qds['nPI_xz_slice'].sel(aIBi=-10, P=1.5, t=99).dropna('PI_z').plot()
    # plt.show()

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
    PI_x = qds['PI_x'].values
    PI_y = qds['PI_y'].values
    PI_z = qds['PI_z'].values

    # # IMPURITY MOMENTUM MAGNITUDE DISTRIBUTION ANIMATION WITH CHARACTERIZATION (CARTESIAN)

    # vI0_Vals = np.zeros(PVals.size)
    # vIf_Vals = np.zeros(PVals.size)
    # nPIm_FWHM_indices = []
    # nPIm_distPeak_index = np.zeros(PVals.size, dtype=int)
    # nPIm_FWHM_Vals = np.zeros(PVals.size)
    # nPIm_distPeak_Vals = np.zeros(PVals.size)
    # nPIm_deltaPeak_Vals = np.zeros(PVals.size)
    # nPIm_Tot_Vals = np.zeros(PVals.size)
    # nPIm_Vec = np.empty(PVals.size, dtype=np.object)
    # PIm_Vec = np.empty(PVals.size, dtype=np.object)

    # for ind, P in enumerate(PVals):
    #     qds_nPIm_inf = qds['nPI_mag'].sel(P=P).isel(t=-1).dropna('PI_mag')
    #     PIm_Vals = qds_nPIm_inf.coords['PI_mag'].values
    #     dPIm = PIm_Vals[1] - PIm_Vals[0]

    #     # # Plot nPIm(t=inf)
    #     # qds_nPIm_inf.plot(ax=ax, label='P: {:.1f}'.format(P))
    #     nPIm_Vec[ind] = qds_nPIm_inf.values
    #     PIm_Vec[ind] = PIm_Vals

    #     # # Calculate nPIm(t=inf) normalization
    #     nPIm_Tot_Vals[ind] = np.sum(qds_nPIm_inf.values * dPIm) + qds.sel(P=P).isel(t=-1)['mom_deltapeak'].values

    #     # Calculate FWHM, distribution peak, and delta peak
    #     nPIm_FWHM_Vals[ind] = pfc.FWHM(PIm_Vals, qds_nPIm_inf.values)
    #     nPIm_distPeak_Vals[ind] = np.max(qds_nPIm_inf.values)
    #     nPIm_deltaPeak_Vals[ind] = qds.sel(P=P).isel(t=-1)['mom_deltapeak'].values

    #     D = qds_nPIm_inf.values - np.max(qds_nPIm_inf.values) / 2
    #     indices = np.where(D > 0)[0]
    #     nPIm_FWHM_indices.append((indices[0], indices[-1]))
    #     nPIm_distPeak_index[ind] = np.argmax(qds_nPIm_inf.values)

    #     vI0_Vals[ind] = (P - qds['PB'].sel(P=P).isel(t=0).values) / mI
    #     vIf_Vals[ind] = (P - qds['PB'].sel(P=P).isel(t=-1).values) / mI

    # fig1, ax = plt.subplots()
    # ax.plot(mI * nu * np.ones(PIm_Vals.size), np.linspace(0, 1, PIm_Vals.size), 'y--', label=r'$m_{I}c$')
    # curve = ax.plot(PIm_Vec[0], nPIm_Vec[0], color='k', lw=3, label='')[0]
    # ind_s, ind_f = nPIm_FWHM_indices[0]
    # FWHMcurve = ax.plot(np.linspace(PIm_Vec[0][ind_s], PIm_Vec[0][ind_f], 100), nPIm_Vec[0][ind_s] * np.ones(100), 'b-', linewidth=3.0, label='Incoherent Part FWHM')[0]
    # FWHMmarkers = ax.plot(np.linspace(PIm_Vec[0][ind_s], PIm_Vec[0][ind_f], 2), nPIm_Vec[0][ind_s] * np.ones(2), 'bD', mew=0.75, ms=7.5, label='')[0]

    # Zline = ax.plot(PVals[0] * np.ones(PIm_Vals.size), np.linspace(0, nPIm_deltaPeak_Vals[0], PIm_Vals.size), 'r-', linewidth=3.0, label='Delta Peak (Z-factor)')[0]
    # Zmarker = ax.plot(PVals[0], nPIm_deltaPeak_Vals[0], 'rx', mew=0.75, ms=7.5, label='')[0]
    # norm_text = ax.text(0.61, 0.7, r'$\int n_{|\vec{P_{I}}|} d|\vec{P_{I}}| = $' + '{:.2f}'.format(nPIm_Tot_Vals[0]), transform=ax.transAxes, color='k')
    # P_text = ax.text(0.61, 0.6, r'$\frac{P}{m_{I}c_{BEC}}=\frac{<v_{I}(t_{0})>}{c_{BEC}}=$' + '{:.2f}'.format(Pnorm[0]), transform=ax.transAxes, color='g')
    # vIf_text = ax.text(0.61, 0.5, r'$\frac{<v_{I}(t_{f})>}{c_{BEC}}=$' + '{:.2f}'.format(vIf_Vals[0] / nu), transform=ax.transAxes, color='g')
    # Z_text = ax.text(0.61, 0.4, r'Z-factor ($e^{-N_{ph}=|S(t)|^{2}}$): ' + '{:.2f}'.format(nPIm_deltaPeak_Vals[0]), transform=ax.transAxes, color='r')

    # def animate1(i):
    #     curve.set_xdata(PIm_Vec[i])
    #     curve.set_ydata(nPIm_Vec[i])
    #     ind_s, ind_f = nPIm_FWHM_indices[i]
    #     FWHMcurve.set_xdata(np.linspace(PIm_Vec[i][ind_s], PIm_Vec[i][ind_f], 100))
    #     FWHMcurve.set_ydata(nPIm_Vec[i][ind_s] * np.ones(100))
    #     FWHMmarkers.set_xdata(np.linspace(PIm_Vec[i][ind_s], PIm_Vec[i][ind_f], 2))
    #     FWHMmarkers.set_ydata(nPIm_Vec[i][ind_s] * np.ones(2))
    #     Zline.set_xdata(PVals[i] * np.ones(PIm_Vals.size))
    #     Zline.set_ydata(np.linspace(0, nPIm_deltaPeak_Vals[i], PIm_Vals.size))
    #     Zmarker.set_xdata(PVals[i])
    #     Zmarker.set_ydata(nPIm_deltaPeak_Vals[i])
    #     norm_text.set_text(r'$\int n_{|\vec{P_{I}}|} d|\vec{P_{I}}| = $' + '{:.2f}'.format(nPIm_Tot_Vals[i]))
    #     P_text.set_text(r'$\frac{P}{m_{I}c_{BEC}}=\frac{<v_{I}(t_{0})>}{c_{BEC}}=$' + '{:.2f}'.format(Pnorm[i]))
    #     vIf_text.set_text(r'$\frac{<v_{I}(t_{f})>}{c_{BEC}}=$' + '{:.2f}'.format(vIf_Vals[i] / nu))
    #     Z_text.set_text(r'Z-factor ($e^{-N_{ph}=|S(t)|^{2}}$): ' + '{:.2f}'.format(nPIm_deltaPeak_Vals[i]))

    # ax.legend()
    # ax.set_xlim([-0.01, np.max(PIm_Vec[0])])
    # ax.set_ylim([0, 1.05])
    # ax.set_title('Final Time Impurity Momentum Magnitude Distribution (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax.set_ylabel(r'$n_{|\vec{P_{I}}|}$')
    # ax.set_xlabel(r'$|\vec{P_{I}}|$')

    # anim1 = FuncAnimation(fig1, animate1, interval=1500, frames=range(PVals.size))
    # # anim1.save(animpath + '/aIBi_{0}'.format(aIBi) + '_ImpDist&Char.gif', writer='imagemagick')

    # plt.show()

    # # IMPURITY MOMENTUM DISTRIBUTION 2D SLICE ANIMATION WITH CHARACTERIZATION (CARTESIAN)

    # interpmul = 8
    # # vI_Z_ave = np.zeros(PVals.size)
    # vI_X_ave = np.zeros(PVals.size)
    # vI0_Vals = np.zeros(PVals.size)
    # vIf_Vals = np.zeros(PVals.size)
    # nPIm_deltaPeak_Vals = np.zeros(PVals.size)
    # nPI_xz_inf = qds['nPI_xz_slice'].isel(t=-1)

    # vmin = 1e20
    # vmax = -1e20
    # for ind, P in enumerate(PVals):
    #     qds_nPI_inf = nPI_xz_inf.sel(P=P).dropna('PI_z')
    #     nPI_inf_Vals = qds_nPI_inf.values
    #     PI_x_inf = qds_nPI_inf['PI_x'].values; dPIx = PI_x_inf[1] - PI_x_inf[0]
    #     PI_z_inf = qds_nPI_inf['PI_z'].values; dPIz = PI_z_inf[1] - PI_z_inf[0]
    #     PI_xg, PI_zg = np.meshgrid(PI_x_inf, PI_z_inf, indexing='ij')

    #     vI_X_ave[ind] = (np.sum(PI_xg * nPI_inf_Vals * dPIx * dPIz)) / mI

    #     nPImax = np.max(nPI_inf_Vals)
    #     nPImin = np.min(nPI_inf_Vals)
    #     if nPImax > vmax:
    #         vmax = nPImax
    #     if nPImin < vmin:
    #         vmin = nPImin

    #     nPIm_deltaPeak_Vals[ind] = qds.sel(P=P).isel(t=-1)['mom_deltapeak'].values
    #     vI0_Vals[ind] = (P - qds['PB'].sel(P=P).isel(t=0).values) / mI
    #     vIf_Vals[ind] = (P - qds['PB'].sel(P=P).isel(t=-1).values) / mI
    #     # vI_Z_ave[ind] = (np.sum(PI_zg * nPI_inf_Vals * dPIx * dPIz) + P * nPIm_deltaPeak_Vals[ind]) / mI

    # fig1, ax = plt.subplots()

    # nPI_xz_interp0, PI_xg_interp0, PI_zg_interp0 = pfc.xinterp2D(nPI_xz_inf.isel(P=0).dropna('PI_z'), 'PI_x', 'PI_z', interpmul)
    # quad = ax.pcolormesh(PI_zg_interp0, PI_xg_interp0, nPI_xz_interp0[:-1, :-1], vmin=vmin, vmax=vmax, cmap='inferno')
    # P_text = ax.text(0.65, 0.9, r'$\frac{P}{m_{I}c_{BEC}}=\frac{<v_{I}(t_{0})>}{c_{BEC}}=$' + '{:.2f}'.format(Pnorm[0]), transform=ax.transAxes, fontsize='small', color='w')
    # vIf_text = ax.text(0.65, 0.82, r'$\frac{<v_{I}(t_{f})>}{c_{BEC}}=$' + '{:.2f}'.format(vIf_Vals[0] / nu), transform=ax.transAxes, fontsize='small', color='w')
    # # vIZ_text = ax.text(0.65, 0.58, r'$\frac{1}{m_{I}c_{BEC}}<P_{I,z}(t_{f})>=$' + '{:.2f}'.format(vI_Z_ave[0] / nu), transform=ax.transAxes, fontsize='small', color='w')  # for some reason this is different than average impurity velocity...***related to not including the delta-peak
    # vIX_text = ax.text(0.65, 0.74, r'$\frac{1}{m_{I}c_{BEC}}<P_{I,x}(t_{f})>=$' + '{:.2f}'.format(vI_X_ave[0] / nu), transform=ax.transAxes, fontsize='small', color='w')
    # Z_text = ax.text(0.65, 0.66, r'Z-factor: ' + '{:.2f}'.format(nPIm_deltaPeak_Vals[0]), transform=ax.transAxes, fontsize='small', color='w')
    # # Z_text = ax.text(0.6, 0.58, r'Z-factor ($e^{-N_{ph}=|S(t)|^{2}}$): ' + '{:.2f}'.format(nPIm_deltaPeak_Vals[0]), transform=ax.transAxes, fontsize='small', color='w')

    # def animate1(i):
    #     nPI_xz_interp, PI_xg_interp, PI_zg_interp = pfc.xinterp2D(nPI_xz_inf.isel(P=i).dropna('PI_z'), 'PI_x', 'PI_z', interpmul)
    #     quad.set_array(nPI_xz_interp[:-1, :-1].ravel())
    #     P_text.set_text(r'$\frac{P}{m_{I}c_{BEC}}=\frac{<v_{I}(t_{0})>}{c_{BEC}}=$' + '{:.2f}'.format(Pnorm[i]))
    #     vIf_text.set_text(r'$\frac{<v_{I}(t_{f})>}{c_{BEC}}=$' + '{:.2f}'.format(vIf_Vals[i] / nu))
    #     # vIZ_text.set_text(r'$\frac{1}{m_{I}c_{BEC}}<P_{I,z}(t_{f})>=$' + '{:.2f}'.format(vI_Z_ave[i] / nu))
    #     vIX_text.set_text(r'$\frac{1}{m_{I}c_{BEC}}<P_{I,x}(t_{f})>=$' + '{:.2f}'.format(vI_X_ave[i] / nu))
    #     Z_text.set_text(r'Z-factor: ' + '{:.2f}'.format(nPIm_deltaPeak_Vals[i]))
    #     # Z_text.set_text(r'Z-factor ($e^{-N_{ph}=|S(t)|^{2}}$): ' + '{:.2f}'.format(nPIm_deltaPeak_Vals[i]))

    # ax.set_title('Final Time Impurity Longitudinal Momentum Distribution (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax.set_ylabel(r'$P_{I,x}$')
    # ax.set_xlabel(r'$P_{I,z}$')
    # ax.set_xlim([-2.5, 2.5])
    # ax.set_ylim([-2.5, 2.5])
    # ax.grid(True, linewidth=0.5)
    # fig1.colorbar(quad, ax=ax, extend='both')

    # anim1 = FuncAnimation(fig1, animate1, interval=1500, frames=range(PVals.size))
    # # anim1.save(animpath + '/aIBi_{0}'.format(aIBi) + '_ImpDist_xz_slice.gif', writer='imagemagick')

    # plt.show()

    # IMPURITY MOMENTUM DISTRIBUTION Z-INTEGRATION ANIMATION WITH CHARACTERIZATION (CARTESIAN)

    vI_Z_ave = np.zeros(PVals.size)
    vI0_Vals = np.zeros(PVals.size)
    vIf_Vals = np.zeros(PVals.size)
    nPIm_deltaPeak_Vals = np.zeros(PVals.size)
    nPI_z_inf = qds['nPI_z_int'].isel(t=-1)

    for ind, P in enumerate(PVals):
        qds_nPI_inf = nPI_z_inf.sel(P=P).dropna('PI_z')
        nPI_inf_Vals = qds_nPI_inf.values

        nPIm_deltaPeak_Vals[ind] = qds.sel(P=P).isel(t=-1)['mom_deltapeak'].values
        vI0_Vals[ind] = (P - qds['PB'].sel(P=P).isel(t=0).values) / mI
        vIf_Vals[ind] = (P - qds['PB'].sel(P=P).isel(t=-1).values) / mI
        PI_z_inf = qds_nPI_inf['PI_z'].values; dPIz = PI_z_inf[1] - PI_z_inf[0]
        vI_Z_ave[ind] = (np.sum(PI_z_inf * nPI_inf_Vals * dPIz) + P * nPIm_deltaPeak_Vals[ind]) / mI

        # print(np.sum(nPI_inf_Vals * dPIz) + nPIm_deltaPeak_Vals[ind])

    fig1, ax = plt.subplots()

    nPI_z_inf_P = nPI_z_inf.isel(P=0).dropna('PI_z')
    curve = ax.plot(nPI_z_inf_P['PI_z'].values / mc, nPI_z_inf_P.values, 'k-')[0]
    Zline = ax.plot(PVals[0] * np.ones(100), np.linspace(0, nPIm_deltaPeak_Vals[0], 100), 'r-', linewidth=3.0, label='Delta Peak (Z-factor)')[0]
    Zmarker = ax.plot(PVals[0], nPIm_deltaPeak_Vals[0], 'rx', mew=0.75, ms=7.5, label='')[0]
    ax.plot(np.ones(100), np.linspace(0, 1, 100), 'k--', label=r'$c_{BEC}$')

    P_text = ax.text(0.65, 0.9, r'$\frac{P}{m_{I}c_{BEC}}=\frac{<v_{I}(t_{0})>}{c_{BEC}}=$' + '{:.2f}'.format(Pnorm[0]), transform=ax.transAxes, fontsize='small', color='g')
    vIf_text = ax.text(0.65, 0.82, r'$\frac{<v_{I}(t_{f})>}{c_{BEC}}=$' + '{:.2f}'.format(vIf_Vals[0] / nu), transform=ax.transAxes, fontsize='small', color='g')
    vIZ_text = ax.text(0.65, 0.74, r'$\frac{1}{m_{I}c_{BEC}}<P_{I,z}(t_{f})>=$' + '{:.2f}'.format(vI_Z_ave[0] / nu), transform=ax.transAxes, fontsize='small', color='g')  # for some reason this is different than average impurity velocity after large values of P
    Z_text = ax.text(0.65, 0.66, r'Z-factor: ' + '{:.2f}'.format(nPIm_deltaPeak_Vals[0]), transform=ax.transAxes, fontsize='small', color='g')

    def animate1(i):
        nPI_z_inf_P = nPI_z_inf.isel(P=i).dropna('PI_z')
        curve.set_xdata(nPI_z_inf_P['PI_z'].values / mc)
        curve.set_ydata(nPI_z_inf_P.values)
        Zline.set_xdata(PVals[i] * np.ones(100))
        Zline.set_ydata(np.linspace(0, nPIm_deltaPeak_Vals[i], 100))
        Zmarker.set_xdata(PVals[i])
        Zmarker.set_ydata(nPIm_deltaPeak_Vals[i])

        P_text.set_text(r'$\frac{P}{m_{I}c_{BEC}}=\frac{<v_{I}(t_{0})>}{c_{BEC}}=$' + '{:.2f}'.format(Pnorm[i]))
        vIf_text.set_text(r'$\frac{<v_{I}(t_{f})>}{c_{BEC}}=$' + '{:.2f}'.format(vIf_Vals[i] / nu))
        vIZ_text.set_text(r'$\frac{1}{m_{I}c_{BEC}}<P_{I,z}(t_{f})>=$' + '{:.2f}'.format(vI_Z_ave[i] / nu))
        Z_text.set_text(r'Z-factor: ' + '{:.2f}'.format(nPIm_deltaPeak_Vals[i]))

    ax.legend(loc=2)
    ax.set_title('Final Time Integrated Impurity Momentum Distribution (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    ax.set_ylabel(r'$n_{P_{I,z}}=\int dP_{I,x}dP_{I,y} n_{\vec{P_{I}}}$')
    ax.set_xlabel(r'$\frac{1}{m_{I}c_{BEC}}P_{I,z}$')
    ax.set_xlim([np.min(PI_z), np.max(PI_z)])
    ax.set_ylim([0, 1.05])

    anim1 = FuncAnimation(fig1, animate1, interval=1500, frames=range(PVals.size))
    # anim1.save(animpath + '/aIBi_{0}'.format(aIBi) + '_ImpDist_z_int.gif', writer='imagemagick')

    plt.show()
