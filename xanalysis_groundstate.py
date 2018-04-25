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

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # Toggle parameters

    toggleDict = {'Location': 'home', 'Dynamics': 'imaginary', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon'}

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

    # # # Concatenate Individual Datasets

    # ds_list = []; P_list = []; aIBi_list = []; mI_list = []
    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     if filename == 'quench_Dataset_cart.nc':
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
    # ds_tot.to_netcdf(innerdatapath + '/quench_Dataset_cart.nc')

    # # Analysis of Total Dataset

    qds = xr.open_dataset(innerdatapath + '/quench_Dataset_cart.nc')
    PVals = qds['P'].values
    tVals = qds['t'].values
    n0 = qds.attrs['n0']
    gBB = qds.attrs['gBB']
    nu = pfc.nu(gBB)
    mI = qds.attrs['mI']
    mB = qds.attrs['mB']

    aIBi = -10
    qds_aIBi = qds.sel(aIBi=aIBi)

    # # # Z-FACTOR (SPHERICAL)

    # Zfac_ds = np.exp(-1 * qds_aIBi['Nph'])
    # Zfac_Vals = np.zeros((PVals.size, tVals.size))
    # for Pind, P in enumerate(PVals):
    #     for tind, t in enumerate(tVals):
    #         Zfac_Vals[Pind, tind] = Zfac_ds.sel(P=P, t=t).values

    # fig, ax = plt.subplots()
    # ax.plot(PVals, Zfac_Vals[:, -1], 'k-')
    # ax.set_title('Z-Factor (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax.set_xlabel('P')
    # ax.set_ylabel('Z-Factor (' + r'$e^{- N_{ph}}$' + ')')

    # fig2, ax2 = plt.subplots()
    # quadZ = ax2.pcolormesh(tVals, PVals, Zfac_Vals, norm=colors.LogNorm())
    # ax2.set_xscale('log')
    # ax2.set_xlabel('Imaginary Time')
    # ax2.set_ylabel('P')
    # ax2.set_title('Z-Factor (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # fig2.colorbar(quadZ, ax=ax2, extend='max')
    # plt.show()

    # # # ENERGY CHARACTERIZATION (SPHERICAL)

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
    # Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)

    # fig, ax = plt.subplots()
    # ax.plot(Pinf_Vals, Einf_Vals, 'k-', label='Energy')
    # ax.plot(Pinf_Vals, Einf_2ndderiv_Vals, 'ro', label='2nd Derivative of Energy')
    # ax.legend()
    # ax.set_title('Ground State Energy (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax.set_xlabel('P')

    # fig2, ax2 = plt.subplots()
    # quadEnergy = ax2.pcolormesh(tVals, PVals, Energy_Vals, norm=colors.SymLogNorm(linthresh=0.03))
    # ax2.set_xscale('log')
    # ax2.set_xlabel('Imaginary Time')
    # ax2.set_ylabel('P')
    # ax2.set_title('Energy (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # fig2.colorbar(quadEnergy, ax=ax2, extend='max')
    # plt.show()

    # # # PHONON MODE CHARACTERIZATION (SPHERICAL)

    # CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')

    # kAve_Vals = np.zeros(PVals.size)
    # thFWHM_Vals = np.zeros(PVals.size)
    # PhDen_k_Vec = np.empty(PVals.size, dtype=np.object)
    # PhDen_th_Vec = np.empty(PVals.size, dtype=np.object)
    # CSAmp_ds_inf = CSAmp_ds.isel(t=-1)
    # for Pind, P in enumerate(PVals):
    #     CSAmp = CSAmp_ds_inf.sel(P=P).values
    #     Nph = qds_aIBi.isel(t=-1).sel(P=P)['Nph'].values
    #     PhDen = (1 / Nph) * kgrid.dV() * np.abs(CSAmp.reshape(CSAmp.size))**2
    #     PhDen_k = kgrid.integrateFunc(PhDen, 'th'); PhDen_k_Vec[Pind] = PhDen_k
    #     PhDen_th = kgrid.integrateFunc(PhDen, 'k'); PhDen_th_Vec[Pind] = PhDen_th
    #     kAve_Vals[Pind] = np.dot(kVec, PhDen_k)
    #     thFWHM_Vals[Pind] = pfc.FWHM(thVec, PhDen_th)

    # fig1, ax1 = plt.subplots(1, 2)
    # ax1[0].plot(PVals, kAve_Vals, 'b-')
    # ax1[1].plot(PVals, thFWHM_Vals, 'g-')
    # ax1[0].set_xlabel('P')
    # ax1[0].set_ylabel('Average ' + r'$|\vec{k}|$')
    # ax1[0].set_title('Mean of ' + r'$|\vec{k}|$' + ' Distribution of Emitted Phonons (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax1[1].set_xlabel('P')
    # ax1[1].set_ylabel('FWHM of ' + r'$\theta$')
    # ax1[1].set_title('Breadth of ' + r'$\theta$' + ' Distribution of Emitted Phonons (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))

    # plt.show()

    # fig2, ax2 = plt.subplots()
    # curve2 = ax2.plot(kVec, PhDen_k_Vec[0], color='g', lw=2)[0]
    # P_text2 = ax2.text(0.85, 0.9, 'P: {:.2f}'.format(PVals[0]), transform=ax2.transAxes, color='r')
    # ax2.set_xlim([-0.01, np.max(kVec)])
    # ax2.set_ylim([0, .005])
    # ax2.set_title('Phonon Momentum Magnitude Distribution (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax2.set_ylabel(r'$\int n_{\vec{k}} \cdot \sin(\theta) d\theta$' + '  where  ' + r'$n_{\vec{k}}=\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2} dV_{\vec{k}}$')
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
    # ax3.set_ylim([0, .0005])
    # ax3.set_title('Phonon Momentum Direction Distribution (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax3.set_ylabel(r'$\int n_{\vec{k}} \cdot k^{2} dk$' + '  where  ' + r'$n_{\vec{k}}=\frac{1}{N_{ph}}|\beta_{\vec{k}}|^{2} dV_{\vec{k}}$')
    # ax3.set_xlabel(r'$|\vec{k}|$')

    # def animate3(i):
    #     curve3.set_ydata(PhDen_th_Vec[i])
    #     P_text3.set_text('P: {:.2f}'.format(PVals[i]))
    # anim3 = FuncAnimation(fig3, animate3, interval=1000, frames=range(PVals.size))
    # anim3.save(animpath + '/aIBi_{0}'.format(aIBi) + '_PhononDist_theta.gif', writer='imagemagick')

    # # plt.draw()
    # # plt.show()

    # # GROUND STATE DISTRIBUTION CHARACTERIZATION (CARTESIAN)

    # nPIm_FWHM_Vals = np.zeros(PVals.size)
    # nPIm_distPeak_Vals = np.zeros(PVals.size)
    # nPIm_deltaPeak_Vals = np.zeros(PVals.size)
    # fig, ax = plt.subplots()
    # for ind, P in enumerate(PVals):
    #     qds_nPIm_inf = qds_aIBi['nPI_mag'].sel(P=P).isel(t=-1).dropna('PI_mag')
    #     PIm_Vals = qds_nPIm_inf.coords['PI_mag'].values
    #     dPIm = PIm_Vals[1] - PIm_Vals[0]

    #     # # Plot nPIm(t=inf)
    #     # qds_nPIm_inf.plot(ax=ax, label='P: {:.1f}'.format(P))

    #     # # Calculate nPIm(t=inf) normalization
    #     nPIm_Tot = np.sum(qds_nPIm_inf.values * dPIm) + qds_aIBi.sel(P=P).isel(t=-1)['mom_deltapeak'].values

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
    # plt.show()

    # # Plot characterization of nPIm(t=inf)
    # ax.plot(PVals, nPIm_FWHM_Vals, 'b-', label='Incoherent Dist FWHM')
    # ax.plot(PVals, nPIm_distPeak_Vals, 'g-', label='Incoherent Dist Peak')
    # ax.plot(PVals, nPIm_deltaPeak_Vals, 'r-', label='Delta Peak (Z-factor)')
    # ax.legend()
    # ax.set_xlabel('$P$')
    # ax.set_title(r'$n_{|P_{I}|}$' + ' Characterization (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
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
