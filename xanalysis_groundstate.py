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

    toggleDict = {'Location': 'work', 'Dynamics': 'imaginary', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon'}

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
    # ax.set_title('Ground State Energy (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # ax.set_xlabel('P')

    # fig2, ax2 = plt.subplots()
    # quadEnergy = ax2.pcolormesh(tVals, PVals, Energy_Vals, norm=colors.SymLogNorm(linthresh=0.03))
    # ax2.set_xscale('log')
    # ax2.set_xlabel('Imaginary Time')
    # ax2.set_ylabel('P')
    # ax2.set_title('Energy (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # fig2.colorbar(quadEnergy, ax=ax2, extend='max')
    # plt.show()

    # # POLARON SOUND VELOCITY (SPHERICAL)

    # # Check to see if linear part of polaron (total system) energy spectrum has slope equal to sound velocity

    # aIBi_Vals = qds.coords['aIBi'].values
    # vsound_Vals = np.zeros(aIBi_Vals.size)
    # vI_Vals = np.zeros(aIBi_Vals.size)
    # for aind, aIBi in enumerate(aIBi_Vals):
    #     qds_aIBi = qds.sel(aIBi=aIBi).isel(t=-1)
    #     CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    #     kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    #     Energy_Vals_inf = np.zeros(PVals.size)
    #     PI_Vals = np.zeros(PVals.size)
    #     for Pind, P in enumerate(PVals):
    #         CSAmp = CSAmp_ds.sel(P=P).values
    #         Energy_Vals_inf[Pind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)
    #         PI_Vals[Pind] = P - qds_aIBi.sel(P=P)['Pph'].values

    #     Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)
    #     Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 5 * PVals.size)
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

    # fig, ax = plt.subplots()
    # ax.plot(aIBi_Vals, vsound_Vals, 'ro', label='Post-Transition Polaron Sound Velocity (' + r'$\frac{\partial E}{\partial P}$' + ')')
    # ax.plot(aIBi_Vals, vI_Vals, 'go', label='Post-Transition Impurity Velocity (' + r'$\frac{P-P_{ph}}{m_{I}}$' + ')')
    # ax.plot(aIBi_Vals, nu * np.ones(aIBi_Vals.size), 'k--', label='BEC Sound Speed')
    # ax.legend()
    # ax.set_title('Velocity Comparison')
    # ax.set_xlabel(r'$a_{IB}^{-1}$')
    # plt.show()

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
    # # plt.show()

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
