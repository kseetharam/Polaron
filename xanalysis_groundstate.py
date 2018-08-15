import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
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

    # xgrid = Grid.Grid('CARTESIAN_3D')
    # xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    # (Nx, Ny, Nz) = (len(xgrid.getArray('x')), len(xgrid.getArray('y')), len(xgrid.getArray('z')))

    # kxfft = np.fft.fftfreq(Nx) * 2 * np.pi / dx; kyfft = np.fft.fftfreq(Nx) * 2 * np.pi / dy; kzfft = np.fft.fftfreq(Nx) * 2 * np.pi / dz

    # kgrid = Grid.Grid('CARTESIAN_3D')
    # kgrid.initArray_premade('kx', np.fft.fftshift(kxfft)); kgrid.initArray_premade('ky', np.fft.fftshift(kyfft)); kgrid.initArray_premade('kz', np.fft.fftshift(kzfft))

    # kx = kgrid.getArray('kx')
    # dVx_const = ((2 * np.pi)**(3)) * xgrid.dV()[0]

    # dkx = kx[1] - kx[0]
    # xg = np.fft.fftshift(np.fft.fftfreq(kx.size) * 2 * np.pi / dkx)

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'imaginary', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon', 'Longtime': 'false'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, 1)
        animpath = '/home/kis/Dropbox/VariationalResearch/DataAnalysis/figs'
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, 1)
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

    if toggleDict['Longtime'] == 'true':
        innerdatapath = innerdatapath + '_longtime'
    elif toggleDict['Longtime'] == 'false':
        innerdatapath = innerdatapath

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
    aIBi = -10
    qds = xr.open_dataset(innerdatapath + '/quench_Dataset.nc')
    qds_aIBi = qds.sel(aIBi=aIBi)
    # qds = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    # qds_aIBi = qds

    PVals = qds['P'].values
    tVals = qds['t'].values
    n0 = qds.attrs['n0']
    gBB = qds.attrs['gBB']
    nu = pfc.nu(gBB)
    mI = qds.attrs['mI']
    mB = qds.attrs['mB']

    # # # # PHONON POSITION DISTRIBUTION (CARTESIAN)

    # Pinit = 3.0
    # nx_ds = qds_aIBi['nxyz_xz_slice'].isel(t=-1).sel(P=Pinit, method='nearest')
    # Nx = nx_ds.coords['x'].values.size
    # nx_interp_vals, xg_interp, zg_interp = pfc.xinterp2D(nx_ds, 'x', 'z', 5)
    # fig, ax = plt.subplots()
    # quad1 = ax.pcolormesh(zg_interp, xg_interp, nx_interp_vals[:-1, :-1])
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # # nx_ds.plot()
    # plt.show()
    # nPB_ds = qds_aIBi['nPB_xz_slice'].isel(t=-1).sel(P=Pinit, method='nearest')
    # PBx = nPB_ds.coords['PB_x'].values
    # PBz = nPB_ds.coords['PB_x'].values
    # print(PBz[1] - PBz[0])
    # kz = np.fft.fftshift(np.fft.fftfreq(Nx) * 2 * np.pi / dy)
    # print(kz[1] - kz[0])

    # # # ENERGY AND IMPURITY VELOCITY DATA CONVERSION FOR MATHEMATICA

    # mmdatapath = innerdatapath + '/mm/aIBi_{0}'.format(aIBi)

    # CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)

    # Energy_Vals = np.zeros((PVals.size, tVals.size))
    # vI_Vals = np.zeros((PVals.size, tVals.size))

    # for Pind, P in enumerate(PVals):
    #     for tind, t in enumerate(tVals):
    #         CSAmp = CSAmp_ds.sel(P=P, t=t).values
    #         Energy_Vals[Pind, tind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)
    #         vI_Vals[Pind, tind] = (P - qds_aIBi.sel(P=P, t=t)['Pph'].values) / mI

    # for Pind, P in enumerate(PVals):
    #     data = np.concatenate((P * np.ones(tVals.size)[:, np.newaxis], tVals[:, np.newaxis], Energy_Vals[Pind, :][:, np.newaxis], vI_Vals[Pind, :][:, np.newaxis]), axis=1)
    #     np.savetxt(mmdatapath + '/aIBi_{:d}_P_{:.2f}.dat'.format(aIBi, P), data)

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

    # # fig2, ax2 = plt.subplots()
    # # quadEnergy = ax2.pcolormesh(tVals, PVals, Energy_Vals, norm=colors.SymLogNorm(linthresh=0.03))
    # # ax2.set_xscale('log')
    # # ax2.set_xlabel('Imaginary Time')
    # # ax2.set_ylabel('P')
    # # ax2.set_title('Energy (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # # fig2.colorbar(quadEnergy, ax=ax2, extend='max')

    # fig3, ax3 = plt.subplots()
    # Pind = 8
    # ax3.plot(tVals, np.abs(Energy_Vals[Pind, :]), 'k-')
    # ax3.set_yscale('log')
    # ax3.set_xscale('log')
    # ax3.set_title('Ground State Energy (' + r'$a_{IB}^{-1}=$' + '{0}, '.format(aIBi) + r'$P=$' + '{:.2f})'.format(PVals[Pind]))
    # ax3.set_xlabel('Imaginary time')
    # plt.show()

    # # # POLARON SOUND VELOCITY (SPHERICAL)

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
    # print(100 * (vsound_Vals - nu) / nu)
    # fig, ax = plt.subplots()
    # ax.plot(aIBi_Vals, vsound_Vals, 'ro', label='Post-Transition Polaron Sound Velocity (' + r'$\frac{\partial E}{\partial P}$' + ')')
    # ax.plot(aIBi_Vals, vI_Vals, 'go', label='Post-Transition Impurity Velocity (' + r'$\frac{P-P_{ph}}{m_{I}}$' + ')')
    # ax.plot(aIBi_Vals, nu * np.ones(aIBi_Vals.size), 'k--', label='BEC Sound Speed')
    # ax.set_ylim([0, 1.2])
    # ax.legend()
    # ax.set_title('Velocity Comparison')
    # ax.set_xlabel(r'$a_{IB}^{-1}$')
    # plt.show()

    # # # POLARON SOUND VELOCITY SATURATION (SPHERICAL)

    # # Check to see if linear part of polaron (total system) energy spectrum has slope equal to sound velocity

    # # aIBi = -10
    # fig, ax = plt.subplots()
    # aIBi_Vals = qds.coords['aIBi'].values
    # colorList = ['b', 'g', 'r']
    # for aind, aIBi in enumerate(aIBi_Vals):
    #     qds_aIBi = qds.sel(aIBi=aIBi)
    #     tVals = qds_aIBi['t'].values
    #     vsound_Vals = np.zeros(tVals.size)
    #     vI_Vals = np.zeros(tVals.size)
    #     CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    #     kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    #     # get sound mask for Einf
    #     Energy_Vals_inf = np.zeros(PVals.size)
    #     PI_Vals = np.zeros(PVals.size)
    #     for Pind, P in enumerate(PVals):
    #         CSAmp = CSAmp_ds.sel(P=P).isel(t=-1).values
    #         Energy_Vals_inf[Pind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)
    #         PI_Vals[Pind] = P - qds_aIBi.sel(P=P).isel(t=-1)['Pph'].values
    #     Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)
    #     Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 5 * PVals.size)
    #     Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
    #     Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)
    #     sound_mask = np.abs(Einf_2ndderiv_Vals) <= 5e-3

    #     for tind, t in enumerate(tVals):
    #         Energy_Vals_inf = np.zeros(PVals.size)
    #         PI_Vals = np.zeros(PVals.size)
    #         for Pind, P in enumerate(PVals):
    #             CSAmp = CSAmp_ds.sel(P=P, t=t).values
    #             Energy_Vals_inf[Pind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)
    #             PI_Vals[Pind] = P - qds_aIBi.sel(P=P, t=t)['Pph'].values

    #         Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)
    #         Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 5 * PVals.size)
    #         Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
    #         Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)

    #         # sound_mask = np.abs(Einf_2ndderiv_Vals) <= 5e-3
    #         Einf_sound = Einf_Vals[sound_mask]
    #         Pinf_sound = Pinf_Vals[sound_mask]
    #         [vsound_Vals[tind], vs_const] = np.polyfit(Pinf_sound, Einf_sound, deg=1)

    #         vI_inf_tck = interpolate.splrep(PVals, PI_Vals / mI, s=0)
    #         vI_inf_Vals = 1 * interpolate.splev(Pinf_Vals, vI_inf_tck, der=0)
    #         vI_Vals[tind] = np.polyfit(Pinf_sound, vI_inf_Vals[sound_mask], deg=0)

    #     vsound_tr = vsound_Vals - nu
    #     vI_tr = vI_Vals - nu
    #     ax.plot(tVals, vsound_tr, color=colorList[aind], linestyle='none', marker='o', markerfacecolor='none', label='')
    #     ax.plot(tVals, vI_tr, color=colorList[aind], linestyle='none', marker='x', label='')

    # legend_elements = [Line2D([0], [0], marker='o', color='k', label='Translated Post-Transition Polaron Sound Velocity (' + r'$\frac{\partial E}{\partial P}-c_{BEC}$' + ')',
    #                           markerfacecolor='none', markersize=10, linestyle='none'),
    #                    Line2D([0], [0], marker='x', color='k', label='Translated Post-Transition Impurity Velocity (' + r'$\frac{P-P_{ph}}{m_{I}}-c_{BEC}$' + ')',
    #                           markersize=10, linestyle='none')]
    # for aind, aIBi in enumerate(aIBi_Vals):
    #     legend_elements.append(Line2D([0], [0], color=colorList[aind], lw=4, label=r'$a_{IB}^{-1}=$' + '{:.1f}'.format(aIBi)))
    # ax.legend(handles=legend_elements, loc=1)
    # # ax.set_xscale('symlog', linthreshy=1)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # # ax.legend()
    # ax.set_ylim([0.009, 3])
    # ax.set_title('Velocity Saturation')
    # ax.set_xlabel(r'$\tau=-it$')
    # plt.show()

    # # # PHONON MODE CHARACTERIZATION - INTEGRATED PLOTS (SPHERICAL)

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

    # # # PHONON MODE CHARACTERIZATION - 2D PLOTS (SPHERICAL)

    # CSAmp_ds = (qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']).isel(t=-1)
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # list_of_unit_vectors = list(kgrid.arrays.keys())
    # list_of_functions = [lambda k: (2 * np.pi)**(-2) * k**2, np.sin]
    # # sphfac = kgrid.function_prod(list_of_unit_vectors, list_of_functions)
    # sphfac = 1
    # kDiff = kgrid.diffArray('k')
    # thDiff = kgrid.diffArray('th')

    # PphVals = qds_aIBi.isel(t=-1)['Pph'].values
    # PimpVals = PVals - PphVals
    # nk = xr.DataArray(np.full((PVals.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[PVals, kVec, thVec], dims=['P', 'k', 'th'])
    # for Pind, P in enumerate(PVals):
    #     CSAmp_Vals = CSAmp_ds.sel(P=P).values
    #     CSAmp_flat = CSAmp_Vals.reshape(CSAmp_Vals.size)
    #     Nph = qds_aIBi.isel(t=-1).sel(P=P)['Nph'].values
    #     # Nph = 1
    #     PhDen = (1 / Nph) * sphfac * np.abs(CSAmp_flat)**2
    #     nk.sel(P=P)[:] = PhDen.reshape((len(kVec), len(thVec))).real.astype(float)

    # # # Full transition

    # # fig1, ax1 = plt.subplots()
    # # vmin = 1
    # # vmax = 0
    # # for Pind, Pv in enumerate(PVals):
    # #     vec = nk.sel(P=Pv).values
    # #     if np.min(vec) < vmin:
    # #         vmin = np.min(vec)
    # #     if np.max(vec) > vmax:
    # #         vmax = np.max(vec)
    # #     print(vmin, vmax)

    # # # print(vmin, vmax)
    # # vmin = 0
    # # vmax = 500
    # # # vmin = 1e13
    # # # vmax = 1e14

    # # nk0_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(nk.isel(P=0), 'k', 'th', 5)
    # # xg_interp = kg_interp * np.sin(thg_interp)
    # # zg_interp = kg_interp * np.cos(thg_interp)

    # # quad1 = ax1.pcolormesh(zg_interp, xg_interp, nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # # quad1m = ax1.pcolormesh(zg_interp, -1 * xg_interp, nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # # curve1 = ax1.plot(PphVals[0], 0, marker='x', markersize=10, color="magenta", label=r'$P_{ph}$')[0]
    # # curve1m = ax1.plot(PimpVals[0], 0, marker='o', markersize=10, color="red", label=r'$P_{imp}$')[0]
    # # curvec = ax1.plot(mI * nu, 0, marker='s', markersize=5, color="white", label=r'$m_{I}c$')[0]

    # # P_text = ax1.text(0.83, 0.95, 'P: {:.2f}'.format(PVals[0]), transform=ax1.transAxes, color='g')
    # # mIc_text = ax1.text(0.83, 0.85, r'$m_{I}c$' + ': {:.2f}'.format(mI * nu), transform=ax1.transAxes, color='w')
    # # Pimp_text = ax1.text(0.83, 0.8, r'$P_{imp}$' + ': {:.2f}'.format(PimpVals[0]), transform=ax1.transAxes, color='r')
    # # Pph_text = ax1.text(0.83, 0.75, r'$P_{ph}$' + ': {:.2f}'.format(PphVals[0]), transform=ax1.transAxes, color='m')
    # # ax1.set_xlim([-1.5, 1.5])
    # # ax1.set_ylim([-1.5, 1.5])
    # # # ax1.set_xlim([-3, 3])
    # # # ax1.set_ylim([-3, 3])
    # # ax1.legend(loc=2)
    # # ax1.grid(True, linewidth=0.5)
    # # ax1.set_title('Ind Phonon Distribution (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # # ax1.set_xlabel(r'$k_{z}$')
    # # ax1.set_ylabel(r'$k_{x}$')
    # # fig1.colorbar(quad1, ax=ax1, extend='both')

    # # def animate1(i):
    # #     nk_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(nk.isel(P=i), 'k', 'th', 5)
    # #     quad1.set_array(nk_interp_vals[:-1, :-1].ravel())
    # #     quad1m.set_array(nk_interp_vals[:-1, :-1].ravel())
    # #     curve1.set_xdata(PphVals[i])
    # #     curve1m.set_xdata(PimpVals[i])
    # #     P_text.set_text('P: {:.2f}'.format(PVals[i]))
    # #     Pimp_text.set_text(r'$P_{imp}$' + ': {:.2f}'.format(PimpVals[i]))
    # #     Pph_text.set_text(r'$P_{ph}$' + ': {:.2f}'.format(PphVals[i]))

    # # anim1 = FuncAnimation(fig1, animate1, interval=500, frames=range(PVals.size), blit=False)
    # # # anim1.save(animpath + '/aIBi_{:d}'.format(aIBi) + '_indPhononDist_2D_fulltransition.gif', writer='imagemagick')

    # # plt.draw()
    # # plt.show()

    # # Supersonic only

    # Pinit = 0.9
    # nkP = nk.sel(P=Pinit, method='nearest')
    # Pinit = 1 * nkP['P'].values
    # nk = nk.sel(P=slice(Pinit, PVals[-1]))
    # PVals = nk.coords['P'].values

    # fig1, ax1 = plt.subplots()
    # vmin = 1
    # vmax = 0
    # for Pind, Pv in enumerate(PVals):
    #     vec = nk.sel(P=Pv).values
    #     if np.min(vec) < vmin:
    #         vmin = np.min(vec)
    #     if np.max(vec) > vmax:
    #         vmax = np.max(vec)

    # # vmin = 1e13
    # # vmax = 1e14

    # nk0_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(nk.isel(P=0), 'k', 'th', 5)
    # xg_interp = kg_interp * np.sin(thg_interp)
    # zg_interp = kg_interp * np.cos(thg_interp)
    # # print(zg_interp[0, 1] - zg_interp[0, 0])

    # quad1 = ax1.pcolormesh(zg_interp, xg_interp, nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # quad1m = ax1.pcolormesh(zg_interp, -1 * xg_interp, nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # curve1 = ax1.plot(PphVals[0], 0, marker='x', markersize=10, color="magenta", label=r'$P_{ph}$')[0]
    # curve1m = ax1.plot(PimpVals[0], 0, marker='o', markersize=10, color="red", label=r'$P_{imp}$')[0]
    # curvec = ax1.plot(mI * nu, 0, marker='s', markersize=5, color="white", label=r'$m_{I}c$')[0]

    # P_text = ax1.text(0.83, 0.95, 'P: {:.2f}'.format(PVals[0]), transform=ax1.transAxes, color='g')
    # mIc_text = ax1.text(0.83, 0.85, r'$m_{I}c$' + ': {:.2f}'.format(mI * nu), transform=ax1.transAxes, color='w')
    # Pimp_text = ax1.text(0.83, 0.8, r'$P_{imp}$' + ': {:.2f}'.format(PimpVals[0]), transform=ax1.transAxes, color='r')
    # Pph_text = ax1.text(0.83, 0.75, r'$P_{ph}$' + ': {:.2f}'.format(PphVals[0]), transform=ax1.transAxes, color='m')
    # ax1.set_xlim([-0.1, 0.1])
    # ax1.set_ylim([-0.01, 0.01])
    # # ax1.set_xlim([-3, 3])
    # # ax1.set_ylim([-3, 3])
    # ax1.legend(loc=2)
    # ax1.grid(True, linewidth=0.5)
    # ax1.set_title('Ind Phonon Distribution (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax1.set_xlabel(r'$k_{z}$')
    # ax1.set_ylabel(r'$k_{x}$')
    # fig1.colorbar(quad1, ax=ax1, extend='both')

    # def animate1(i):
    #     nk_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(nk.isel(P=i), 'k', 'th', 5)
    #     quad1.set_array(nk_interp_vals[:-1, :-1].ravel())
    #     quad1m.set_array(nk_interp_vals[:-1, :-1].ravel())
    #     curve1.set_xdata(PphVals[i])
    #     curve1m.set_xdata(PimpVals[i])
    #     P_text.set_text('P: {:.2f}'.format(PVals[i]))
    #     Pimp_text.set_text(r'$P_{imp}$' + ': {:.2f}'.format(PimpVals[i]))
    #     Pph_text.set_text(r'$P_{ph}$' + ': {:.2f}'.format(PphVals[i]))

    # anim1 = FuncAnimation(fig1, animate1, interval=1500, frames=range(PVals.size), blit=False)
    # # anim1.save(animpath + '/aIBi_{:d}'.format(aIBi) + '_indPhononDist_2D_supersonic.gif', writer='imagemagick')

    # plt.draw()
    # plt.show()

    # # PHONON MODE POSITION CHARACTERIZATION - 2D PLOTS (SPHERICAL)

    CSAmp_ds = (qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']).isel(t=-1)
    kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    kVec = kgrid.getArray('k')
    thVec = kgrid.getArray('th')
    kDiff = kgrid.diffArray('k')
    thDiff = kgrid.diffArray('th')
    Bk = xr.DataArray(np.full((len(kVec), len(thVec)), np.nan, dtype=complex), coords=[kVec, thVec], dims=['k', 'th'])
    for Pind, P in enumerate(PVals):
        if Pind != 10:
            continue
        CSAmp_Vals = CSAmp_ds.sel(P=P).values
        Nph = qds_aIBi.isel(t=-1).sel(P=P)['Nph'].values
        Bk[:] = CSAmp_Vals.reshape((len(kVec), len(thVec)))
        Bk_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(Bk, 'k', 'th', 5)
        kxg_interp = kg_interp * np.sin(thg_interp)
        kzg_interp = kg_interp * np.cos(thg_interp)

        kxg_interp2 = np.concatenate((-1 * np.fliplr(kxg_interp), kxg_interp), axis=1)
        kzg_interp2 = np.concatenate((kzg_interp, kzg_interp), axis=1)
        # Bk_interp2_vals = np.concatenate((np.fliplr(Bk_interp_vals), Bk_interp_vals), axis=1)
        Bk_interp2_vals = np.concatenate((Bk_interp_vals, Bk_interp_vals), axis=1)  # THINK ABOUT WHY NO FLIPLR HERE

        # fig1, ax1 = plt.subplots()
        # ax1.scatter(kzg_interp, kxg_interp)
        # plt.show()

        (zdim, xdim) = kzg_interp.shape
        # now zg, xg, and Bk_interp_vals form a nonlinear grid of Bk vals in 2D -> we need to reinterpolate onto a linear grid in kx,kz space (don't forget to include the -xg branch after reinterpolation) and then do a 2D FFT, then save this to an array to plot, include Nph
        npoints = zdim
        # kzL = np.linspace(np.min(kzg_interp), np.max(kzg_interp), npoints)
        # kxL = np.linspace(np.min(kxg_interp), np.max(kxg_interp), npoints)
        # print(kzL)
        # print(kxL)

        fig, axes = plt.subplots(nrows=1, ncols=3)
        # fig1, ax1 = plt.subplots()

        if P < 0.9:
            [vmin, vmax] = [0, 500]
            axes[0].set_xlim([-1.5, 1.5])
            axes[0].set_ylim([-1.5, 1.5])
            axes[1].set_xlim([-1.5, 1.5])
            axes[1].set_ylim([-1.5, 1.5])
            axes[2].set_xlim([-1.5, 1.5])
            axes[2].set_ylim([-1.5, 1.5])

            # ax1.set_xlim([-1.5, 1.5])
            # ax1.set_ylim([-1.5, 1.5])
            kzL = np.linspace(-2, 2, 2 * npoints)
            kxL = np.linspace(0, 2, npoints)
            kzLg_interp, kxLg_interp = np.meshgrid(kzL, kxL, indexing='ij')

            kzL2 = np.linspace(-2, 2, 2 * npoints)
            kxL2 = np.linspace(-2, 2, 2 * npoints)
            kzLg_interp2, kxLg_interp2 = np.meshgrid(kzL2, kxL2, indexing='ij')

        else:
            # [vmin, vmax] = [0, 9.2e13]
            [vmin, vmax] = [0, 1e18]
            axes[0].set_xlim([-0.1, 0.1])
            axes[0].set_ylim([-0.01, 0.01])
            axes[1].set_xlim([-0.1, 0.1])
            axes[1].set_ylim([-0.01, 0.01])
            axes[2].set_xlim([-0.1, 0.1])
            axes[2].set_ylim([-0.01, 0.01])

            # ax1.set_xlim([-0.1, 0.1])
            # ax1.set_ylim([-0.01, 0.01])
            kzL = np.linspace(-0.1, 0.1, int(0.1 * npoints))
            kxL = np.linspace(0, 0.01, int(0.1 * npoints))
            kzLg_interp, kxLg_interp = np.meshgrid(kzL, kxL, indexing='ij')

            kzL2 = np.linspace(-0.1, 0.1, int(0.1 * npoints))
            kxL2 = np.linspace(-0.01, 0.01, int(0.2 * npoints))
            kzLg_interp2, kxLg_interp2 = np.meshgrid(kzL2, kxL2, indexing='ij')

        Bk_Lg = interpolate.griddata((kzg_interp.flatten(), kxg_interp.flatten()), Bk_interp_vals.flatten(), (kzLg_interp, kxLg_interp), method='cubic')

        PhDen = ((1 / Nph) * np.abs(Bk_interp_vals)**2).real.astype(float)
        PhDen_Lg = ((1 / Nph) * np.abs(Bk_Lg)**2).real.astype(float)

        Bk_Lg2 = interpolate.griddata((kzg_interp2.flatten(), kxg_interp2.flatten()), Bk_interp2_vals.flatten(), (kzLg_interp2, kxLg_interp2), method='cubic')
        PhDen_Lg2 = ((1 / Nph) * np.abs(Bk_Lg2)**2).real.astype(float)

        quad = axes[0].pcolormesh(kzLg_interp, kxLg_interp, PhDen_Lg[:-1, :-1], vmin=vmin, vmax=vmax)
        quadm = axes[0].pcolormesh(kzLg_interp, -1 * kxLg_interp, PhDen_Lg[:-1, :-1], vmin=vmin, vmax=vmax)
        fig.colorbar(quad, ax=axes[0], extend='both')
        quad1 = axes[1].pcolormesh(kzg_interp, kxg_interp, PhDen[:-1, :-1], vmin=vmin, vmax=vmax)
        quad1m = axes[1].pcolormesh(kzg_interp, -1 * kxg_interp, PhDen[:-1, :-1], vmin=vmin, vmax=vmax)
        fig.colorbar(quad1, ax=axes[1], extend='both')

        quad3 = axes[2].pcolormesh(kzLg_interp2, kxLg_interp2, PhDen_Lg2[:-1, :-1], vmin=vmin, vmax=vmax)
        fig.colorbar(quad3, ax=axes[2], extend='both')

        # ax1.scatter(kzg_interp, kxg_interp, c='b')
        # ax1.scatter(kzg_interp, -1 * kxg_interp, c='b')
        # ax1.scatter(kzLg_interp, kxLg_interp, c='r')
        # ax1.scatter(kzLg_interp, -1 * kxLg_interp, c='r')

        fig2, ax2 = plt.subplots()

        dkz = kzL[1] - kzL[0]
        dkx = kxL[1] - kxL[0]
        zL = np.fft.fftshift(np.fft.fftfreq(kzL.size) * 2 * np.pi / dkz)
        xL = np.fft.fftshift(np.fft.fftfreq(kxL.size) * 2 * np.pi / dkx)
        zLg, xLg = np.meshgrid(zL, xL, indexing='ij')
        dzL = zL[1] - zL[0]
        dxL = xL[1] - xL[0]

        # print(Bk_Lg[np.logical_not(np.isnan(Bk_Lg))])
        Bk_Lg[np.isnan(Bk_Lg)] = 0
        beta_kzkx = np.fft.ifftshift(Bk_Lg)
        amp_beta_zx_preshift = np.fft.ifftn(beta_kzkx) / (dzL * dxL)
        amp_beta_zx = np.fft.fftshift(amp_beta_zx_preshift)
        nzx = ((1 / Nph) * np.abs(amp_beta_zx)**2).real.astype(float)
        quad2 = ax2.pcolormesh(zLg, xLg, nzx, vmin=np.min(nzx), vmax=np.max(nzx))
        # ax2.set_xlim([-10, 10])
        # ax2.set_ylim([-10, 10])
        fig2.colorbar(quad2, ax=ax2, extend='both')

        fig3, ax3 = plt.subplots()

        dkz2 = kzL2[1] - kzL2[0]
        dkx2 = kxL2[1] - kxL2[0]
        zL2 = np.fft.fftshift(np.fft.fftfreq(kzL2.size) * 2 * np.pi / dkz2)
        xL2 = np.fft.fftshift(np.fft.fftfreq(kxL2.size) * 2 * np.pi / dkx2)
        zLg2, xLg2 = np.meshgrid(zL2, xL2, indexing='ij')
        dzL2 = zL2[1] - zL2[0]
        dxL2 = xL2[1] - xL2[0]

        # print(Bk_Lg[np.logical_not(np.isnan(Bk_Lg))])
        Bk_Lg2[np.isnan(Bk_Lg2)] = 0
        beta_kzkx2 = np.fft.ifftshift(Bk_Lg2)
        amp_beta_zx_preshift2 = np.fft.ifftn(beta_kzkx2) / (dzL2 * dxL2)
        amp_beta_zx2 = np.fft.fftshift(amp_beta_zx_preshift2)
        nzx2 = ((1 / Nph) * np.abs(amp_beta_zx2)**2).real.astype(float)
        quad3 = ax3.pcolormesh(zLg2, xLg2, nzx2, vmin=0, vmax=np.max(nzx2))

        # zZ = np.linspace(-4e3, 4e3, 5 * zL2.size)
        # xZ = np.linspace(-60e3, 60e3, 5 * xL2.size)
        # zZg, xZg = np.meshgrid(zZ, xZ, indexing='ij')
        # nzxZ = interpolate.griddata((zLg2.flatten(), xLg2.flatten()), nzx2.flatten(), (zZg, xZg), method='cubic')
        # quad3 = ax3.pcolormesh(zZg, xZg, nzxZ, vmin=0, vmax=np.max(nzxZ))

        # ax3.set_xlim([-20, 20])
        # ax3.set_ylim([-20, 20])
        fig3.colorbar(quad3, ax=ax3, extend='both')

        print(P)
        plt.show()
        break

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
