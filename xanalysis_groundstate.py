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
from timeit import default_timer as timer


if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # ---- INITIALIZE GRIDS ----

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.375, 0.375, 0.375)
    (Lx, Ly, Lz) = (105, 105, 105)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'imaginary', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon', 'Longtime': 'false', 'ReducedInterp': 'false', 'kGrid_ext': 'false'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, 1)
        animpath = '/home/kis/Dropbox/VariationalResearch/DataAnalysis/figs'
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, 1)
        animpath = '/media/kis/Storage/Dropbox/VariationalResearch/DataAnalysis/figs'
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/regal/demler_lab/kis/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, 1)
        animpath = ''

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

    # # # Concatenate Individual Datasets (aIBi specific)

    # aIBi_List = [-10, -5, -2]
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

    # # Analysis of Total Dataset
    interpdatapath = innerdatapath + '/interp'
    aIBi = -10
    # qds = xr.open_dataset(innerdatapath + '/quench_Dataset.nc')
    # qds_aIBi = qds.sel(aIBi=aIBi)
    qds = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    qds_aIBi = qds

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

    # # # Nph (SPHERICAL)

    # Nph_ds = qds_aIBi['Nph']
    # Nph_Vals = np.zeros((PVals.size, tVals.size))
    # for Pind, P in enumerate(PVals):
    #     for tind, t in enumerate(tVals):
    #         Nph_Vals[Pind, tind] = Nph_ds.sel(P=P, t=t).values

    # fig, ax = plt.subplots()
    # ax.plot(PVals, Nph_Vals[:, -1], 'k-')
    # ax.set_title('Phonon Number (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # ax.set_xlabel('P')
    # ax.set_ylabel(r'$N_{ph}$')
    # plt.show()

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

    # # # ENERGY CHARACTERIZATION MULTIPLE INTERACTION STRENGTHS (SPHERICAL)

    # aIBi_Vals = np.array([-10, -5, -2])
    # colorList = ['b', 'g', 'r']
    # fig, ax = plt.subplots()
    # for aind, aIBi in enumerate(aIBi_Vals):
    #     qds_aIBi = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    #     CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
    #     kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)

    #     Energy_Vals_inf = np.zeros(PVals.size)
    #     for Pind, P in enumerate(PVals):
    #         CSAmp = CSAmp_ds.sel(P=P).isel(t=-1).values
    #         Energy_Vals_inf[Pind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)

    #     Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)
    #     Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 5 * PVals.size)
    #     Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
    #     Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)
    #     ax.plot(Pinf_Vals, Einf_2ndderiv_Vals, color=colorList[aind], linestyle='', marker='o', label=r'$a_{IB}^{-1}=$' + '{:.1f}'.format(aIBi))
    # ax.legend()
    # ax.set_title('2nd Derivative of Ground State Energy')
    # ax.set_xlabel('P')
    # plt.show()

    # # # POLARON SOUND VELOCITY (SPHERICAL)

    # # Check to see if linear part of polaron (total system) energy spectrum has slope equal to sound velocity

    # aIBi_Vals = np.array([-10, -5, -2])
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
    # aIBi_Vals = np.array([-10, -5, -2])
    # colorList = ['b', 'g', 'r']
    # for aind, aIBi in enumerate(aIBi_Vals):
    #     qds_aIBi = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
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

    #     # # try fitting an exponential curve (also tried polynomial) to end of velocity vs time curves to determine saturated value
    #     # vstr_fit = np.polyfit(tVals[-5:], np.log(vsound_Vals[-5:]), deg=1)
    #     # vItr_fit = np.polyfit(tVals[-5:], np.log(vI_Vals[-5:]), deg=1)
    #     # print(vstr_fit, vItr_fit)
    #     # print((vsound_Vals[-1] - nu) / nu, (np.exp(vstr_fit[1]) - nu) / nu)
    #     # print((vI_Vals[-1] - nu) / nu, (np.exp(vItr_fit[1]) - nu) / nu)
    #     # # ax.plot(tVals, np.exp(np.poly1d(vstr_fit)(np.log(tVals))), color=colorList[aind], linestyle='-', label='')
    #     # # ax.plot(tVals, np.exp(np.poly1d(vItr_fit)(np.log(tVals))), color=colorList[aind], linestyle='--', label='')

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

    # fig1a, ax1a = plt.subplots()
    # ax1a.plot(PVals, kAve_Vals, 'b-', label='Mean')
    # ax1a.plot(PVals, kFWHM_Vals, 'g-', label='FWHM')
    # ax1a.legend()
    # ax1a.set_xlabel('P')
    # ax1a.set_title('Characteristics of ' + r'$|\vec{k}|$' + ' Distribution of Individual Phonons (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))

    # fig1b, ax1b = plt.subplots()
    # ax1b.plot(PVals, thAve_Vals, 'b-', label='Mean')
    # ax1b.plot(PVals, thFWHM_Vals, 'g-', label='FWHM')
    # ax1b.legend()
    # ax1b.set_xlabel('P')
    # ax1b.set_title('Characteristics of ' + r'$\theta$' + ' Distribution of Individual Phonons (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))

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
    # # anim2.save(animpath + '/aIBi_{0}'.format(aIBi) + '_PhononDist_kmag.gif', writer='imagemagick')

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
    # # anim3.save(animpath + '/aIBi_{0}'.format(aIBi) + '_PhononDist_theta.gif', writer='imagemagick')

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

    # # # FULL RECONSTRUCTION OF 3D CARTESIAN BETA_K FROM 2D SPHERICAL BETA_K (doing actual interpolation in 2D spherical instead of 3D nonlinear cartesian)

    # CSAmp_ds = (qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']).isel(t=-1)
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')

    # Pind = 2
    # P = PVals[Pind]
    # print('P: {0}'.format(P))
    # print('dk: {0}'.format(kVec[1] - kVec[0]))

    # CSAmp_Vals = CSAmp_ds.sel(P=P).values
    # Nph = qds_aIBi.isel(t=-1).sel(P=P)['Nph'].values
    # Bk_2D_vals = CSAmp_Vals.reshape((len(kVec), len(thVec)))

    # kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # kxg_Sph = kg * np.sin(thg)
    # kzg_Sph = kg * np.cos(thg)
    # # Normalization of the original data array - this checks out
    # dk = kg[1, 0] - kg[0, 0]
    # dth = thg[0, 1] - thg[0, 0]
    # PhDen_Sph = ((1 / Nph) * np.abs(Bk_2D_vals)**2).real.astype(float)
    # Bk_norm = np.sum(dk * dth * (2 * np.pi)**(-2) * kg**2 * np.sin(thg) * PhDen_Sph)
    # print('Original (1/Nph)|Bk|^2 normalization (Spherical 2D): {0}'.format(Bk_norm))

    # # Set reduced bounds of k-space and other things dependent on subsonic or supersonic
    # if P < 0.9:
    #     [vmin, vmax] = [0, 500]
    #     # linDimMajor = 1.5
    #     # linDimMinor = 1.5
    #     linDimMajor = 6.5
    #     linDimMinor = 6.5
    #     ext_major_rat = 0.35
    #     ext_minor_rat = 0.35
    #     poslinDim = 2
    #     Npoints = 400  # actual number of points will be ~Npoints-1
    # else:
    #     # [vmin, vmax] = [0, 9.2e13]
    #     # linDimMajor = 0.1  # For the worse grid value data, there is some dependancy on the final FFT on what range of k we pick...(specifically the z-axis = lindimMajor changing in range 0.1 - 0.4), For better data grid, FFT still vanishes after lindimMajor >=0.4 -> probably due to resolution in k-space not capturing the features
    #     # linDimMinor = 0.01
    #     linDimMajor = 0.2
    #     linDimMinor = 0.02
    #     ext_major_rat = 0.025
    #     ext_minor_rat = 0.0025
    #     poslinDim = 2500
    #     Npoints = 400

    # # Remove k values outside reduced k-space bounds (as |Bk|~0 there) and save the average of these values to add back in later before FFT
    # if toggleDict['ReducedInterp'] == 'true':
    #     kred_ind = np.argwhere(kg[:, 0] > (1.5 * linDimMajor))[0][0]
    #     kg_red = np.delete(kg, np.arange(kred_ind, kVec.size), 0)
    #     thg_red = np.delete(thg, np.arange(kred_ind, kVec.size), 0)
    #     Bk_red = np.delete(Bk_2D_vals, np.arange(kred_ind, kVec.size), 0)
    #     Bk_remainder = np.delete(Bk_2D_vals, np.arange(0, kred_ind), 0)
    #     Bk_rem_ave = np.average(Bk_remainder)
    #     kVec_red = kVec[0:kred_ind]
    #     kmax_rem = np.max(kVec)
    #     kVec = kVec_red
    #     kg = kg_red
    #     thg = thg_red
    #     Bk_2D_vals = Bk_red

    # # CHECK WHY ALL BK AMPLITUDES HAVE ZERO IMAGINARY PART, EVEN FOR SUPERSONIC CASE? IS THIS BECAUSE ITS THE GROUNDSTATE?
    # # print(np.imag(Bk_2D_vals))

    # # Create linear 3D cartesian grid and reinterpolate Bk_3D onto this grid
    # kxL_pos, dkxL = np.linspace(1e-10, linDimMinor, Npoints // 2, retstep=True, endpoint=False); kxL = np.concatenate((1e-10 - 1 * np.flip(kxL_pos[1:], axis=0), kxL_pos))
    # kyL_pos, dkyL = np.linspace(1e-10, linDimMinor, Npoints // 2, retstep=True, endpoint=False); kyL = np.concatenate((1e-10 - 1 * np.flip(kyL_pos[1:], axis=0), kyL_pos))
    # kzL_pos, dkzL = np.linspace(1e-10, linDimMajor, Npoints // 2, retstep=True, endpoint=False); kzL = np.concatenate((1e-10 - 1 * np.flip(kzL_pos[1:], axis=0), kzL_pos))
    # kxLg_3D, kyLg_3D, kzLg_3D = np.meshgrid(kxL, kyL, kzL, indexing='ij')

    # # Re-interpret grid points of linear 3D Cartesian as nonlinear 3D spherical grid, find unique (k,th) points
    # kg_3Di = np.sqrt(kxLg_3D**2 + kyLg_3D**2 + kzLg_3D**2)
    # thg_3Di = np.arccos(kzLg_3D / kg_3Di)
    # phig_3Di = np.arctan2(kyLg_3D, kxLg_3D)

    # kg_3Di_flat = kg_3Di.reshape(kg_3Di.size)
    # thg_3Di_flat = thg_3Di.reshape(thg_3Di.size)
    # tups_3Di = np.column_stack((kg_3Di_flat, thg_3Di_flat))
    # tups_3Di_unique, tups_inverse = np.unique(tups_3Di, return_inverse=True, axis=0)

    # # Perform interpolation on 2D projection and reconstruct full matrix on 3D linear cartesian grid
    # print('3D Cartesian grid Ntot: {:1.2E}'.format(kzLg_3D.size))
    # print('Unique interp points: {:1.2E}'.format(tups_3Di_unique[:, 0].size))
    # interpstart = timer()
    # Bk_2D_CartInt = interpolate.griddata((kg.flatten(), thg.flatten()), Bk_2D_vals.flatten(), tups_3Di_unique, method='cubic')
    # # Bk_2D_Rbf = interpolate.Rbf(kg, thg, Bk_2D.values)
    # # Bk_2D_CartInt = Bk_2D_Rbf(tups_3Di_unique)
    # interpend = timer()
    # print('Interp Time: {0}'.format(interpend - interpstart))
    # BkLg_3D_flat = Bk_2D_CartInt[tups_inverse]
    # BkLg_3D = BkLg_3D_flat.reshape(kg_3Di.shape)

    # BkLg_3D[np.isnan(BkLg_3D)] = 0
    # PhDenLg_3D = ((1 / Nph) * np.abs(BkLg_3D)**2).real.astype(float)
    # BkLg_3D_norm = np.sum(dkxL * dkyL * dkzL * (2 * np.pi)**(-3) * PhDenLg_3D)
    # print('Interpolated (1/Nph)|Bk|^2 normalization (Linear Cartesian 3D): {0}'.format(BkLg_3D_norm))

    # # Add the remainder of Bk back in (values close to zero for large k) (Note: can also do this more easily by setting a fillvalue in griddata and interpolating)
    # if toggleDict['ReducedInterp'] == 'true' and toggleDict['kGrid_ext'] == 'true':
    #     kL_max_major = ext_major_rat * kmax_rem / np.sqrt(2)
    #     kL_max_minor = ext_minor_rat * kmax_rem / np.sqrt(2)
    #     print('kL_red_max_major: {0}, kL_ext_max_major: {1}, dkL_major: {2}'.format(np.max(kzL), kL_max_major, dkzL))
    #     print('kL_red_max_minor: {0}, kL_ext_max_minor: {1}, dkL_minor: {2}'.format(np.max(kxL), kL_max_minor, dkxL))
    #     kx_addon = np.arange(linDimMinor, kL_max_minor, dkxL); ky_addon = np.arange(linDimMinor, kL_max_minor, dkyL); kz_addon = np.arange(linDimMajor, kL_max_major, dkzL)
    #     print('kL_ext_addon size -  major: {0}, minor: {1}'.format(2 * kz_addon.size, 2 * kx_addon.size))
    #     kxL_ext = np.concatenate((1e-10 - 1 * np.flip(kx_addon, axis=0), np.concatenate((kxL, kx_addon))))
    #     kyL_ext = np.concatenate((1e-10 - 1 * np.flip(ky_addon, axis=0), np.concatenate((kyL, kx_addon))))
    #     kzL_ext = np.concatenate((1e-10 - 1 * np.flip(kz_addon, axis=0), np.concatenate((kzL, kx_addon))))

    #     ax = kxL.size; ay = kyL.size; az = kzL.size
    #     mx = kx_addon.size; my = ky_addon.size; mz = kz_addon.size

    #     BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones((mz, ax, ay)), np.concatenate((BkLg_3D, Bk_rem_ave * np.ones((mx, ay, az))), axis=0)), axis=0)
    #     BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones(((az + 2 * mz), mx, ay)), np.concatenate((BkLg_3D_ext, Bk_rem_ave * np.ones(((ax + 2 * mx), my, az))), axis=1)), axis=1)
    #     BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones(((az + 2 * mz), (ax + 2 * mx), my)), np.concatenate((BkLg_3D_ext, Bk_rem_ave * np.ones(((ax + 2 * mx), (ay + 2 * my), mz))), axis=2)), axis=2)

    #     kxL = kxL_ext; kyL = kyL_ext; kzL = kzL_ext
    #     BkLg_3D = BkLg_3D_ext
    #     print('Cartesian Interp Extended Grid Shape: {0}'.format(BkLg_3D.shape))

    # # Fourier Transform to get 3D position distribution
    # xL = np.fft.fftshift(np.fft.fftfreq(kxL.size) * 2 * np.pi / dkxL)
    # yL = np.fft.fftshift(np.fft.fftfreq(kyL.size) * 2 * np.pi / dkyL)
    # zL = np.fft.fftshift(np.fft.fftfreq(kzL.size) * 2 * np.pi / dkzL)
    # dxL = xL[1] - xL[0]; dyL = yL[1] - yL[0]; dzL = zL[1] - zL[0]
    # dVxyz = dxL * dyL * dzL
    # # print(dzL, 2 * np.pi / (kzL.size * dkzL))

    # xLg_3D, yLg_3D, zLg_3D = np.meshgrid(xL, yL, zL, indexing='ij')
    # beta_kxkykz = np.fft.ifftshift(BkLg_3D)
    # amp_beta_xyz_preshift = np.fft.ifftn(beta_kxkykz) / dVxyz
    # amp_beta_xyz = np.fft.fftshift(amp_beta_xyz_preshift)
    # nxyz = ((1 / Nph) * np.abs(amp_beta_xyz)**2).real.astype(float)
    # nxyz_norm = np.sum(dVxyz * nxyz)
    # print('Linear grid (1/Nph)*n(x,y,z) normalization (Cartesian 3D): {0}'.format(nxyz_norm))

    # # Calculate real space distribution of atoms in the BEC
    # uk2 = 0.5 * (1 + (pfc.epsilon(kxLg_3D, kyLg_3D, kzLg_3D, mB) + gBB * n0) / pfc.omegak(kxLg_3D, kyLg_3D, kzLg_3D, mB, n0, gBB))
    # vk2 = uk2 - 1
    # uk = np.sqrt(uk2); vk = np.sqrt(vk2)

    # uB_kxkykz = np.fft.ifftshift(uk * BkLg_3D)
    # uB_xyz = np.fft.fftshift(np.fft.ifftn(uB_kxkykz) / dVxyz)
    # vB_kxkykz = np.fft.ifftshift(vk * BkLg_3D)
    # vB_xyz = np.fft.fftshift(np.fft.ifftn(vB_kxkykz) / dVxyz)
    # # na_xyz = np.sum(vk2 * dkxL * dkyL * dkzL) + np.abs(uB_xyz - np.conjugate(vB_xyz))**2
    # na_xyz = np.abs(uB_xyz - np.conjugate(vB_xyz))**2
    # na_xyz_norm = na_xyz / np.sum(na_xyz * dVxyz)
    # print(np.sum(vk2 * dkxL * dkyL * dkzL), np.max(np.abs(uB_xyz - np.conjugate(vB_xyz))**2))

    # # # Create DataSet for 3D Betak and position distribution slices
    # # PhDen_da = xr.DataArray(PhDenLg_3D, coords=[kxL, kyL, kzL], dims=['kx', 'ky', 'kz'])
    # # nxyz_da = xr.DataArray(nxyz, coords=[xL, yL, zL], dims=['x', 'y', 'z'])

    # # data_dict = {'PhDen': PhDen_da, 'nxyz': nxyz_da}
    # # coords_dict = {'kx': kxL, 'ky': kyL, 'kz': kzL, 'x': xL, 'y': yL, 'z': zL}
    # # attrs_dict = {'P': P, 'aIBi': aIBi}
    # # interp_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    # # interp_ds.to_netcdf(interpdatapath + '/InterpDat_P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    # # Consistency check: use 2D ky=0 slice of |Bk|^2 to calculate phonon density and compare it to phonon density from original spherical interpolated data
    # kxL_0ind = kxL.size // 2; kyL_0ind = kyL.size // 2; kzL_0ind = kzL.size // 2  # find position of zero of each axis: kxL=0, kyL=0, kzL=0
    # kxLg_ky0slice = kxLg_3D[:, kyL_0ind, :]
    # kzLg_ky0slice = kzLg_3D[:, kyL_0ind, :]
    # PhDenLg_ky0slice = PhDenLg_3D[:, kyL_0ind, :]

    # # Take 2D slices of position distribution
    # zLg_y0slice = zLg_3D[:, yL.size // 2, :]
    # xLg_y0slice = xLg_3D[:, yL.size // 2, :]
    # nxyz_y0slice = nxyz[:, yL.size // 2, :]

    # # Interpolate 2D slice of position distribution
    # posmult = 5
    # zL_y0slice_interp = np.linspace(-1 * poslinDim, poslinDim, posmult * zL.size); xL_y0slice_interp = np.linspace(-1 * poslinDim, poslinDim, posmult * xL.size)
    # xLg_y0slice_interp, zLg_y0slice_interp = np.meshgrid(xL_y0slice_interp, zL_y0slice_interp, indexing='ij')
    # nxyz_y0slice_interp = interpolate.griddata((xLg_y0slice.flatten(), zLg_y0slice.flatten()), nxyz_y0slice.flatten(), (xLg_y0slice_interp, zLg_y0slice_interp), method='cubic')

    # # Take 2D slices of atom position distribution and interpolate
    # na_xyz_y0slice = na_xyz_norm[:, yL.size // 2, :]
    # na_xyz_y0slice_interp = interpolate.griddata((xLg_y0slice.flatten(), zLg_y0slice.flatten()), na_xyz_y0slice.flatten(), (xLg_y0slice_interp, zLg_y0slice_interp), method='cubic')

    # # All Plotting: (a) 2D ky=0 slice of |Bk|^2, (b) 2D slice of position distribution

    # # if P > 0.9:
    # #     vmax = np.max(PhDen_Sph)
    # # vmax = np.max(PhDen_Sph)
    # # vmin = 1e-16

    # fig1, ax1 = plt.subplots()
    # quad1 = ax1.pcolormesh(kzg_Sph, kxg_Sph, PhDen_Sph[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(PhDen_Sph)), vmax=np.max(PhDen_Sph)), cmap='inferno')
    # quad1m = ax1.pcolormesh(kzg_Sph, -1 * kxg_Sph, PhDen_Sph[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(PhDen_Sph)), vmax=np.max(PhDen_Sph)), cmap='inferno')
    # ax1.set_xlim([-1 * linDimMajor, linDimMajor])
    # ax1.set_ylim([-1 * linDimMinor, linDimMinor])
    # ax1.set_xlabel('kz (Impurity Propagation Direction)')
    # ax1.set_ylabel('kx')
    # ax1.set_title('Individual Phonon Momentum Distribution (Data)')
    # fig1.colorbar(quad1, ax=ax1, extend='both')

    # fig2, ax2 = plt.subplots()
    # quad2 = ax2.pcolormesh(kzLg_ky0slice, kxLg_ky0slice, PhDenLg_ky0slice[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(PhDen_Sph)), vmax=np.max(PhDen_Sph)), cmap='inferno')
    # ax2.set_xlim([-1 * linDimMajor, linDimMajor])
    # ax2.set_ylim([-1 * linDimMinor, linDimMinor])
    # # quad2 = ax2.pcolormesh(kzLg_ky0slice, kxLg_ky0slice, PhDenLg_ky0slice[:-1, :-1], norm=colors.LogNorm(vmin=1, vmax=np.max(PhDen_Sph)), cmap='inferno')
    # # ax2.set_xlim([-1 * 0.75, 0.75])
    # # ax2.set_ylim([-1 * 0.75, 0.75])
    # ax2.set_xlabel('kz (Impurity Propagation Direction)')
    # ax2.set_ylabel('kx')
    # ax2.set_title('Individual Phonon Momentum Distribution (Interp)')
    # fig2.colorbar(quad2, ax=ax2, extend='both')

    # fig3, ax3 = plt.subplots()
    # quad3 = ax3.pcolormesh(zLg_y0slice_interp, xLg_y0slice_interp, nxyz_y0slice_interp[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(nxyz_y0slice_interp)), vmax=np.max(nxyz_y0slice_interp)), cmap='inferno')
    # ax3.set_xlabel('z (Impurity Propagation Direction)')
    # ax3.set_ylabel('x')
    # ax3.set_title('Individual Phonon Position Distribution (Interp)')
    # fig3.colorbar(quad3, ax=ax3, extend='both')

    # fig4, ax4 = plt.subplots()
    # quad4 = ax4.pcolormesh(zLg_y0slice_interp, xLg_y0slice_interp, na_xyz_y0slice_interp[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(na_xyz_y0slice_interp)), vmax=np.max(na_xyz_y0slice_interp)), cmap='inferno')
    # ax4.set_xlabel('z (Impurity Propagation Direction)')
    # ax4.set_ylabel('x')
    # ax4.set_title('Individual Atom Position Distribution (Interp)')
    # fig4.colorbar(quad4, ax=ax4, extend='both')

    # # fig3, ax3 = plt.subplots()
    # # quad3 = ax3.pcolormesh(zLg_y0slice_interp, xLg_y0slice_interp, nxyz_y0slice_interp[:-1, :-1], norm=colors.SymLogNorm(linthresh=0.01, vmin=np.min(nxyz_y0slice_interp), vmax=np.max(nxyz_y0slice_interp)), cmap='inferno')
    # # ax3.set_xlabel('z (Impurity Propagation Direction)')
    # # ax3.set_ylabel('x')
    # # ax3.set_title('Individual Phonon Position Distribution (Interp)')
    # # fig3.colorbar(quad3, ax=ax3, extend='both')

    # # fig4, ax4 = plt.subplots()
    # # quad4 = ax4.pcolormesh(zLg_y0slice_interp, xLg_y0slice_interp, na_xyz_y0slice_interp[:-1, :-1], norm=colors.SymLogNorm(linthresh=0.01, vmin=np.min(na_xyz_y0slice_interp), vmax=np.max(na_xyz_y0slice_interp)), cmap='inferno')
    # # ax4.set_xlabel('z (Impurity Propagation Direction)')
    # # ax4.set_ylabel('x')
    # # ax4.set_title('Individual Atom Position Distribution (Interp)')
    # # fig4.colorbar(quad4, ax=ax4, extend='both')

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
