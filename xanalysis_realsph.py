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
from timeit import default_timer as timer


if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)
    # (dx, dy, dz) = (0.25, 0.25, 0.25)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    # NGridPoints_cart = 1.37e5

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'real', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon', 'Longtime': 'false'}

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

    if toggleDict['Longtime'] == 'true':
        innerdatapath = innerdatapath + '_longtime'
    elif toggleDict['Longtime'] == 'false':
        innerdatapath = innerdatapath

    # # # Concatenate Individual Datasets

    # ds_list = []; P_list = []; aIBi_list = []; mI_list = []
    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     if filename == 'quench_Dataset.nc':
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
    # ds_tot.to_netcdf(innerdatapath + '/quench_Dataset.nc')

    # # # Analysis of Total Dataset

    qds = xr.open_dataset(innerdatapath + '/quench_Dataset.nc')
    # # qds = xr.open_dataset(innerdatapath + '/P_0.900_aIBi_-6.23.nc')
    # tVals = qds['t'].values
    # dt = tVals[1] - tVals[0]
    # PVals = qds['P'].values
    # aIBiVals = qds.coords['aIBi'].values
    # n0 = qds.attrs['n0']
    # gBB = qds.attrs['gBB']
    # nu = pfc.nu(gBB)
    # mI = qds.attrs['mI']
    # mB = qds.attrs['mB']

    # # aIBi = -10
    # # qds_aIBi = qds.sel(aIBi=aIBi)

    # # fig, ax = plt.subplots()
    # # for P in PVals:
    # #     Nph = qds_aIBi.sel(P=P)['Nph'].values
    # #     dNph = np.diff(Nph)
    # #     ax.plot(dNph / dt)

    # plt.show()

    # # # INDIVDUAL PHONON MOMENTUM DISTRIBUTION DATASET CREATION

    # start1 = timer()

    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds.coords['k'].values); kgrid.initArray_premade('th', qds.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # list_of_unit_vectors = list(kgrid.arrays.keys())
    # list_of_functions = [lambda k: (2 * np.pi)**(-2) * k**2, np.sin]
    # # sphfac = kgrid.function_prod(list_of_unit_vectors, list_of_functions)
    # sphfac = 1
    # kDiff = kgrid.diffArray('k')
    # thDiff = kgrid.diffArray('th')
    # dkMat, dthMat = np.meshgrid(kDiff, thDiff, indexing='ij')
    # dkMat_flat = dkMat.reshape(dkMat.size)
    # dthMat_flat = dthMat.reshape(dthMat.size)

    # nk_da = xr.DataArray(np.full((aIBiVals.size, PVals.size, tVals.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[aIBiVals, PVals, tVals, kVec, thVec], dims=['aIBi', 'P', 't', 'k', 'th'])
    # Delta_nk_da = xr.DataArray(np.full((aIBiVals.size, PVals.size, tVals.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[aIBiVals, PVals, tVals, kVec, thVec], dims=['aIBi', 'P', 't', 'k', 'th'])
    # for aind, aIBi in enumerate(aIBiVals):
    #     for Pind, P in enumerate(PVals):
    #         start2 = timer()
    #         for tind, t in enumerate(tVals):
    #             CSAmp = (qds.sel(aIBi=aIBi, P=P, t=t)['Real_CSAmp'].values + 1j * qds.sel(aIBi=aIBi, P=P, t=t)['Imag_CSAmp'].values); CSAmp_flat = CSAmp.reshape(CSAmp.size)
    #             Nph = qds.sel(aIBi=aIBi, P=P, t=t)['Nph'].values
    #             PhDen = (1 / Nph) * sphfac * np.abs(CSAmp_flat)**2
    #             Delta_CSAmp_flat = pfs.CSAmp_timederiv(CSAmp_flat, kgrid, P, aIBi, mI, mB, n0, gBB)
    #             eta = (1 / Nph) * sphfac * (np.conj(CSAmp_flat) * Delta_CSAmp_flat + np.conj(Delta_CSAmp_flat) * CSAmp_flat)
    #             Delta_PhDen = eta - PhDen * np.sum(eta * dkMat_flat * dthMat_flat)

    #             nk_da.sel(aIBi=aIBi, P=P, t=t)[:] = PhDen.reshape((len(kVec), len(thVec))).real.astype(float)
    #             Delta_nk_da.sel(aIBi=aIBi, P=P, t=t)[:] = Delta_PhDen.reshape((len(kVec), len(thVec))).real.astype(float)
    #         print('aIBi = {0},P = {1}, Time = {2}'.format(aIBi, P, timer() - start2))

    # data_dict = {'nk_ind': nk_da, 'Delta_nk_ind': Delta_nk_da}
    # coords_dict = {'aIBi': aIBiVals, 'P': PVals, 't': tVals}
    # attrs_dict = qds.attrs
    # nk_ind_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    # nk_ind_ds.to_netcdf(innerdatapath + '/nk_ind_Dataset.nc')
    # end = timer()
    # print('Total time: {0}'.format(end - start1))

    # # INDIVDUAL PHONON MOMENTUM DISTRIBUTION DATASET ANALYSIS

    nk_ds = xr.open_dataset(innerdatapath + '/nk_ind_Dataset.nc')
    # nk_ds = xr.open_dataset(innerdatapath + '/nk_ind_Dataset_withjac.nc')
    tVals = nk_ds['t'].values
    dt = tVals[1] - tVals[0]
    PVals = nk_ds['P'].values
    aIBiVals = nk_ds.coords['aIBi'].values
    n0 = nk_ds.attrs['n0']
    gBB = nk_ds.attrs['gBB']
    nu = pfc.nu(gBB)
    mI = nk_ds.attrs['mI']
    mB = nk_ds.attrs['mB']
    aBB = gBB * mB / (4 * np.pi)
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)
    tscale = xi / nu

    kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', nk_ds.coords['k'].values); kgrid.initArray_premade('th', nk_ds.coords['th'].values)
    kVec = kgrid.getArray('k')
    thVec = kgrid.getArray('th')

    # kMat, thMat = np.meshgrid(kVec, thVec, indexing='ij')
    # xMat = kMat * np.sin(thMat)
    # zMat = kMat * np.cos(thMat)

    # # SINGLE PLOT

    # aIBi = -10
    # Pind = -3
    # tind = 100

    # nk = nk_ds.sel(aIBi=aIBi).isel(P=Pind, t=tind)['nk_ind']
    # Delta_nk = nk_ds.sel(aIBi=aIBi).isel(P=Pind, t=tind)['Delta_nk_ind']
    # Delta_nk_2 = (nk_ds.sel(aIBi=aIBi).isel(P=Pind, t=tind)['nk_ind'] - nk_ds.sel(aIBi=aIBi).isel(P=Pind, t=tind - 1)['nk_ind']) / dt

    # nk_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(nk, 'k', 'th', 5)
    # xg_interp = kg_interp * np.sin(thg_interp)
    # zg_interp = kg_interp * np.cos(thg_interp)

    # fig1, ax1 = plt.subplots()
    # quad1 = ax1.pcolormesh(zg_interp, xg_interp, nk_interp_vals)
    # ax1.pcolormesh(zg_interp, -1 * xg_interp, nk_interp_vals)
    # ax1.set_title('aIBi={0}, P={1}, t={2}'.format(aIBi, PVals[Pind], tVals[tind]))
    # ax1.set_xlim([-3, 3])
    # ax1.set_ylim([-3, 3])
    # fig1.colorbar(quad1, ax=ax1, extend='both')

    # # fig2, ax2 = plt.subplots()
    # # quad2 = ax2.pcolormesh(zMat, xMat, Delta_nk_2.values)
    # # ax2.pcolormesh(zMat, -1 * xMat, Delta_nk_2.values)
    # # ax2.set_title('aIBi={0}, P={1}, t={2}'.format(aIBi, PVals[Pind], tVals[tind]))
    # # # ax2.set_xlim([-3, 3])
    # # # ax2.set_ylim([-3, 3])
    # # fig2.colorbar(quad2, ax=ax2, extend='both')

    # plt.show()

    # ANIMATIONS

    aIBi = -10
    # P = 0.4
    # P = 1.4
    P = 5

    tmax = 40
    tmax_ind = (np.abs(tVals - tmax)).argmin()

    nk = nk_ds.sel(aIBi=aIBi).sel(P=P, method='nearest')['nk_ind']
    Delta_nk = nk_ds.sel(aIBi=aIBi).sel(P=P, method='nearest')['Delta_nk_ind']

    P = 1 * nk['P'].values

    Pph = qds.sel(aIBi=aIBi).sel(P=P)['Pph'].values
    Pimp = P - Pph

    # Phonon probability

    fig1, ax1 = plt.subplots()
    # vmin = 1
    # vmax = 0
    # for Pind, Pv in enumerate(PVals):
    #     for tind, t in enumerate(tVals):
    #         vec = nk_ds.sel(aIBi=aIBi, P=Pv, t=t)['nk_ind'].values
    #         # vec = nk.sel(t=t).values
    #         if np.min(vec) < vmin:
    #             vmin = np.min(vec)
    #         if np.max(vec) > vmax:
    #             vmax = np.max(vec)

    vmin = 0
    vmax = 700
    # vmax = 4

    nk0_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(nk.isel(t=0), 'k', 'th', 5)
    xg_interp = kg_interp * np.sin(thg_interp)
    zg_interp = kg_interp * np.cos(thg_interp)

    quad1 = ax1.pcolormesh(zg_interp, xg_interp, nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    quad1m = ax1.pcolormesh(zg_interp, -1 * xg_interp, nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    curve1 = ax1.plot(Pph[0], 0, marker='x', markersize=10, color="magenta", label=r'$P_{ph}$')[0]
    curve1m = ax1.plot(Pimp[0], 0, marker='o', markersize=10, color="red", label=r'$P_{imp}$')[0]

    t_text = ax1.text(0.81, 0.9, r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[0] / tscale), transform=ax1.transAxes, color='r')
    # ax1.set_xlim([-2, 2])
    # ax1.set_ylim([-2, 2])
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])
    ax1.legend(loc=2)
    ax1.grid(True, linewidth=0.5)
    ax1.set_title('Ind Phonon Distribution (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$P=$' + '{:.2f})'.format(P))
    ax1.set_xlabel(r'$k_{z}$')
    ax1.set_ylabel(r'$k_{x}$')
    fig1.colorbar(quad1, ax=ax1, extend='both')

    def animate1(i):
        nk_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(nk.isel(t=i), 'k', 'th', 5)
        quad1.set_array(nk_interp_vals[:-1, :-1].ravel())
        quad1m.set_array(nk_interp_vals[:-1, :-1].ravel())
        curve1.set_xdata(Pph[i])
        curve1m.set_xdata(Pimp[i])
        t_text.set_text(r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[i] / tscale))
    anim1 = FuncAnimation(fig1, animate1, interval=1e-5, frames=range(tmax_ind + 1), blit=False)
    anim1.save(animpath + '/aIBi_{:d}_P_{:.2f}'.format(aIBi, P) + '_indPhononDist_2D.gif', writer='imagemagick')

    # Change in phonon probability

    fig2, ax2 = plt.subplots()

    # vmin = 1
    # vmax = 0
    # for Pind, Pv in enumerate(PVals):
    #     for tind, t in enumerate(tVals):
    #         if t == tVals[-1]:
    #             break
    #         vec = nk_ds.sel(aIBi=aIBi, P=Pv).isel(t=tind + 1)['nk_ind'].values - nk_ds.sel(aIBi=aIBi, P=Pv).isel(t=tind)['nk_ind'].values
    #         vec = vec / dt
    #         # vec = nk.sel(t=t).values
    #         if np.min(vec) < vmin:
    #             vmin = np.min(vec)
    #         if np.max(vec) > vmax:
    #             vmax = np.max(vec)

    # vmin = -0.9
    # vmax = 0.82
    # vmax = 0.6
    vmin = -25
    vmax = 110

    dnk0 = (nk.isel(t=1) - nk.isel(t=0)) / dt
    dnk0_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(dnk0, 'k', 'th', 5)
    xg_interp = kg_interp * np.sin(thg_interp)
    zg_interp = kg_interp * np.cos(thg_interp)

    quad2 = ax2.pcolormesh(zg_interp, xg_interp, dnk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    quad2m = ax2.pcolormesh(zg_interp, -1 * xg_interp, dnk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    curve2 = ax2.plot(Pph[0], 0, marker='x', markersize=10, color="magenta", label=r'$P_{ph}$')[0]
    curve2m = ax2.plot(Pimp[0], 0, marker='o', markersize=10, color="red", label=r'$P_{imp}$')[0]
    t_text = ax2.text(0.81, 0.9, r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[0] / tscale), transform=ax2.transAxes, color='r')
    # ax2.set_xlim([-2, 2])
    # ax2.set_ylim([-2, 2])
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    ax2.legend(loc=2)
    ax2.grid(True, linewidth=0.5)
    ax2.set_title('Ind Phonon Distribution Time Derivative (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$P=$' + '{:.2f})'.format(P))
    ax2.set_xlabel(r'$k_{z}$')
    ax2.set_ylabel(r'$k_{x}$')
    fig2.colorbar(quad2, ax=ax2, extend='both')

    def animate2(i):
        dnk = (nk.isel(t=i + 1) - nk.isel(t=i)) / dt
        dnk_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(dnk, 'k', 'th', 5)
        quad2.set_array(dnk_interp_vals[:-1, :-1].ravel())
        quad2m.set_array(dnk_interp_vals[:-1, :-1].ravel())
        curve2.set_xdata(Pph[i])
        curve2m.set_xdata(Pimp[i])
        t_text.set_text(r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[i] / tscale))
    anim2 = FuncAnimation(fig2, animate2, interval=1e-5, frames=range(tmax_ind + 1), blit=False)
    anim2.save(animpath + '/aIBi_{:d}_P_{:.2f}'.format(aIBi, P) + '_indPhononDistDeriv_2D_num.gif', writer='imagemagick')

    # fig3, ax3 = plt.subplots()
    # # vmin = 1
    # # vmax = 0
    # # for Pind, Pv in enumerate(PVals):
    # #     for tind, t in enumerate(tVals):
    # #         vec = nk_ds.sel(aIBi=aIBi, P=Pv, t=t)['nk_ind'].values
    # #         # vec = nk.sel(t=t).values
    # #         if np.min(vec) < vmin:
    # #             vmin = np.min(vec)
    # #         if np.max(vec) > vmax:
    # #             vmax = np.max(vec)

    # vmin = -0.9
    # vmax = 0.82

    # Delta_nk0_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(Delta_nk.isel(t=0), 'k', 'th', 5)
    # xg_interp = kg_interp * np.sin(thg_interp)
    # zg_interp = kg_interp * np.cos(thg_interp)

    # quad3 = ax3.pcolormesh(zg_interp, xg_interp, Delta_nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # quad3m = ax3.pcolormesh(zg_interp, -1 * xg_interp, Delta_nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # curve3 = ax3.plot(Pph[0], 0, marker='x', markersize=10, color="magenta", label=r'$P_{ph}$')[0]
    # curve3m = ax3.plot(Pimp[0], 0, marker='o', markersize=10, color="red", label=r'$P_{imp}$')[0]
    # t_text = ax3.text(0.81, 0.9, r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[0] / tscale), transform=ax3.transAxes, color='r')
    # ax3.set_xlim([-2, 2])
    # ax3.set_ylim([-2, 2])
    # ax3.legend(loc=2)
    # ax3.grid(True, linewidth=0.5)
    # ax3.set_title('Ind Phonon Distribution Time Derivative (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$P=$' + '{:.2f})'.format(P))
    # ax3.set_xlabel(r'$k_{z}$')
    # ax3.set_ylabel(r'$k_{x}$')
    # fig3.colorbar(quad3, ax=ax3, extend='both')

    # def animate3(i):
    #     Delta_nk_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(Delta_nk.isel(t=i), 'k', 'th', 5)
    #     quad3.set_array(Delta_nk_interp_vals[:-1, :-1].ravel())
    #     quad3m.set_array(Delta_nk_interp_vals[:-1, :-1].ravel())
    #     curve3.set_xdata(Pph[i])
    #     curve3m.set_xdata(Pimp[i])
    #     t_text.set_text(r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[i] / tscale))
    # anim3 = FuncAnimation(fig3, animate3, interval=1e-5, frames=range(tmax_ind + 1), blit=False)
    # anim3.save(animpath + '/aIBi_{:d}_P_{:.2f}'.format(aIBi, P) + '_indPhononDistDeriv_2D_an.gif', writer='imagemagick')

    # plt.draw()
    # plt.show()
