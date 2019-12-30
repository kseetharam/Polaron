import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from matplotlib.animation import writers
import os
import itertools
import pf_dynamic_cart as pfc
import pf_dynamic_sph as pfs
import Grid
from scipy import interpolate
from timeit import default_timer as timer


if __name__ == "__main__":

    # # Initialization

    # fm = matplotlib.font_manager.json_load(os.path.expanduser("~/.cache/matplotlib/fontlist-v310.json"))
    # fm.findfont("serif", rebuild_if_missing=False)
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Adobe Garamond Pro']
    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})
    mpegWriter = writers['ffmpeg'](fps=0.75, bitrate=1800)

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

    # # # Concatenate Individual Datasets (aIBi specific)

    # aIBi_List = [-15.0, -12.5, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -3.5, -2.0, -1.0, -0.75, -0.5, -0.1]

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

    # # # Concatenate Individual Datasets (aIBi specific, IRcuts)

    # IRrat_Vals = [1, 2, 5, 10, 50, 1e2, 5e2, 1e3, 5e3, 1e4]
    # aIBi_List = [-10.0, -5.0, -2.0, -0.5]
    # for IRrat in IRrat_Vals:
    #     IRdatapath = innerdatapath + '/IRratio_{:.1E}'.format(IRrat)
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

    # # PHASE DIAGRAM (SPHERICAL)

    Pnormdes = 0.5
    Pind = np.abs(PVals / (mI * nu) - Pnormdes).argmin()
    P = PVals[Pind]

    ZVals = np.zeros(aIBi_Vals.size)
    for aind, aIBi in enumerate(aIBi_Vals):
        qds_aIBi = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
        ZVals[aind] = np.exp(-1 * qds_aIBi.isel(P=Pind, t=-1)['Nph'].values)

    xmin = np.min(aIBi_Vals)
    xmax = 1.01 * np.max(aIBi_Vals)

    fig, ax = plt.subplots()
    ax.plot(aIBi_Vals, ZVals, 'g-')
    ax.set_title('Quasiparticle Residue (' + r'$\frac{P}{m_{I}c_{BEC}}=$' + '{:.2f})'.format(P / (mI * nu)))
    ax.set_xlabel(r'$a_{IB}^{-1}$')
    ax.set_ylabel(r'$Z=e^{-N_{ph}}$')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, 1.1])
    plt.show()

    # # # FIG 1 (first half)

    # # BOGOLIUBOV DISPERSION (SPHERICAL)

    kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds.coords['k'].values); kgrid.initArray_premade('th', qds.coords['th'].values)
    kVals = kgrid.getArray('k')
    wk_Vals = pfs.omegak(kVals, mB, n0, gBB)
    fig, ax = plt.subplots()
    ax.plot(kVals, wk_Vals, 'k-', label='')
    ax.plot(kVals, nu * kVals, 'b--', label=r'$c_{BEC}|k|$')
    ax.set_title('Bogoliubov Phonon Dispersion')
    ax.set_xlabel(r'$|k|$')
    ax.set_ylabel(r'$\omega_{|k|}$')
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 3])
    ax.legend(loc=2, fontsize='x-large')
    plt.show()

    # # PHASE DIAGRAM (SPHERICAL)

    Pcrit = np.zeros(aIBi_Vals.size)
    for aind, aIBi in enumerate(aIBi_Vals):
        qds_aIBi = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
        CSAmp_ds = qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']
        kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)

        Energy_Vals_inf = np.zeros(PVals.size)
        for Pind, P in enumerate(PVals):
            CSAmp = CSAmp_ds.sel(P=P).isel(t=-1).values
            Energy_Vals_inf[Pind] = pfs.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)

        Einf_tck = interpolate.splrep(PVals, Energy_Vals_inf, s=0)
        Pinf_Vals = np.linspace(np.min(PVals), np.max(PVals), 2 * PVals.size)
        Einf_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=0)
        Einf_2ndderiv_Vals = 1 * interpolate.splev(Pinf_Vals, Einf_tck, der=2)
        # Pcrit[aind] = Pinf_Vals[np.argwhere(Einf_2ndderiv_Vals < 0)[-2][0] + 3]
        Pcrit[aind] = Pinf_Vals[np.argmin(np.gradient(Einf_2ndderiv_Vals)) - 0]  # there is a little bit of fudging with the -3 here so that aIBi=-10 gives me Pcrit/(mI*c) = 1 -> I can also just generate data for weaker interactions and see if it's better

    Pcrit_norm = Pcrit / (mI * nu)
    Pcrit_tck = interpolate.splrep(aIBi_Vals, Pcrit_norm, s=0, k=3)
    aIBi_interpVals = np.linspace(np.min(aIBi_Vals), np.max(aIBi_Vals), 5 * aIBi_Vals.size)
    Pcrit_interpVals = 1 * interpolate.splev(aIBi_interpVals, Pcrit_tck, der=0)

    print(Pcrit_norm)
    print(Pcrit_norm[1], Pcrit_norm[5], Pcrit_norm[-5])

    scalefac = 1.0
    # scalefac = 0.95  # just to align weakly interacting case slightly to 1 (it's pretty much there, would just need higher resolution data)
    Pcrit_norm = scalefac * Pcrit_norm
    Pcrit_interpVals = scalefac * Pcrit_interpVals

    xmin = np.min(aIBi_interpVals)
    xmax = 1.01 * np.max(aIBi_interpVals)
    ymin = 0
    ymax = 1.01 * np.max(Pcrit_interpVals)

    font = {'family': 'serif', 'color': 'black', 'size': 14}
    sfont = {'family': 'serif', 'color': 'black', 'size': 13}

    fig, ax = plt.subplots()
    ax.plot(aIBi_Vals, Pcrit_norm, 'kx')
    ax.plot(aIBi_interpVals, Pcrit_interpVals, 'k-')
    # f1 = interpolate.interp1d(aIBi_Vals, Pcrit_norm, kind='cubic')
    # ax.plot(aIBi_interpVals, f1(aIBi_interpVals), 'k-')
    ax.set_title('Ground State Phase Diagram')
    ax.set_xlabel(r'$a_{IB}^{-1}$')
    ax.set_ylabel(r'$\frac{P}{m_{I}c_{BEC}}$')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.fill_between(aIBi_interpVals, Pcrit_interpVals, ymax, facecolor='b', alpha=0.25)
    ax.fill_between(aIBi_interpVals, ymin, Pcrit_interpVals, facecolor='g', alpha=0.25)
    ax.text(-3.0, ymin + 0.175 * (ymax - ymin), 'Polaron', fontdict=font)
    ax.text(-2.9, ymin + 0.1 * (ymax - ymin), '(' + r'$Z>0$' + ')', fontdict=sfont)
    ax.text(-6.5, ymin + 0.6 * (ymax - ymin), 'Cherenkov', fontdict=font)
    ax.text(-6.35, ymin + 0.525 * (ymax - ymin), '(' + r'$Z=0$' + ')', fontdict=sfont)
    plt.show()

    # # # ENERGY DERIVATIVES (SPHERICAL)

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

    # fig, axes = plt.subplots(nrows=3, ncols=1)
    # axes[0].plot(Pinf_Vals, Einf_Vals, 'k-')
    # axes[0].set_title('Ground State Energy (' + r'$a_{IB}^{-1}=$' + '{0})'.format(aIBi))
    # axes[0].set_xlabel('P')
    # axes[0].set_ylim([1.1 * np.min(Einf_Vals), -0.5])
    # axes[0].set_xlim([0, 2.0])

    # axes[1].plot(Pinf_Vals, Einf_1stderiv_Vals, 'k-')
    # axes[1].set_title('First Derivative of Energy')
    # axes[1].set_xlabel('P')
    # axes[1].plot(Pinf_Vals, vsound * np.ones(Pinf_Vals.size), 'r--', linewidth=2.0)
    # axes[1].set_ylim([0, 1.2 * np.max(Einf_1stderiv_Vals)])
    # axes[1].set_xlim([0, 2.0])

    # axes[2].plot(Pinf_Vals[::2], Einf_2ndderiv_Vals[::2], 'ko')
    # axes[2].set_title('Second Derivative of Energy')
    # axes[2].set_xlabel('P')
    # axes[2].plot(Pinf_Vals, ms * np.ones(Pinf_Vals.size), 'c--', linewidth=2.0)
    # axes[2].set_ylim([0, 1.2 * np.max(Einf_2ndderiv_Vals)])
    # axes[2].set_xlim([0, 2.0])

    # # # This plot below is for saturation/convergence of the energy with imaginary time
    # # fig3, ax3 = plt.subplots()
    # # Pind = 8
    # # ax3.plot(tVals, np.abs(Energy_Vals[Pind, :]), 'k-')
    # # ax3.set_yscale('log')
    # # ax3.set_xscale('log')
    # # ax3.set_title('Ground State Energy (' + r'$a_{IB}^{-1}=$' + '{0}, '.format(aIBi) + r'$P=$' + '{:.2f})'.format(PVals[Pind]))
    # # ax3.set_xlabel('Imaginary time')

    # fig.tight_layout()
    # plt.show()

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
    # plt.show()

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
    # plt.show()

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
    # plt.show()

    # # # Nph (SPHERICAL)

    # # IRrat_Vals = np.array([1, 2, 5, 10, 50, 1e2, 5e2, 1e3, 5e3, 1e4])
    # IRrat_Vals = np.array([1, 2, 5, 10, 50, 1e2])

    # aIBi_List = [-10.0, -5.0, -2.0, -0.5]

    # aIBi = aIBi_List[1]
    # IRrat = IRrat_Vals[0]
    # IRdatapath = innerdatapath + '/IRratio_{:.1E}'.format(IRrat)
    # qds_aIBi = (xr.open_dataset(IRdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))).isel(t=-1)

    # PVals = qds_aIBi['P'].values
    # n0 = qds_aIBi.attrs['n0']
    # gBB = qds_aIBi.attrs['gBB']
    # mI = qds_aIBi.attrs['mI']
    # mB = qds_aIBi.attrs['mB']
    # nu = np.sqrt(n0 * gBB / mB)

    # Nph_ds = qds_aIBi['Nph']
    # Nph_Vals = Nph_ds.values

    # Pind = np.argmin(np.abs(PVals - 3.0 * mI * nu))
    # Nph_IRcuts = np.zeros(IRrat_Vals.size)
    # for ind, IRrat in enumerate(IRrat_Vals):
    #     IRdatapath = innerdatapath + '/IRratio_{:.1E}'.format(IRrat)
    #     qds_IRrat = (xr.open_dataset(IRdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))).isel(t=-1)
    #     kmin = np.min(qds_IRrat.coords['k'].values)
    #     Nph_ds_IRrat = qds_IRrat['Nph']
    #     Nph_IRcuts[ind] = Nph_ds_IRrat.values[Pind]

    # fig, axes = plt.subplots(nrows=1, ncols=2)
    # axes[0].plot(PVals / (mI * nu), Nph_Vals, 'k-')
    # axes[0].set_title('Phonon Number (' + r'$aIB^{-1}=$' + '{0})'.format(aIBi))
    # axes[0].set_xlabel(r'$\frac{P}{m_{I}c_{BEC}}$')
    # axes[0].set_ylabel(r'$N_{ph}$')

    # axes[1].plot(IRrat_Vals, Nph_IRcuts, 'g-')
    # axes[1].set_xlabel('IR Cutoff Increase Ratio')
    # axes[1].set_ylabel(r'$N_{ph}$')
    # axes[1].set_title('Phonon Number (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c_{BEC}}=$' + '{:.1f})'.format(PVals[Pind] / (mI * nu)))

    # fig.tight_layout()
    # plt.show()

    # # IMPURITY DISTRIBUTION ANIMATION WITH CHARACTERIZATION (CARTESIAN)

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
    # print(Pnorm[Pind])
    # print(nPIm_deltaPeak_Vals[Pind])

    # fig1, ax = plt.subplots()
    # ax.plot(mI * nu * np.ones(PIm_Vals.size), np.linspace(0, 1, PIm_Vals.size), 'y--', label=r'$m_{I}c$')
    # curve = ax.plot(PIm_Vec[Pind], nPIm_Vec[Pind], color='k', lw=3, label='')
    # ind_s, ind_f = nPIm_FWHM_indices[Pind]
    # FWHMcurve = ax.plot(np.linspace(PIm_Vec[Pind][ind_s], PIm_Vec[Pind][ind_f], 100), nPIm_Vec[Pind][ind_s] * np.ones(100), 'b-', linewidth=3.0, label='Incoherent Part FWHM')
    # FWHMmarkers = ax.plot(np.linspace(PIm_Vec[Pind][ind_s], PIm_Vec[Pind][ind_f], 2), nPIm_Vec[Pind][ind_s] * np.ones(2), 'bD', mew=0.75, ms=7.5, label='')

    # Zline = ax.plot(PVals[Pind] * np.ones(PIm_Vals.size), np.linspace(0, nPIm_deltaPeak_Vals[Pind], PIm_Vals.size), 'r-', linewidth=3.0, label='Delta Peak (Z-factor)')
    # Zmarker = ax.plot(PVals[Pind], nPIm_deltaPeak_Vals[Pind], 'rx', mew=0.75, ms=7.5, label='')
    # norm_text = ax.text(0.7, 0.65, r'$\int n_{|\vec{P_{I}}|} d|\vec{P_{I}}| = $' + '{:.2f}'.format(nPIm_Tot_Vals[Pind]), transform=ax.transAxes, color='k')

    # ax.legend()
    # ax.set_xlim([-0.01, np.max(PIm_Vec[Pind])])
    # # ax.set_xlim([-0.01, 8])
    # ax.set_ylim([0, 1.05])
    # ax.set_title('Impurity Momentum Magnitude Distribution (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c_{BEC}}=$' + '{:.2f})'.format(Pnorm[Pind]))
    # ax.set_ylabel(r'$n_{|\vec{P_{I}}|}$')
    # ax.set_xlabel(r'$|\vec{P_{I}}|$')

    # # Plot characterization of nPIm(t=inf)
    # fig2, axes = plt.subplots(nrows=1, ncols=2)
    # axes[0].plot(PVals, nPIm_deltaPeak_Vals, 'r-')
    # axes[0].set_xlabel('$P$')
    # # axes[0].set_ylabel(r'$Z$')
    # axes[0].set_title('Delta Peak (Z-factor)')

    # axes[1].plot(PVals, nPIm_FWHM_Vals, 'b-')
    # axes[1].set_xlabel('$P$')
    # # axes[1].set_ylabel('FWHM')
    # axes[1].set_title('Incoherent Part FWHM')
    # fig2.tight_layout()

    # plt.show()
