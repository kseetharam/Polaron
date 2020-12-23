import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from matplotlib.animation import writers
import pf_dynamic_cart as pfc
import pf_dynamic_sph as pfs
import Grid
from scipy import interpolate
from timeit import default_timer as timer


if __name__ == "__main__":

    mpegWriter = writers['ffmpeg'](fps=2, bitrate=1800)
    matplotlib.rcParams.update({'font.size': 12})
    labelsize = 13
    legendsize = 12

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (60, 60, 60)
    (dx, dy, dz) = (0.25, 0.25, 0.25)
    higherCutoff = False; cutoffRat = 1.5
    betterResolution = True; resRat = 0.5

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.375, 0.375, 0.375)
    # higherCutoff = False; cutoffRat = 1.5
    # betterResolution = False; resRat = 0.5

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    massRat = 1.0
    IRrat = 1

    print(NGridPoints_cart)

    # Toggle parameters

    toggleDict = {'Dynamics': 'real'}

    # import solarized
    # Sol = solarized.Solarized()
    # cmap = Sol.cmap(type='linear')

    # cmap = 'gist_gray'
    cmap = 'afmhot'
    # cmap = 'inferno'
    my_cmap = matplotlib.cm.get_cmap(cmap)
    avmin = 1e-5; avmax = 1e-1

    # ---- SET OUTPUT DATA FOLDER ----

    datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    animpath = '/Users/kis/Dropbox/VariationalResearch/DataAnalysis/figs/rdyn_twophonon/hostGasDensity/NGridPoints_1.11E+08_resRat_0.50'
    if higherCutoff is True:
        datapath = datapath + '_cutoffRat_{:.2f}'.format(cutoffRat)
    if betterResolution is True:
        datapath = datapath + '_resRat_{:.2f}'.format(resRat)
    datapath = datapath + '/massRatio={:.1f}'.format(massRat)

    if toggleDict['Dynamics'] == 'real':
        innerdatapath = datapath + '/redyn'
        cartdatapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}/redyn_cart'.format(1.44e6, 1)
    elif toggleDict['Dynamics'] == 'imaginary':
        innerdatapath = datapath + '/imdyn'
        cartdatapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}/imdyn_cart'.format(1.44e6, 1)

    innerdatapath = innerdatapath + '_spherical'

    # # Analysis of Total Dataset
    interpdatapath = innerdatapath + '/interp'

    aIBi = -5
    # Pnorm_des = 4.0
    # Pnorm_des = 3.0
    Pnorm_des = 2.067
    # Pnorm_des = 1.8
    # Pnorm_des = 1.4
    # Pnorm_des = 1.34
    # Pnorm_des = 1.28
    # Pnorm_des = 1.22
    # Pnorm_des = 1.1
    # Pnorm_des = 1.04
    # Pnorm_des = 0.8
    # Pnorm_des = 0.52

    linDimList = [(2, 2), (10, 10)]
    linDimMajor, linDimMinor = linDimList[1]

    qds_orig = xr.open_dataset(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(Pnorm_des * 0.7926654595212022, aIBi))

    n0 = qds_orig.attrs['n0']; gBB = qds_orig.attrs['gBB']; mI = qds_orig.attrs['mI']; mB = qds_orig.attrs['mB']
    nu = np.sqrt(n0 * gBB / mB)
    mc = mI * nu
    aBB = (mB / (4 * np.pi)) * gBB
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)
    tscale = xi / nu
    P = qds_orig.attrs['P']
    Pnorm = P / mc
    tVals = qds_orig['tc'].values
    t = tVals[-1]
    # print(tVals)

    print(xi)
    print('P/mc: {:.2f}'.format(P / mc))
    print(P)
    print(massRat, aIBi)
    print(t / tscale)

    # All Plotting:

    # # ORIGINAL SPHERICAL DATA PLOTS

    # Individual Phonon Momentum Distribution(Original Spherical data)

    Bk_2D_orig = (qds_orig['Real_CSAmp'] + 1j * qds_orig['Imag_CSAmp']).sel(tc=t).values
    Nph_orig = qds_orig['Nph'].sel(t=t).values
    PhDen_orig_Vals = ((1 / Nph_orig) * np.abs(Bk_2D_orig)**2).real.astype(float)

    kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds_orig.coords['k'].values); kgrid.initArray_premade('th', qds_orig.coords['th'].values)
    kVec = kgrid.getArray('k'); dk = kVec[1] - kVec[0]
    thVec = kgrid.getArray('th'); dth = thVec[1] - thVec[0]
    print(P, np.max(kVec))
    kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    kxg = kg * np.sin(thg)
    kzg = kg * np.cos(thg)

    # interpmul = 2

    # PhDen_orig_da = xr.DataArray(PhDen_orig_Vals, coords=[kVec, thVec], dims=['k', 'th'])
    # PhDen_orig_smooth, kg_orig_smooth, thg_orig_smooth = pfc.xinterp2D(PhDen_orig_da, 'k', 'th', interpmul)
    # dk_smooth = kg_orig_smooth[1, 0] - kg_orig_smooth[0, 0]
    # dth_smooth = thg_orig_smooth[0, 1] - thg_orig_smooth[0, 0]
    # kxg_smooth = kg_orig_smooth * np.sin(thg_orig_smooth)
    # kzg_smooth = kg_orig_smooth * np.cos(thg_orig_smooth)
    # PhDen_orig_sum = np.sum(PhDen_orig_Vals * kg**2 * np.sin(thg) * dk * dth * (2 * np.pi)**(-2))
    # PhDen_smooth_sum = np.sum(PhDen_orig_smooth * kg_orig_smooth**2 * np.sin(thg_orig_smooth) * dk_smooth * dth_smooth * (2 * np.pi)**(-2))
    # print(PhDen_orig_sum, PhDen_smooth_sum)

    fig1, ax1 = plt.subplots()
    vmax = np.max(PhDen_orig_Vals)
    # vmax = 8414555  # P=2.4
    # vmax = 2075494  # P=1.20
    # vmax = 1055106  # P=0.38
    quad1 = ax1.pcolormesh(kzg, kxg, PhDen_orig_Vals, norm=colors.LogNorm(vmin=1e-3, vmax=vmax), cmap='inferno')
    quad1m = ax1.pcolormesh(kzg, -1 * kxg, PhDen_orig_Vals, norm=colors.LogNorm(vmin=1e-3, vmax=vmax), cmap='inferno')
    ax1.set_xlim([-1 * linDimMajor, linDimMajor])
    ax1.set_ylim([-1 * linDimMinor, linDimMinor])
    # print(vmax)
    ax1.set_xlabel('kz (Impurity Propagation Direction)')
    ax1.set_ylabel('kx')
    ax1.set_title('Individual Phonon Momentum Distribution (Sph Orig)', size='smaller')
    fig1.colorbar(quad1, ax=ax1, extend='both')

    # CARTESIAN INTERPOLATION PLOTS

    interp_ds = xr.open_dataset(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_t_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, t, linDimMajor, linDimMinor))
    # interp_ds = xr.open_dataset(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}_unique.nc'.format(P, aIBi, linDimMajor, linDimMinor)); print('unique')
    kxL = interp_ds['kx'].values; dkxL = kxL[1] - kxL[0]
    kyL = interp_ds['ky'].values; dkyL = kyL[1] - kyL[0]
    kzL = interp_ds['kz'].values; dkzL = kzL[1] - kzL[0]
    xL = interp_ds['x'].values
    yL = interp_ds['y'].values
    zL = interp_ds['z'].values
    PI_mag = interp_ds['PI_mag'].values
    kxLg_xz_slice, kzLg_xz_slice = np.meshgrid(kxL, kzL, indexing='ij')
    xLg_xz, zLg_xz = np.meshgrid(xL, zL, indexing='ij')
    xLg_xy, yLg_xy = np.meshgrid(xL, yL, indexing='ij')
    PhDenLg_xz_slice = interp_ds['PhDen_xz'].values

    PIx = interp_ds['PI_x'].values
    PIy = interp_ds['PI_y'].values
    PIz = interp_ds['PI_z'].values
    PIxg, PIzg = np.meshgrid(PIx, PIz, indexing='ij')

    nPI_mag = interp_ds['nPI_mag'].values
    # mom_deltapeak = interp_ds.attrs['mom_deltapeak']
    mom_deltapeak = interp_ds['mom_deltapeak'].values
    print(mom_deltapeak)

    n0 = interp_ds.attrs['n0']
    gBB = interp_ds.attrs['gBB']
    mI = interp_ds.attrs['mI']
    mB = interp_ds.attrs['mB']
    nu = np.sqrt(n0 * gBB / mB)
    mc = mI * nu

    # # Individual Phonon Momentum Distribution (Cart Interp)
    # fig2, ax2 = plt.subplots()
    # quad2 = ax2.pcolormesh(kzLg_xz_slice, kxLg_xz_slice, PhDenLg_xz_slice, norm=colors.LogNorm(vmin=1e-3, vmax=np.max(PhDen_orig_Vals)), cmap='inferno')
    # ax2.set_xlabel('kz (Impurity Propagation Direction)')
    # ax2.set_ylabel('kx')
    # ax2.set_title('Individual Phonon Momentum Distribution (Cart Interp)', size='smaller')
    # fig2.colorbar(quad2, ax=ax2, extend='both')

    # # Impurity Momentum Magnitude Distribution (Interp)
    # fig5, ax5 = plt.subplots()
    # ax5.plot(mc * np.ones(PI_mag.size), np.linspace(0, 1, PI_mag.size), 'y--', label=r'$m_{I}c_{BEC}$')
    # curve = ax5.plot(PI_mag, nPI_mag, color='k', lw=3, label='')
    # # D = nPI_mag - np.max(nPI_mag) / 2
    # # indices = np.where(D > 0)[0]
    # # ind_s, ind_f = indices[0], indices[-1]
    # # FWHMcurve = ax5.plot(np.linspace(PI_mag[ind_s], PI_mag[ind_f], 100), nPI_mag[ind_s] * np.ones(100), 'b-', linewidth=3.0, label='Incoherent Part FWHM')
    # # FWHMmarkers = ax5.plot(np.linspace(PI_mag[ind_s], PI_mag[ind_f], 2), nPI_mag[ind_s] * np.ones(2), 'bD', mew=0.75, ms=7.5, label='')
    # Zline = ax5.plot(P * np.ones(PI_mag.size), np.linspace(0, mom_deltapeak, PI_mag.size), 'r-', linewidth=3.0, label='Delta Peak (Z-factor)')
    # Zmarker = ax5.plot(P, mom_deltapeak, 'rx', mew=0.75, ms=7.5, label='')
    # dPIm = PI_mag[1] - PI_mag[0]
    # nPIm_Tot = np.sum(nPI_mag * dPIm) + mom_deltapeak
    # norm_text = ax5.text(0.7, 0.65, r'$\int n_{|\vec{P_{I}}|} d|\vec{P_{I}}| = $' + '{:.2f}'.format(nPIm_Tot), transform=ax5.transAxes, color='k')

    # ax5.legend()
    # ax5.set_xlim([-0.01, np.max(PI_mag)])
    # ax5.set_ylim([0, 1.05])
    # ax5.set_title('Impurity Momentum Magnitude Distribution (Cart Interp) (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c_{BEC}}=$' + '{:.2f})'.format(P / mc), size='smaller')
    # ax5.set_ylabel(r'$n_{|\vec{P_{I}}|}$')
    # ax5.set_xlabel(r'$|\vec{P_{I}}|$')

    # # # Slices

    # nPI_xz_slice = interp_ds['nPI_xz_slice'].values
    # fig4, ax4 = plt.subplots()
    # quad4 = ax4.pcolormesh(PIzg / mc, PIxg / mc, nPI_xz_slice, norm=colors.LogNorm(vmin=1e-3, vmax=5e-1), cmap=cmap)
    # ax4.set_xlabel(r'$PI_{z}/mc$', fontsize=labelsize)
    # ax4.set_ylabel(r'$PI_{x}/mc$', fontsize=labelsize)
    # ax4.set_xlim([-5, 5])
    # ax4.set_ylim([-5, 5])
    # ax4.plot(np.ones(PIx.size), PIx / mc, 'k--')
    # fig4.colorbar(quad4, ax=ax4, extend='both')

    # BARE ATOM POSITION DISTRIBUTIONS

    # # Slices

    # na_xz_slice = interp_ds['na_xz_slice'].values
    # na_xy_slice = interp_ds['na_xy_slice'].values
    # print(np.max(na_xz_slice))

    # fig4, ax4 = plt.subplots()
    # # quad4 = ax4.pcolormesh(zLg_xz / xi, xLg_xz / xi, na_xz_slice, norm=colors.LogNorm(vmin=np.abs(np.min(na_xz_slice)), vmax=np.max(na_xz_slice)), cmap='inferno')
    # quad4 = ax4.pcolormesh(zLg_xz / xi, xLg_xz / xi, na_xz_slice, norm=colors.LogNorm(vmin=1e-10, vmax=np.max(na_xz_slice)), cmap=cmap)
    # # quad4 = ax4.pcolormesh(zLg_xz / xi, xLg_xz / xi, na_xz_slice, norm=colors.LogNorm(vmin=avmin, vmax=avmax), cmap=cmap)
    # ax4.set_xlabel(r'$z/\xi$', fontsize=labelsize)
    # ax4.set_ylabel(r'$x/\xi$', fontsize=labelsize)
    # # ax4.set_title('Host Gas Density')
    # fig4.colorbar(quad4, ax=ax4, extend='both')

    # fig6, ax6 = plt.subplots()
    # # quad6 = ax6.pcolormesh(yLg_xy / xi, xLg_xy / xi, na_xy_slice, norm=colors.LogNorm(vmin=np.abs(np.min(na_xy_slice)), vmax=np.max(na_xy_slice)), cmap='inferno')
    # quad6 = ax6.pcolormesh(yLg_xy / xi, xLg_xy / xi, na_xy_slice, norm=colors.LogNorm(vmin=1e-10, vmax=np.max(na_xy_slice)), cmap=cmap)
    # # quad6 = ax6.pcolormesh(yLg_xy / xi, xLg_xy / xi, na_xy_slice, norm=colors.LogNorm(vmin=avmin, vmax=avmax), cmap=cmap)
    # ax6.set_xlabel(r'$y/\xi$', fontsize=labelsize)
    # ax6.set_ylabel(r'$x/\xi$', fontsize=labelsize)
    # # ax6.set_title('Host Gas Density')
    # fig6.colorbar(quad6, ax=ax6, extend='both')

    # Integration

    dx = xL[1] - xL[0]; dy = yL[1] - yL[0]; dz = zL[1] - zL[0]

    na_xz_int = interp_ds['na_xz_int'].values; na_xz_int_norm = na_xz_int / (np.sum(na_xz_int) * dx * dz)
    na_xy_int = interp_ds['na_xy_int'].values; na_xy_int_norm = na_xy_int / (np.sum(na_xy_int) * dx * dy)
    print(np.max(na_xz_int_norm), np.max(na_xy_int_norm))

    fig7, ax7 = plt.subplots()
    # quad7 = ax7.pcolormesh(zLg_xz / xi, xLg_xz / xi, na_xz_int_norm, norm=colors.LogNorm(vmin=np.abs(np.min(na_xz_int_norm)), vmax=np.max(na_xz_int_norm)), cmap='inferno')
    # quad7 = ax7.pcolormesh(zLg_xz / xi, xLg_xz / xi, na_xz_int_norm, norm=colors.LogNorm(vmin=5e-5, vmax=np.max(na_xz_int_norm)), cmap='inferno')
    quad7 = ax7.pcolormesh(zLg_xz / xi, xLg_xz / xi, na_xz_int_norm, norm=colors.LogNorm(vmin=avmin, vmax=avmax), cmap=cmap)
    ax7.set_xlabel(r'$z/\xi$', fontsize=labelsize)
    ax7.set_ylabel(r'$x/\xi$', fontsize=labelsize)
    ax7.text(0.67, 0.9, r'$t/(\xi c^{-1})$' + ': {:.1f}'.format(t / tscale), transform=ax7.transAxes, color='w')
    fig7.colorbar(quad7, ax=ax7, extend='both')

    fig8, ax8 = plt.subplots()
    # quad8 = ax8.pcolormesh(yLg_xy / xi, xLg_xy / xi, na_xy_int_norm, norm=colors.LogNorm(vmin=np.abs(np.min(na_xy_int_norm)), vmax=np.max(na_xy_int_norm)), cmap='inferno')
    # quad8 = ax8.pcolormesh(yLg_xy / xi, xLg_xy / xi, na_xy_int_norm, norm=colors.LogNorm(vmin=5e-5, vmax=np.max(na_xy_int_norm)), cmap='inferno')
    quad8 = ax8.pcolormesh(yLg_xy / xi, xLg_xy / xi, na_xy_int_norm, norm=colors.LogNorm(vmin=avmin, vmax=avmax), cmap=cmap)
    ax8.text(0.67, 0.9, r'$t/(\xi c^{-1})$' + ': {:.1f}'.format(t / tscale), transform=ax8.transAxes, color='w')
    ax8.set_xlabel(r'$y/\xi$', fontsize=labelsize)
    ax8.set_ylabel(r'$x/\xi$', fontsize=labelsize)
    fig8.colorbar(quad8, ax=ax8, extend='both')

    # fig9, ax9 = plt.subplots()
    # ax9.plot(zL / xi, np.sum(na_xz_int_norm, axis=0) * dx)
    # ax9.set_xlim([-30, 30])
    # ax9.set_xlabel(r'$z/\xi$', fontsize=labelsize)
    # ax9.set_title('Integrated density (impurity propagation direction)')

    # fig10, ax10 = plt.subplots()
    # ax10.plot(zL / xi, np.sum(na_xz_int_norm, axis=1) * dx)
    # ax10.set_xlim([-30, 30])
    # ax10.set_xlabel(r'$z/\xi$', fontsize=labelsize)
    # ax10.set_title('Integrated density (transverse direction)')

    # # BARE ATOM POSITION ANIMATION

    tVals_anim = np.array([0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97])
    tVals = tVals_anim

    na_xz_intnorm_array = np.empty(tVals.size, dtype=np.object)
    for tind, t in enumerate(tVals):
        interp_ds = xr.open_dataset(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_t_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, t, linDimMajor, linDimMinor))
        na_xz_int = interp_ds['na_xz_int'].values; na_xz_int_norm = na_xz_int / (np.sum(na_xz_int) * dx * dz)
        na_xz_intnorm_array[tind] = na_xz_int_norm

    fig_a1, ax_a1 = plt.subplots()
    quad_a1 = ax_a1.pcolormesh(zLg_xz / xi, xLg_xz / xi, na_xz_intnorm_array[0][:-1, :-1], norm=colors.LogNorm(vmin=avmin, vmax=avmax), cmap=cmap)
    t_text = ax_a1.text(0.67, 0.9, r'$t/(\xi c^{-1})$' + ': {:.1f}'.format(tVals[0] / tscale), transform=ax_a1.transAxes, color='w')
    ax_a1.set_xlabel(r'$z/\xi$', fontsize=labelsize)
    ax_a1.set_ylabel(r'$x/\xi$', fontsize=labelsize)
    fig_a1.colorbar(quad_a1, ax=ax_a1, extend='both')

    def animate_Den(i):
        quad_a1.set_array(na_xz_intnorm_array[i][:-1, :-1].ravel())
        t_text.set_text(r'$t/(\xi c^{-1})$' + ': {:.1f}'.format(tVals[i] / tscale))

    anim_Den = FuncAnimation(fig_a1, animate_Den, interval=1000, frames=range(tVals.size), repeat=True)
    anim_Den_filename = '/integratedDensity_xz_mRat_{:.1f}_Pnorm_{:.2f}_aIBi_{:.2f}.mp4'.format(massRat, P / mc, aIBi)
    anim_Den.save(animpath + anim_Den_filename, writer=mpegWriter)

    # # MOMENTUM DISTRIVUTION ANIMATION

    tVals_anim = np.array([0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97])
    tVals = tVals_anim

    na_xz_intnorm_array = np.empty(tVals.size, dtype=np.object)
    for tind, t in enumerate(tVals):
        interp_ds = xr.open_dataset(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_t_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, t, linDimMajor, linDimMinor))
        na_xz_int = interp_ds['nPI_xz_slice'].values
        na_xz_intnorm_array[tind] = na_xz_int

    fig_a1, ax_a1 = plt.subplots()
    quad_a1 = ax_a1.pcolormesh(PIzg / mc, PIxg / mc, na_xz_intnorm_array[0][:-1, :-1], norm=colors.LogNorm(vmin=1e-3, vmax=1e0), cmap=cmap)
    t_text = ax_a1.text(0.67, 0.9, r'$t/(\xi c^{-1})$' + ': {:.1f}'.format(tVals[0] / tscale), transform=ax_a1.transAxes, color='w')
    ax_a1.set_xlabel(r'$Pz/mc$', fontsize=labelsize)
    ax_a1.set_ylabel(r'$Px/\xi$', fontsize=labelsize)
    fig_a1.colorbar(quad_a1, ax=ax_a1, extend='both')
    ax_a1.plot(np.ones(PIx.size), PIx / mc, 'k--')
    ax_a1.set_xlim([-5, 5])
    ax_a1.set_ylim([-5, 5])

    def animate_Den(i):
        quad_a1.set_array(na_xz_intnorm_array[i][:-1, :-1].ravel())
        t_text.set_text(r'$t/(\xi c^{-1})$' + ': {:.1f}'.format(tVals[i] / tscale))

    anim_Den = FuncAnimation(fig_a1, animate_Den, interval=1000, frames=range(tVals.size), repeat=True)
    anim_Den_filename = '/impurityMomentum_xz_mRat_{:.1f}_Pnorm_{:.2f}_aIBi_{:.2f}.mp4'.format(massRat, P / mc, aIBi)
    anim_Den.save(animpath + anim_Den_filename, writer=mpegWriter)

    # # # MOMENTUM DISTRIBUTION SLICE ANIMATION

    tVals_anim = np.array([0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97])
    tVals = tVals_anim

    nPI_z_array = np.empty(tVals.size, dtype=np.object)
    maxVal = 1e-10
    for tind, t in enumerate(tVals):
        interp_ds = xr.open_dataset(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_t_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, t, linDimMajor, linDimMinor))
        nPI_z_array[tind] = interp_ds['nPI_xz_slice'].sel(PI_x=0, method='nearest').values
        if np.max(nPI_z_array[tind]) > maxVal:
            maxVal = np.max(nPI_z_array[tind])

    fig_a1, ax_a1 = plt.subplots()
    line_a1 = ax_a1.plot(PIz / mc, nPI_z_array[0])[0]
    t_text = ax_a1.text(0.67, 0.9, r'$t/(\xi c^{-1})$' + ': {:.1f}'.format(tVals[0] / tscale), transform=ax_a1.transAxes, color='r')
    ax_a1.set_xlabel(r'$P_{\mathrm{imp},z}/mc$', fontsize=labelsize)
    # ax_a1.plot(np.ones(PIz.size), mc, 'k--')
    ax_a1.vlines(mc, 0, maxVal, color='k', linestyle='--')
    # ax_a1.set_xlim([-5, 5])
    # ax_a1.set_ylim([-5, 5])

    def animate_Den(i):
        if i > tVals.size:
            return
        line_a1.set_ydata(nPI_z_array[i])
        t_text.set_text(r'$t/(\xi c^{-1})$' + ': {:.1f}'.format(tVals[i] / tscale))

    anim_Den = FuncAnimation(fig_a1, animate_Den, interval=1000, frames=range(tVals.size), repeat=True)
    anim_Den_filename = '/impurityMomentum_z_slice_mRat_{:.1f}_Pnorm_{:.2f}_aIBi_{:.2f}.mp4'.format(massRat, P / mc, aIBi)
    anim_Den.save(animpath + anim_Den_filename, writer=mpegWriter)

    plt.show()
