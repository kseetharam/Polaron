import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pf_dynamic_cart as pfc
import pf_dynamic_sph as pfs
import Grid
from scipy import interpolate
from timeit import default_timer as timer


if __name__ == "__main__":

    # ---- INITIALIZE GRIDS ----

    # (Lx, Ly, Lz) = (60, 60, 60)
    # (dx, dy, dz) = (0.25, 0.25, 0.25)
    # higherCutoff = False; cutoffRat = 1.5
    # betterResolution = True; resRat = 0.5

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)
    higherCutoff = False; cutoffRat = 1.5
    betterResolution = False; resRat = 0.5

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    massRat = 1.0
    IRrat = 1

    # Toggle parameters

    toggleDict = {'Dynamics': 'imaginary', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon', 'Old': False}

    # ---- SET OUTPUT DATA FOLDER ----

    datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    animpath = '/media/kis/Storage/Dropbox/VariationalResearch/DataAnalysis/figs'
    if higherCutoff is True:
        datapath = datapath + '_cutoffRat_{:.2f}'.format(cutoffRat)
    if betterResolution is True:
        datapath = datapath + '_resRat_{:.2f}'.format(resRat)
    datapath = datapath + '/massRatio={:.1f}'.format(massRat)

    if toggleDict['Old'] is True:
        datapath = datapath + '_old'

    if toggleDict['Dynamics'] == 'real':
        innerdatapath = datapath + '/redyn'
        animpath = animpath + '/rdyn'
        cartdatapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}/redyn_cart'.format(1.44e6, 1)
    elif toggleDict['Dynamics'] == 'imaginary':
        innerdatapath = datapath + '/imdyn'
        animpath = animpath + '/idyn'
        cartdatapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}/imdyn_cart'.format(1.44e6, 1)

    innerdatapath = innerdatapath + '_spherical'

    # # Analysis of Total Dataset
    interpdatapath = innerdatapath + '/interp'

    # aIBi = -10
    # # Pnorm_des = 2.64
    # Pnorm_des = 1.0

    aIBi = -10
    Pnorm_des = 2.0
    # Pnorm_des = 1.0
    # Pnorm_des = 0.1

    linDimList = [(2, 2), (10, 10)]
    linDimMajor, linDimMinor = linDimList[0]

    qds_orig = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    n0 = qds_orig.attrs['n0']; gBB = qds_orig.attrs['gBB']; mI = qds_orig.attrs['mI']; mB = qds_orig.attrs['mB']
    nu = np.sqrt(n0 * gBB / mB)
    mc = mI * nu
    PVals = qds_orig['P'].values
    Pnorm = PVals / mc
    Pind = np.abs(Pnorm - Pnorm_des).argmin().astype(int)
    P = PVals[Pind]
    print('P (orig): {:.2f}'.format(P))

    # All Plotting:

    # # ORIGINAL SPHERICAL DATA PLOTS

    # Individual Phonon Momentum Distribution(Original Spherical data)
    if Lx == 60:
        Bk_2D_orig = (qds_orig['Real_CSAmp'] + 1j * qds_orig['Imag_CSAmp']).sel(P=P).isel(tc=-1).values
    else:
        Bk_2D_orig = (qds_orig['Real_CSAmp'] + 1j * qds_orig['Imag_CSAmp']).sel(P=P).isel(t=-1).values
    Nph_orig = qds_orig['Nph'].sel(P=P).isel(t=-1).values
    PhDen_orig_Vals = ((1 / Nph_orig) * np.abs(Bk_2D_orig)**2).real.astype(float)

    kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds_orig.coords['k'].values); kgrid.initArray_premade('th', qds_orig.coords['th'].values)
    kVec = kgrid.getArray('k'); dk = kVec[1] - kVec[0]
    thVec = kgrid.getArray('th'); dth = thVec[1] - thVec[0]
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

    # # GROUND STATE PLOTS
    # fig1, ax1 = plt.subplots()
    # vmax = 17000
    # axislim = 1.2
    # interpmul = 5
    # wk_Vals = pfs.omegak(kVec, mB, n0, gBB)
    # bdiff = 100 * np.abs(wk_Vals - nu * kVec) / (nu * kVec)
    # kind = np.abs(bdiff - 1).argmin().astype(int)
    # klin = kVec[kind]
    # PhDen_orig_da = xr.DataArray(PhDen_orig_Vals, coords=[kVec, thVec], dims=['k', 'th'])
    # PhDen_orig_smooth, kg_orig_smooth, thg_orig_smooth = pfc.xinterp2D(PhDen_orig_da, 'k', 'th', interpmul)
    # kxg_smooth = kg_orig_smooth * np.sin(thg_orig_smooth)
    # kzg_smooth = kg_orig_smooth * np.cos(thg_orig_smooth)
    # quad1 = ax1.pcolormesh(kzg_smooth, kxg_smooth, PhDen_orig_smooth, vmin=0, vmax=vmax, cmap='inferno')
    # quad1m = ax1.pcolormesh(kzg_smooth, -1 * kxg_smooth, PhDen_orig_smooth, vmin=0, vmax=vmax, cmap='inferno')
    # patch_klin = plt.Circle((0, 0), klin, edgecolor='tab:cyan', facecolor='None')
    # ax1.add_patch(patch_klin)
    # ax1.set_xlim([-1 * axislim, axislim])
    # ax1.set_ylim([-1 * axislim, axislim])
    # ax1.grid(True, linewidth=0.5)
    # ax1.legend([patch_klin], [r'Linear Excitations'], loc=2, fontsize='small')
    # ax1.set_xlabel('kz (Impurity Propagation Direction)')
    # ax1.set_ylabel('kx')
    # ax1.set_title('Individual Phonon Distribution (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c}=$' + '{:.2f})'.format(Pnorm[Pind]))
    # fig1.colorbar(quad1, ax=ax1, extend='both')
    # # filepath = '/media/kis/Storage/Dropbox/VariationalResearch/DataAnalysis/figs/rdyn_twophonon/distributionAnims/GroundStatePlots'
    # # filename = '/aIBi_{:.2f}_P_{:.2f}'.format(aIBi, P) + '_indPhononDist_2D_GS'
    # # plt.savefig(filepath + filename + '.png')

    # fig, ax = plt.subplots()
    # quad = ax.pcolormesh(kzg_smooth, kxg_smooth, PhDen_orig_smooth, norm=colors.LogNorm(vmin=1e-3, vmax=np.max(PhDen_orig_Vals)), cmap='inferno')
    # quadm = ax.pcolormesh(kzg_smooth, -1 * kxg_smooth, PhDen_orig_smooth, norm=colors.LogNorm(vmin=1e-3, vmax=np.max(PhDen_orig_Vals)), cmap='inferno')
    # ax.set_xlim([-1 * linDimMajor, linDimMajor])
    # ax.set_ylim([-1 * linDimMinor, linDimMinor])
    # ax.set_xlabel('kz (Impurity Propagation Direction)')
    # ax.set_ylabel('kx')
    # ax.set_title('Individual Phonon Momentum Distribution (Sph Interp)', size='smaller')
    # fig.colorbar(quad, ax=ax, extend='both')

    # CARTESIAN INTERPOLATION PLOTS

    interp_ds = xr.open_dataset(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, linDimMajor, linDimMinor))
    # interp_ds = xr.open_dataset(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}_unique.nc'.format(P, aIBi, linDimMajor, linDimMinor)); print('unique')
    kxL = interp_ds['kx'].values; dkxL = kxL[1] - kxL[0]
    kyL = interp_ds['ky'].values; dkyL = kyL[1] - kyL[0]
    kzL = interp_ds['kz'].values; dkzL = kzL[1] - kzL[0]
    xL = interp_ds['x'].values
    zL = interp_ds['z'].values
    PI_mag = interp_ds['PI_mag'].values
    kxLg_xz_slice, kzLg_xz_slice = np.meshgrid(kxL, kzL, indexing='ij')
    xLg_xz_slice, zLg_xz_slice = np.meshgrid(xL, zL, indexing='ij')
    PhDenLg_xz_slice = interp_ds['PhDen_xz'].values

    nPI_mag = interp_ds['nPI_mag'].values
    mom_deltapeak = interp_ds.attrs['mom_deltapeak']

    n0 = interp_ds.attrs['n0']
    gBB = interp_ds.attrs['gBB']
    mI = interp_ds.attrs['mI']
    mB = interp_ds.attrs['mB']
    nu = np.sqrt(n0 * gBB / mB)
    mc = mI * nu

    # Individual Phonon Momentum Distribution (Cart Interp)
    fig2, ax2 = plt.subplots()
    quad2 = ax2.pcolormesh(kzLg_xz_slice, kxLg_xz_slice, PhDenLg_xz_slice, norm=colors.LogNorm(vmin=1e-3, vmax=np.max(PhDen_orig_Vals)), cmap='inferno')
    ax2.set_xlabel('kz (Impurity Propagation Direction)')
    ax2.set_ylabel('kx')
    ax2.set_title('Individual Phonon Momentum Distribution (Cart Interp)', size='smaller')
    fig2.colorbar(quad2, ax=ax2, extend='both')

    # Impurity Momentum Magnitude Distribution (Interp)
    fig5, ax5 = plt.subplots()
    ax5.plot(mc * np.ones(PI_mag.size), np.linspace(0, 1, PI_mag.size), 'y--', label=r'$m_{I}c_{BEC}$')
    curve = ax5.plot(PI_mag, nPI_mag, color='k', lw=3, label='')
    D = nPI_mag - np.max(nPI_mag) / 2
    indices = np.where(D > 0)[0]
    ind_s, ind_f = indices[0], indices[-1]
    FWHMcurve = ax5.plot(np.linspace(PI_mag[ind_s], PI_mag[ind_f], 100), nPI_mag[ind_s] * np.ones(100), 'b-', linewidth=3.0, label='Incoherent Part FWHM')
    FWHMmarkers = ax5.plot(np.linspace(PI_mag[ind_s], PI_mag[ind_f], 2), nPI_mag[ind_s] * np.ones(2), 'bD', mew=0.75, ms=7.5, label='')
    Zline = ax5.plot(P * np.ones(PI_mag.size), np.linspace(0, mom_deltapeak, PI_mag.size), 'r-', linewidth=3.0, label='Delta Peak (Z-factor)')
    Zmarker = ax5.plot(P, mom_deltapeak, 'rx', mew=0.75, ms=7.5, label='')
    dPIm = PI_mag[1] - PI_mag[0]
    nPIm_Tot = np.sum(nPI_mag * dPIm) + mom_deltapeak
    norm_text = ax5.text(0.7, 0.65, r'$\int n_{|\vec{P_{I}}|} d|\vec{P_{I}}| = $' + '{:.2f}'.format(nPIm_Tot), transform=ax5.transAxes, color='k')

    ax5.legend()
    ax5.set_xlim([-0.01, np.max(PI_mag)])
    ax5.set_ylim([0, 1.05])
    ax5.set_title('Impurity Momentum Magnitude Distribution (Cart Interp) (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c_{BEC}}=$' + '{:.2f})'.format(P / mc), size='smaller')
    ax5.set_ylabel(r'$n_{|\vec{P_{I}}|}$')
    ax5.set_xlabel(r'$|\vec{P_{I}}|$')

    # ORIGINAL CARTESIAN DATA PLOTS

    # Impurity Momentum Magnitude Distribution (Original Cartesian data)

    qds_orig_cart = xr.open_dataset(cartdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi)).isel(t=-1)
    qds_nPIm_inf = qds_orig_cart['nPI_mag'].sel(P=P, method='nearest').dropna('PI_mag')
    P_cart = qds_nPIm_inf.coords['P'].values
    PI_mag_cart = qds_nPIm_inf.coords['PI_mag'].values
    nPI_mag_cart = qds_nPIm_inf.values
    mom_deltapeak_cart = qds_orig_cart.sel(P=P_cart)['mom_deltapeak'].values
    print('P (cart): {:.2f}'.format(P_cart))

    fig6, ax6 = plt.subplots()
    ax6.plot(mc * np.ones(PI_mag_cart.size), np.linspace(0, 1, PI_mag_cart.size), 'y--', label=r'$m_{I}c_{BEC}$')
    curve = ax6.plot(PI_mag_cart, nPI_mag_cart, color='k', lw=3, label='')
    D = nPI_mag_cart - np.max(nPI_mag_cart) / 2
    indices = np.where(D > 0)[0]
    ind_s, ind_f = indices[0], indices[-1]
    FWHMcurve = ax6.plot(np.linspace(PI_mag_cart[ind_s], PI_mag_cart[ind_f], 100), nPI_mag_cart[ind_s] * np.ones(100), 'b-', linewidth=3.0, label='Incoherent Part FWHM')
    FWHMmarkers = ax6.plot(np.linspace(PI_mag_cart[ind_s], PI_mag_cart[ind_f], 2), nPI_mag_cart[ind_s] * np.ones(2), 'bD', mew=0.75, ms=7.5, label='')
    Zline = ax6.plot(P_cart * np.ones(PI_mag_cart.size), np.linspace(0, mom_deltapeak_cart, PI_mag_cart.size), 'r-', linewidth=3.0, label='Delta Peak (Z-factor)')
    Zmarker = ax6.plot(P_cart, mom_deltapeak_cart, 'rx', mew=0.75, ms=7.5, label='')
    dPIm_cart = PI_mag_cart[1] - PI_mag_cart[0]
    nPIm_Tot = np.sum(nPI_mag_cart * dPIm_cart) + mom_deltapeak_cart
    norm_text = ax6.text(0.7, 0.65, r'$\int n_{|\vec{P_{I}}|} d|\vec{P_{I}}| = $' + '{:.2f}'.format(nPIm_Tot), transform=ax6.transAxes, color='k')

    ax6.legend()
    ax6.set_xlim([-0.01, np.max(PI_mag)])
    ax6.set_ylim([0, 1.05])
    ax6.set_title('Impurity Momentum Magnitude Distribution (Cart Orig) (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$\frac{P}{m_{I}c_{BEC}}=$' + '{:.2f})'.format(P_cart / mc), size='smaller')
    ax6.set_ylabel(r'$n_{|\vec{P_{I}}|}$')
    ax6.set_xlabel(r'$|\vec{P_{I}}|$')

    # print(qds_orig.sel(P=P).isel(t=-1)['Nph'].values, interp_ds.attrs['Nph_interp'], qds_orig_cart.sel(P=P_cart)['NB'].values)
    # print(qds_orig.sel(P=P).isel(t=-1)['Nph'].values, qds_orig_cart.sel(P=P_cart)['NB'].values)

    # # # POSITION DISTRIBUTIONS

    # # Interpolate 2D slice of position distribution
    # # posmult = 5
    # # kzL_xz_slice_interp = np.linspace(np.min(kzL), np.max(kzL), posmult * kzL.size); kxL_xz_slice_interp = np.linspace(np.min(kxL), np.max(kxL), posmult * kxL.size)
    # # kxLg_xz_slice_interp, kzLg_xz_slice_interp = np.meshgrid(kxL_xz_slice_interp, kzL_xz_slice_interp, indexing='ij')
    # # PhDenLg_xz_slice_interp = interpolate.griddata((kxLg_xz_slice.flatten(), kzLg_xz_slice.flatten()), PhDenLg_xz_slice.flatten(), (kxLg_xz_slice_interp, kzLg_xz_slice_interp), method='cubic')

    # # zL_xz_slice_interp = np.linspace(np.min(zL), np.max(zL), posmult * zL.size); xL_xz_slice_interp = np.linspace(np.min(xL), np.max(xL), posmult * xL.size)
    # # xLg_xz_slice_interp, zLg_xz_slice_interp = np.meshgrid(xL_xz_slice_interp, zL_xz_slice_interp, indexing='ij')
    # # np_xz_slice_interp = interpolate.griddata((xLg_xz_slice.flatten(), zLg_xz_slice.flatten()), np_xz_slice.flatten(), (xLg_xz_slice_interp, zLg_xz_slice_interp), method='cubic')
    # # na_xz_slice_interp = interpolate.griddata((xLg_xz_slice.flatten(), zLg_xz_slice.flatten()), na_xz_slice.flatten(), (xLg_xz_slice_interp, zLg_xz_slice_interp), method='cubic')

    # # xLg_xz_slice = xLg_xz_slice_interp
    # # zLg_xz_slice = zLg_xz_slice_interp
    # # np_xz_slice = np_xz_slice_interp
    # # na_xz_slice = na_xz_slice_interp
    # # # np_xz_slice = interp_ds['np_xz'].values
    # # # na_xz_slice = interp_ds['na_xz'].values

    # # print(np.any(np.isnan(PhDenLg_xz_slice_interp)))

    # # Individual Phonon Position Distribution (Interp)
    # # fig3, ax3 = plt.subplots()
    # # quad3 = ax3.pcolormesh(zLg_xz_slice, xLg_xz_slice, np_xz_slice, norm=colors.LogNorm(vmin=np.abs(np.min(np_xz_slice)), vmax=np.max(np_xz_slice)), cmap='inferno')
    # # poslinDim3 = 2300
    # # ax3.set_xlim([-1 * poslinDim3, poslinDim3])
    # # ax3.set_ylim([-1 * poslinDim3, poslinDim3])
    # # # ax3.set_xlim([-800, 800])
    # # # ax3.set_ylim([-50, 50])
    # # ax3.set_xlabel('z (Impurity Propagation Direction)')
    # # ax3.set_ylabel('x')
    # # ax3.set_title('Individual Phonon Position Distribution (Interp)')
    # # fig3.colorbar(quad3, ax=ax3, extend='both')

    # # Bare Atom Position Distribution (Interp)
    # # fig4, ax4 = plt.subplots()
    # # quad4 = ax4.pcolormesh(zLg_xz_slice, xLg_xz_slice, na_xz_slice, norm=colors.LogNorm(vmin=np.abs(np.min(na_xz_slice)), vmax=np.max(na_xz_slice)), cmap='inferno')
    # # poslinDim4 = 1300
    # # ax4.set_xlim([-1 * poslinDim4, poslinDim4])
    # # ax4.set_ylim([-1 * poslinDim4, poslinDim4])
    # # ax4.set_xlabel('z (Impurity Propagation Direction)')
    # # ax4.set_ylabel('x')
    # # ax4.set_title('Individual Atom Position Distribution (Interp)')
    # # fig4.colorbar(quad4, ax=ax4, extend='both')

    # # # SINGULARITY TEST - NOTE: proper way is to do interpolation starting from same kmin as original grid but then only take values in the new grid that with k_new >= dk_old - quick and dirty way which gets same result is to just start the new grid at 1.01*dk (not at 1*dk since that still has error)

    # print(dk)
    # kpow = 2
    # func_Vals = kg**(-kpow)
    # func_da = xr.DataArray(func_Vals, coords=[kVec, thVec], dims=['k', 'th'])
    # orig_sum = np.sum(func_Vals * kg**2 * np.sin(thg) * dk * dth * (2 * np.pi)**(-2))

    # # kmin_interp = np.min(kVec)
    # kmin_interp = 1.01 * dk
    # k_interp = np.linspace(kmin_interp, np.max(kVec), interpmul * kVec.size)
    # th_interp = np.linspace(np.min(thVec), np.max(thVec), interpmul * thVec.size)

    # kg_interp, thg_interp = np.meshgrid(k_interp, th_interp, indexing='ij')
    # func_interp = interpolate.griddata((kg.flatten(), thg.flatten()), func_da.values.flatten(), (kg_interp, thg_interp), method='linear')

    # dk_interp = kg_interp[1, 0] - kg_interp[0, 0]
    # dth_interp = thg_interp[0, 1] - thg_interp[0, 0]
    # griddata_sum = np.sum(func_interp * kg_interp**2 * np.sin(thg_interp) * dk_interp * dth_interp * (2 * np.pi)**(-2))

    # # rbfi = interpolate.Rbf(kg.flatten(), thg.flatten(), func_da.values.flatten(), function='gaussian')
    # # func_Rbf = rbfi(kg_interp, thg_interp)
    # # rbf_sum = np.sum(func_Rbf * kg_interp**2 * np.sin(thg_interp) * dk_interp * dth_interp * (2 * np.pi)**(-2))

    # # f_spline = interpolate.interp2d(x=kVec, y=thVec, z=np.transpose(func_da.values), kind='linear')
    # # func_spline = np.transpose(f_spline(k_interp, th_interp))
    # # spline_sum = np.sum(func_spline * kg_interp**2 * np.sin(thg_interp) * dk_interp * dth_interp * (2 * np.pi)**(-2))

    # exact_sum = 2 * (2 * np.pi)**(-2) * (3 - kpow)**(-1) * np.max(kVec)**(3 - kpow)  # exact answer if func = k^(-kpow) with kpow < 3
    # print(exact_sum)
    # print(orig_sum)
    # print(griddata_sum)
    # # print(rbf_sum)
    # # print(spline_sum)

    # kmin_val = dk
    # kg_mask = kg >= kmin_val
    # kg_interp_mask = kg_interp >= kmin_val
    # func_Vals_minmask = func_Vals[kg_mask]
    # func_interp_minmask = func_interp[kg_interp_mask]
    # orig_sum_min = np.sum(func_Vals_minmask * kg[kg_mask]**2 * np.sin(thg[kg_mask]) * dk * dth * (2 * np.pi)**(-2))
    # griddata_sum_min = np.sum(func_interp_minmask * kg_interp[kg_interp_mask]**2 * np.sin(thg_interp[kg_interp_mask]) * dk_interp * dth_interp * (2 * np.pi)**(-2))
    # print(orig_sum_min, griddata_sum_min)

    # fig1, ax1 = plt.subplots()
    # interp_vals = func_interp
    # kxg_interp = kg_interp * np.sin(thg_interp)
    # kzg_interp = kg_interp * np.cos(thg_interp)
    # quad1 = ax1.pcolormesh(kzg, kxg, func_Vals, norm=colors.LogNorm(vmin=1e-3, vmax=np.max(func_Vals)), cmap='inferno')
    # quad1m = ax1.pcolormesh(kzg, -1 * kxg, func_Vals, norm=colors.LogNorm(vmin=1e-3, vmax=np.max(func_Vals)), cmap='inferno')
    # fig1.colorbar(quad1, ax=ax1, extend='both')
    # fig, ax = plt.subplots()
    # quad = ax.pcolormesh(kzg_interp, kxg_interp, interp_vals, norm=colors.LogNorm(vmin=1e-3, vmax=np.max(func_Vals)), cmap='inferno')
    # quadm = ax.pcolormesh(kzg_interp, -1 * kxg_interp, interp_vals, norm=colors.LogNorm(vmin=1e-3, vmax=np.max(func_Vals)), cmap='inferno')
    # fig.colorbar(quad, ax=ax, extend='both')

    # ax1.set_xlim([-0.5, 0.5])
    # ax1.set_ylim([-0.5, 0.5])
    # ax.set_xlim([-0.5, 0.5])
    # ax.set_ylim([-0.5, 0.5])

    # plt.show()

    # END TEST

    plt.show()
