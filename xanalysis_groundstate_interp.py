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

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

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

    toggleDict = {'Location': 'work', 'Dynamics': 'imaginary', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon', 'ReducedInterp': 'true', 'kGrid_ext': 'false'}

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
    interpdatapath = innerdatapath + '/interp'
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

    # # # FULL RECONSTRUCTION OF 3D CARTESIAN BETA_K FROM 2D SPHERICAL BETA_K

    # CSAmp_ds = (qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']).isel(t=-1)
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # NphiPoints = 100  # This is the step that dramatically increases memory consumption and runtime of Cartesian griddata interpolation -> also affects quality of normalization of 3D distribution
    # phiVec = np.concatenate((np.linspace(0, np.pi, NphiPoints // 2, endpoint=False), np.linspace(np.pi, 2 * np.pi, NphiPoints // 2, endpoint=False)))
    # Bk_2D = xr.DataArray(np.full((len(kVec), len(thVec)), np.nan, dtype=complex), coords=[kVec, thVec], dims=['k', 'th'])

    # Pind = 3
    # P = PVals[Pind]
    # print('P: {0}'.format(P))

    # CSAmp_Vals = CSAmp_ds.sel(P=P).values
    # Nph = qds_aIBi.isel(t=-1).sel(P=P)['Nph'].values

    # Bk_2D[:] = CSAmp_Vals.reshape((len(kVec), len(thVec)))

    # mult = 5
    # kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # # Normalization of the original data array - this checks out
    # dk0 = kg[1, 0] - kg[0, 0]
    # dth0 = thg[0, 1] - thg[0, 0]
    # Bk_norm = (1 / Nph) * np.sum(dk0 * dth0 * (2 * np.pi)**(-2) * kg**2 * np.sin(thg) * np.abs(Bk_2D.values)**2)
    # print('Original (1/Nph)|Bk|^2 normalization (Spherical 2D): {0}'.format(Bk_norm))
    # # Interpolation of original data array onto a finer spaced (spherical) grid
    # k_interp = np.linspace(np.min(kVec), np.max(kVec), mult * kVec.size); th_interp = np.linspace(np.min(thVec), np.max(thVec), mult * thVec.size)
    # kg_interp, thg_interp = np.meshgrid(k_interp, th_interp, indexing='ij')
    # Bk_interp_vals = interpolate.griddata((kg.flatten(), thg.flatten()), Bk_2D.values.flatten(), (kg_interp, thg_interp), method='cubic')
    # # Normalization of interpolated data array (spherical coordinates) - this doesn't check out for supersonic case but does for subsonic case (why??)
    # dk = kg_interp[1, 0] - kg_interp[0, 0]
    # dth = thg_interp[0, 1] - thg_interp[0, 0]
    # Bk_interp_norm = (1 / Nph) * np.sum(dk * dth * (2 * np.pi)**(-2) * kg_interp**2 * np.sin(thg_interp) * np.abs(Bk_interp_vals)**2)
    # print('Interpolated (1/Nph)|Bk|^2 normalization (Spherical 2D): {0}'.format(Bk_interp_norm))

    # # Set reduced bounds of k-space and other things dependent on subsonic or supersonic
    # if P < 0.9:
    #     [vmin, vmax] = [0, 500]
    #     linDimMajor = 1.5
    #     linDimMinor = 1.5
    #     ext_major_rat = 0.35
    #     ext_minor_rat = 0.35
    #     Npoints = 252  # actual number of points will be ~Npoints-1, want Npoints=2502 (gives 2500 points)
    # else:
    #     [vmin, vmax] = [0, 9.2e13]
    #     linDimMajor = 0.1
    #     linDimMinor = 0.01
    #     ext_major_rat = 0.025
    #     ext_minor_rat = 0.0025
    #     Npoints = 252  # (gives 250 points)

    # # Remove k values outside reduced k-space bounds (as |Bk|~0 there) and save the average of these values to add back in later before FFT
    # kred_ind = np.argwhere(kg_interp[:, 0] > (1.5 * linDimMajor))[0][0]
    # kg_red = np.delete(kg_interp, np.arange(kred_ind, k_interp.size), 0)
    # thg_red = np.delete(thg_interp, np.arange(kred_ind, k_interp.size), 0)
    # Bk_red = np.delete(Bk_interp_vals, np.arange(kred_ind, k_interp.size), 0)
    # Bk_remainder = np.delete(Bk_interp_vals, np.arange(0, kred_ind), 0)
    # Bk_rem_ave = np.average(Bk_remainder)
    # k_interp_red = k_interp[0:kred_ind]
    # kmax_rem = np.max(k_interp)

    # if toggleDict['ReducedInterp'] == 'true':
    #     k_interp = k_interp_red
    #     kg_interp = kg_red
    #     thg_interp = thg_red
    #     Bk_interp_vals = Bk_red

    # # CHECK WHY ALL BK AMPLITUDES HAVE ZERO IMAGINARY PART, EVEN FOR SUPERSONIC CASE? IS THIS BECAUSE ITS THE GROUNDSTATE?
    # # print(np.imag(Bk_2D.values))
    # # print(np.imag(Bk_interp_vals))

    # # 3D reconstruction in spherical coordinates (copy interpolated 2D spherical Bk onto all phi coordinates due to phi symmetry)
    # phi_interp, dphi = np.linspace(np.min(phiVec), np.max(phiVec), 1 * phiVec.size, retstep=True)
    # Bk_3D = xr.DataArray(np.full((len(k_interp), len(th_interp), len(phi_interp)), np.nan, dtype=complex), coords=[k_interp, th_interp, phi_interp], dims=['k', 'th', 'phi'])
    # for phiInd, phi in enumerate(phi_interp):
    #     Bk_3D.sel(phi=phi)[:] = Bk_interp_vals

    # # Re-interpret grid points of 3D spherical reconstruction as nonlinear 3D cartesian grid
    # kg_3Di, thg_3Di, phig_3Di = np.meshgrid(k_interp, th_interp, phi_interp, indexing='ij')
    # kxg = kg_3Di * np.sin(thg_3Di) * np.cos(phig_3Di)
    # kyg = kg_3Di * np.sin(thg_3Di) * np.sin(phig_3Di)
    # kzg = kg_3Di * np.cos(thg_3Di)
    # (Nk, Nth, Nphi) = kzg.shape

    # Bk_3D_vals = Bk_3D.values
    # Bk_3D_vals[np.isnan(Bk_3D_vals)] = 0

    # # dphi = phi_interp[1] - phi_interp[0]
    # Bk_Sph3D_norm = (1 / Nph) * np.sum(dk * dth * dphi * (2 * np.pi)**(-3) * kg_3Di**2 * np.sin(thg_3Di) * np.abs(Bk_3D_vals)**2)
    # print('Interpolated (1/Nph)|Bk|^2 normalization (Spherical 3D): {0}'.format(Bk_Sph3D_norm))

    # # Create linear 3D cartesian grid and reinterpolate Bk_3D onto this grid
    # kxL_pos, dkxL = np.linspace(0, linDimMinor, Npoints // 2, retstep=True, endpoint=False); kxL = np.concatenate((-1 * np.flip(kxL_pos[1:], axis=0), kxL_pos))
    # kyL_pos, dkyL = np.linspace(0, linDimMinor, Npoints // 2, retstep=True, endpoint=False); kyL = np.concatenate((-1 * np.flip(kyL_pos[1:], axis=0), kyL_pos))
    # kzL_pos, dkzL = np.linspace(0, linDimMajor, Npoints // 2, retstep=True, endpoint=False); kzL = np.concatenate((-1 * np.flip(kzL_pos[1:], axis=0), kzL_pos))
    # kzLg_3D, kxLg_3D, kyLg_3D = np.meshgrid(kzL, kxL, kyL, indexing='ij')

    # print('Spherical Interp Grid Shape: {0}'.format(kzg.shape))
    # print('Cartesian Interp Grid Shape: {0}'.format(kzLg_3D.shape))
    # interpstart = timer()
    # BkLg_3D = interpolate.griddata((kzg.flatten(), kxg.flatten(), kyg.flatten()), Bk_3D_vals.flatten(), (kzLg_3D, kxLg_3D, kyLg_3D), method='nearest')
    # interpend = timer()
    # print('Interp Time: {0}'.format(interpend - interpstart))

    # # dkzL = kzL[1] - kzL[0]; dkxL = kxL[1] - kxL[0]; dkyL = kyL[1] - kyL[0]
    # BkLg_3D[np.isnan(BkLg_3D)] = 0
    # BkLg_3D_norm = (1 / Nph) * np.sum(dkzL * dkzL * dkyL * np.abs(BkLg_3D)**2)
    # print('Linear grid (1/Nph)|Bk|^2 normalization (Cartesian 3D): {0}'.format(BkLg_3D_norm))

    # # Consistency check: use 2D ky=0 slice of |Bk|^2 to calculate phonon density and compare it to phonon density from original spherical interpolated data

    # kxL_0ind = kxL.size // 2; kyL_0ind = kyL.size // 2; kzL_0ind = kzL.size // 2  # find position of zero of each axis: kxL=0, kyL=0, kzL=0
    # kxLg_ky0slice = kxLg_3D[:, :, kyL_0ind]
    # kzLg_ky0slice = kzLg_3D[:, :, kyL_0ind]
    # BkLg_ky0slice = BkLg_3D[:, :, kyL_0ind]
    # PhDen_Lg_ky0slice = ((1 / Nph) * np.abs(BkLg_ky0slice)**2).real.astype(float)
    # PhDen_Sph = ((1 / Nph) * np.abs(Bk_interp_vals)**2).real.astype(float)
    # kxg_Sph = kg_interp * np.sin(thg_interp)
    # kzg_Sph = kg_interp * np.cos(thg_interp)

    # # Add the remainder of Bk back in (values close to zero for large k)
    # if toggleDict['ReducedInterp'] == 'true' and toggleDict['kGrid_ext'] == 'true':
    #     kL_max_major = ext_major_rat * kmax_rem / np.sqrt(2)
    #     kL_max_minor = ext_minor_rat * kmax_rem / np.sqrt(2)
    #     print('kL_red_max_major: {0}, kL_ext_max_major: {1}, dkL_major: {2}'.format(np.max(kzL), kL_max_major, dkzL))
    #     print('kL_red_max_minor: {0}, kL_ext_max_minor: {1}, dkL_minor: {2}'.format(np.max(kxL), kL_max_minor, dkxL))
    #     kx_addon = np.arange(linDimMinor, kL_max_minor, dkxL); ky_addon = np.arange(linDimMinor, kL_max_minor, dkyL); kz_addon = np.arange(linDimMajor, kL_max_major, dkzL)
    #     print('kL_ext_addon size -  major: {0}, minor: {1}'.format(2 * kz_addon.size, 2 * kx_addon.size))
    #     kxL_ext = np.concatenate((-1 * np.flip(kx_addon, axis=0), np.concatenate((kxL, kx_addon))))
    #     kyL_ext = np.concatenate((-1 * np.flip(ky_addon, axis=0), np.concatenate((kyL, kx_addon))))
    #     kzL_ext = np.concatenate((-1 * np.flip(kz_addon, axis=0), np.concatenate((kzL, kx_addon))))

    #     ax = kxL.size; ay = kyL.size; az = kzL.size
    #     mx = kx_addon.size; my = ky_addon.size; mz = kz_addon.size
    #     BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones((mz, ax, ay)), np.concatenate((BkLg_3D, Bk_rem_ave * np.ones((mz, ax, ay))), axis=0)), axis=0)
    #     BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones(((az + 2 * mz), mx, ay)), np.concatenate((BkLg_3D_ext, Bk_rem_ave * np.ones(((az + 2 * mz), mx, ay))), axis=1)), axis=1)
    #     BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones(((az + 2 * mz), (ax + 2 * mx), my)), np.concatenate((BkLg_3D_ext, Bk_rem_ave * np.ones(((az + 2 * mz), (ax + 2 * mx), my))), axis=2)), axis=2)

    #     kxL = kxL_ext; kyL = kyL_ext; kzL = kzL_ext
    #     BkLg_3D = BkLg_3D_ext
    #     print('Cartesian Interp Extended Grid Shape: {0}'.format(BkLg_3D.shape))

    # # Fourier Transform to get 3D position distribution

    # zL = np.fft.fftshift(np.fft.fftfreq(kzL.size) * 2 * np.pi / dkzL)
    # xL = np.fft.fftshift(np.fft.fftfreq(kxL.size) * 2 * np.pi / dkxL)
    # yL = np.fft.fftshift(np.fft.fftfreq(kyL.size) * 2 * np.pi / dkyL)
    # dzL = zL[1] - zL[0]; dxL = xL[1] - xL[0]; dyL = yL[1] - yL[0]
    # dVzxy = dxL * dyL * dzL
    # # print(dzL, 2 * np.pi / (kzL.size * dkzL))

    # zLg_3D, xLg_3D, yLg_3D = np.meshgrid(zL, xL, yL, indexing='ij')
    # beta_kzkxky = np.fft.ifftshift(BkLg_3D)
    # amp_beta_zxy_preshift = np.fft.ifftn(beta_kzkxky) / dVzxy
    # amp_beta_zxy = np.fft.fftshift(amp_beta_zxy_preshift)
    # nzxy = ((1 / Nph) * np.abs(amp_beta_zxy)**2).real.astype(float)
    # nzxy_norm = np.sum(dVzxy * nzxy)
    # print('Linear grid (1/Nph)*n(x,y,z) normalization (Cartesian 3D): {0}'.format(nzxy_norm))

    # # Take 2D slices of position distribution
    # zLg_y0slice = zLg_3D[:, :, yL.size // 2]
    # xLg_y0slice = xLg_3D[:, :, yL.size // 2]
    # nzxy_y0slice = nzxy[:, :, yL.size // 2]

    # # Create DataSet for 3D Betak and position distribution slices

    # ReCSA_da = xr.DataArray(np.real(BkLg_3D), coords=[kzL, kxL, kyL], dims=['kz', 'kx', 'ky'])
    # ImCSA_da = xr.DataArray(np.imag(BkLg_3D), coords=[kzL, kxL, kyL], dims=['kz', 'kx', 'ky'])
    # nzxy_da = xr.DataArray(nzxy, coords=[zL, xL, yL], dims=['z', 'x', 'y'])

    # data_dict = {'ReCSA': ReCSA_da, 'ImCSA': ImCSA_da, 'nzxy': nzxy_da}
    # coords_dict = {'kx': kxL, 'ky': kyL, 'kz': kzL, 'x': xL, 'y': yL, 'z': zL}
    # attrs_dict = {'P': P, 'aIBi': aIBi}
    # interp_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    # interp_ds.to_netcdf(interpdatapath + '/InterpDat_P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    # # All Plotting: (a) 2D ky=0 slice of |Bk|^2, (b) 2D slice of position distribution

    # fig, axes = plt.subplots(nrows=1, ncols=2)
    # axes[0].set_xlim([-1 * linDimMajor, linDimMajor])
    # axes[0].set_ylim([-1 * linDimMinor, linDimMinor])
    # axes[1].set_xlim([-1 * linDimMajor, linDimMajor])
    # axes[1].set_ylim([-1 * linDimMinor, linDimMinor])

    # quad1 = axes[0].pcolormesh(kzg_Sph, kxg_Sph, PhDen_Sph[:-1, :-1], vmin=vmin, vmax=vmax)
    # quad1m = axes[0].pcolormesh(kzg_Sph, -1 * kxg_Sph, PhDen_Sph[:-1, :-1], vmin=vmin, vmax=vmax)
    # fig.colorbar(quad1, ax=axes[0], extend='both')
    # quad2 = axes[1].pcolormesh(kzLg_ky0slice, kxLg_ky0slice, PhDen_Lg_ky0slice[:-1, :-1], vmin=vmin, vmax=vmax)
    # fig.colorbar(quad2, ax=axes[1], extend='both')

    # fig2, ax2 = plt.subplots()
    # quad3 = ax2.pcolormesh(zLg_y0slice, xLg_y0slice, nzxy_y0slice, vmin=0, vmax=np.max(nzxy_y0slice))
    # # ax2.set_xlim([-200, 200])
    # # ax2.set_ylim([-3e3, 3e3])
    # fig2.colorbar(quad3, ax=ax2, extend='both')

    # plt.show()

    # # FULL RECONSTRUCTION OF 3D CARTESIAN BETA_K FROM 2D SPHERICAL BETA_K (doing actual interpolation in 2D spherical instead of 3D nonlinear cartesian)

    CSAmp_ds = (qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']).isel(t=-1)
    kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    kVec = kgrid.getArray('k')
    thVec = kgrid.getArray('th')
    NphiPoints = 100  # This is the step that dramatically increases memory consumption and runtime of Cartesian griddata interpolation -> also affects quality of normalization of 3D distribution
    phiVec = np.concatenate((np.linspace(0, np.pi, NphiPoints // 2, endpoint=False), np.linspace(np.pi, 2 * np.pi, NphiPoints // 2, endpoint=False)))
    Bk_2D = xr.DataArray(np.full((len(kVec), len(thVec)), np.nan, dtype=complex), coords=[kVec, thVec], dims=['k', 'th'])

    Pind = 3
    P = PVals[Pind]
    print('P: {0}'.format(P))

    CSAmp_Vals = CSAmp_ds.sel(P=P).values
    Nph = qds_aIBi.isel(t=-1).sel(P=P)['Nph'].values

    Bk_2D[:] = CSAmp_Vals.reshape((len(kVec), len(thVec)))

    mult = 5
    kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # Normalization of the original data array - this checks out
    dk0 = kg[1, 0] - kg[0, 0]
    dth0 = thg[0, 1] - thg[0, 0]
    Bk_norm = (1 / Nph) * np.sum(dk0 * dth0 * (2 * np.pi)**(-2) * kg**2 * np.sin(thg) * np.abs(Bk_2D.values)**2)
    print('Original (1/Nph)|Bk|^2 normalization (Spherical 2D): {0}'.format(Bk_norm))
    # Interpolation of original data array onto a finer spaced (spherical) grid
    k_interp = np.linspace(np.min(kVec), np.max(kVec), mult * kVec.size); th_interp = np.linspace(np.min(thVec), np.max(thVec), mult * thVec.size)
    kg_interp, thg_interp = np.meshgrid(k_interp, th_interp, indexing='ij')
    Bk_interp_vals = interpolate.griddata((kg.flatten(), thg.flatten()), Bk_2D.values.flatten(), (kg_interp, thg_interp), method='cubic')
    # Normalization of interpolated data array (spherical coordinates) - this doesn't check out for supersonic case but does for subsonic case (why??)
    dk = kg_interp[1, 0] - kg_interp[0, 0]
    dth = thg_interp[0, 1] - thg_interp[0, 0]
    Bk_interp_norm = (1 / Nph) * np.sum(dk * dth * (2 * np.pi)**(-2) * kg_interp**2 * np.sin(thg_interp) * np.abs(Bk_interp_vals)**2)
    print('Interpolated (1/Nph)|Bk|^2 normalization (Spherical 2D): {0}'.format(Bk_interp_norm))

    # Set reduced bounds of k-space and other things dependent on subsonic or supersonic
    if P < 0.9:
        [vmin, vmax] = [0, 500]
        linDimMajor = 1.5
        linDimMinor = 1.5
        ext_major_rat = 0.35
        ext_minor_rat = 0.35
        Npoints = 45  # actual number of points will be ~Npoints-1, want Npoints=2502 (gives 2500 points)
    else:
        [vmin, vmax] = [0, 9.2e13]
        linDimMajor = 0.1
        linDimMinor = 0.01
        ext_major_rat = 0.025
        ext_minor_rat = 0.0025
        Npoints = 30  # (gives 250 points)

    # Remove k values outside reduced k-space bounds (as |Bk|~0 there) and save the average of these values to add back in later before FFT
    kred_ind = np.argwhere(kg_interp[:, 0] > (1.5 * linDimMajor))[0][0]
    kg_red = np.delete(kg_interp, np.arange(kred_ind, k_interp.size), 0)
    thg_red = np.delete(thg_interp, np.arange(kred_ind, k_interp.size), 0)
    Bk_red = np.delete(Bk_interp_vals, np.arange(kred_ind, k_interp.size), 0)
    Bk_remainder = np.delete(Bk_interp_vals, np.arange(0, kred_ind), 0)
    Bk_rem_ave = np.average(Bk_remainder)
    k_interp_red = k_interp[0:kred_ind]
    kmax_rem = np.max(k_interp)

    if toggleDict['ReducedInterp'] == 'true':
        k_interp = k_interp_red
        kg_interp = kg_red
        thg_interp = thg_red
        Bk_interp_vals = Bk_red

    # CHECK WHY ALL BK AMPLITUDES HAVE ZERO IMAGINARY PART, EVEN FOR SUPERSONIC CASE? IS THIS BECAUSE ITS THE GROUNDSTATE?
    # print(np.imag(Bk_2D.values))
    # print(np.imag(Bk_interp_vals))

    # Create linear 3D cartesian grid and reinterpolate Bk_3D onto this grid
    kxL_pos, dkxL = np.linspace(1e-10, linDimMinor, Npoints // 2, retstep=True, endpoint=False); kxL = np.concatenate((1e-10 - 1 * np.flip(kxL_pos[1:], axis=0), kxL_pos))
    kyL_pos, dkyL = np.linspace(1e-10, linDimMinor, Npoints // 2, retstep=True, endpoint=False); kyL = np.concatenate((1e-10 - 1 * np.flip(kyL_pos[1:], axis=0), kyL_pos))
    kzL_pos, dkzL = np.linspace(1e-10, linDimMajor, Npoints // 2, retstep=True, endpoint=False); kzL = np.concatenate((1e-10 - 1 * np.flip(kzL_pos[1:], axis=0), kzL_pos))
    kzLg_3D, kxLg_3D, kyLg_3D = np.meshgrid(kzL, kxL, kyL, indexing='ij')

    # Re-interpret grid points of linear 3D Cartesian as nonlinear 3D spherical grid
    kg_3Di = np.sqrt(kxLg_3D**2 + kyLg_3D**2 + kzLg_3D**2)
    thg_3Di = np.arccos(kzLg_3D / kg_3Di)
    phig_3Di = np.arctan2(kyLg_3D, kxLg_3D)

    # print('k')
    # print(np.min(kg_3Di), np.max(kg_3Di))
    # print('\n' + 'th')
    # print(np.min(thg_3Di), np.max(thg_3Di))
    # print('\n' + 'phi')
    # print(np.min(phig_3Di), np.max(phig_3Di))

    k_3Di_unique, k_3Di_inverse = np.unique(kg_3Di, return_inverse=True)
    th_3Di_unique, th_3Di_inverse = np.unique(thg_3Di, return_inverse=True)
    print('Nk unique: {:1.2E}, Nth unique: {:1.2E}, Ntot unique: {:1.2E}'.format(k_3Di_unique.size, th_3Di_unique.size, k_3Di_unique.size * th_3Di_unique.size))
    print('3D Cartesian grid Ntot: {:1.2E}'.format(kzLg_3D.size))
    # print(kg_3Di_unique.size / kg_3Di.size)
    # print(thg_3Di_unique.size / thg_3Di.size)

    kg_3Di_unique, thg_3Di_unique = np.meshgrid(k_3Di_unique, th_3Di_unique, indexing='ij')
    interpstart = timer()
    Bk_2D_CartInt = interpolate.griddata((kg_interp.flatten(), thg_interp.flatten()), Bk_interp_vals.flatten(), (kg_3Di_unique, thg_3Di_unique), method='cubic')
    # Bk_2D_Rbf = interpolate.Rbf(kg, thg, Bk_2D.values)
    # Bk_2D_CartInt = Bk_2D_Rbf(kg_3Di_unique, thg_3Di_unique)
    interpend = timer()
    print('Interp Time: {0}'.format(interpend - interpstart))
    reconstart = timer()
    BkLg_3D = np.zeros(kzLg_3D.shape, dtype='complex')
    # think about playing with zip (zip kg_3Di_unique & thg_2Di_unique and compare to zipped kg_3Di & thg_3Di)
    (Nkz, Nkx, Nky) = kg_3Di.shape
    for iz in np.arange(Nkz):
        for ix in np.arange(Nkx):
            for iy in np.arange(Nky):
                kv = kg_3Di[iz, ix, iy]
                thv = thg_3Di[iz, ix, iy]
                ik_un = np.argwhere(k_3Di_unique == kv)[0][0]
                ith_un = np.argwhere(th_3Di_unique == thv)[0][0]
                BkLg_3D[iz, ix, iy] = Bk_2D_CartInt[ik_un, ith_un]
    reconend = timer()
    print('Recon Time: {0}'.format(reconend - reconstart))

    BkLg_3D[np.isnan(BkLg_3D)] = 0
    BkLg_3D_norm = (1 / Nph) * np.sum(dkzL * dkzL * dkyL * np.abs(BkLg_3D)**2)
    print('Linear grid (1/Nph)|Bk|^2 normalization (Cartesian 3D): {0}'.format(BkLg_3D_norm))

    # Consistency check: use 2D ky=0 slice of |Bk|^2 to calculate phonon density and compare it to phonon density from original spherical interpolated data

    kxL_0ind = kxL.size // 2; kyL_0ind = kyL.size // 2; kzL_0ind = kzL.size // 2  # find position of zero of each axis: kxL=0, kyL=0, kzL=0
    kxLg_ky0slice = kxLg_3D[:, :, kyL_0ind]
    kzLg_ky0slice = kzLg_3D[:, :, kyL_0ind]
    BkLg_ky0slice = BkLg_3D[:, :, kyL_0ind]
    PhDen_Lg_ky0slice = ((1 / Nph) * np.abs(BkLg_ky0slice)**2).real.astype(float)
    PhDen_Sph = ((1 / Nph) * np.abs(Bk_interp_vals)**2).real.astype(float)
    kxg_Sph = kg_interp * np.sin(thg_interp)
    kzg_Sph = kg_interp * np.cos(thg_interp)

    # Add the remainder of Bk back in (values close to zero for large k)
    if toggleDict['ReducedInterp'] == 'true' and toggleDict['kGrid_ext'] == 'true':
        kL_max_major = ext_major_rat * kmax_rem / np.sqrt(2)
        kL_max_minor = ext_minor_rat * kmax_rem / np.sqrt(2)
        print('kL_red_max_major: {0}, kL_ext_max_major: {1}, dkL_major: {2}'.format(np.max(kzL), kL_max_major, dkzL))
        print('kL_red_max_minor: {0}, kL_ext_max_minor: {1}, dkL_minor: {2}'.format(np.max(kxL), kL_max_minor, dkxL))
        kx_addon = np.arange(linDimMinor, kL_max_minor, dkxL); ky_addon = np.arange(linDimMinor, kL_max_minor, dkyL); kz_addon = np.arange(linDimMajor, kL_max_major, dkzL)
        print('kL_ext_addon size -  major: {0}, minor: {1}'.format(2 * kz_addon.size, 2 * kx_addon.size))
        kxL_ext = np.concatenate((-1 * np.flip(kx_addon, axis=0), np.concatenate((kxL, kx_addon))))
        kyL_ext = np.concatenate((-1 * np.flip(ky_addon, axis=0), np.concatenate((kyL, kx_addon))))
        kzL_ext = np.concatenate((-1 * np.flip(kz_addon, axis=0), np.concatenate((kzL, kx_addon))))

        ax = kxL.size; ay = kyL.size; az = kzL.size
        mx = kx_addon.size; my = ky_addon.size; mz = kz_addon.size
        BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones((mz, ax, ay)), np.concatenate((BkLg_3D, Bk_rem_ave * np.ones((mz, ax, ay))), axis=0)), axis=0)
        BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones(((az + 2 * mz), mx, ay)), np.concatenate((BkLg_3D_ext, Bk_rem_ave * np.ones(((az + 2 * mz), mx, ay))), axis=1)), axis=1)
        BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones(((az + 2 * mz), (ax + 2 * mx), my)), np.concatenate((BkLg_3D_ext, Bk_rem_ave * np.ones(((az + 2 * mz), (ax + 2 * mx), my))), axis=2)), axis=2)

        kxL = kxL_ext; kyL = kyL_ext; kzL = kzL_ext
        BkLg_3D = BkLg_3D_ext
        print('Cartesian Interp Extended Grid Shape: {0}'.format(BkLg_3D.shape))

    # Fourier Transform to get 3D position distribution

    zL = np.fft.fftshift(np.fft.fftfreq(kzL.size) * 2 * np.pi / dkzL)
    xL = np.fft.fftshift(np.fft.fftfreq(kxL.size) * 2 * np.pi / dkxL)
    yL = np.fft.fftshift(np.fft.fftfreq(kyL.size) * 2 * np.pi / dkyL)
    dzL = zL[1] - zL[0]; dxL = xL[1] - xL[0]; dyL = yL[1] - yL[0]
    dVzxy = dxL * dyL * dzL
    # print(dzL, 2 * np.pi / (kzL.size * dkzL))

    zLg_3D, xLg_3D, yLg_3D = np.meshgrid(zL, xL, yL, indexing='ij')
    beta_kzkxky = np.fft.ifftshift(BkLg_3D)
    amp_beta_zxy_preshift = np.fft.ifftn(beta_kzkxky) / dVzxy
    amp_beta_zxy = np.fft.fftshift(amp_beta_zxy_preshift)
    nzxy = ((1 / Nph) * np.abs(amp_beta_zxy)**2).real.astype(float)
    nzxy_norm = np.sum(dVzxy * nzxy)
    print('Linear grid (1/Nph)*n(x,y,z) normalization (Cartesian 3D): {0}'.format(nzxy_norm))

    # Take 2D slices of position distribution
    zLg_y0slice = zLg_3D[:, :, yL.size // 2]
    xLg_y0slice = xLg_3D[:, :, yL.size // 2]
    nzxy_y0slice = nzxy[:, :, yL.size // 2]

    # Create DataSet for 3D Betak and position distribution slices

    ReCSA_da = xr.DataArray(np.real(BkLg_3D), coords=[kzL, kxL, kyL], dims=['kz', 'kx', 'ky'])
    ImCSA_da = xr.DataArray(np.imag(BkLg_3D), coords=[kzL, kxL, kyL], dims=['kz', 'kx', 'ky'])
    nzxy_da = xr.DataArray(nzxy, coords=[zL, xL, yL], dims=['z', 'x', 'y'])

    data_dict = {'ReCSA': ReCSA_da, 'ImCSA': ImCSA_da, 'nzxy': nzxy_da}
    coords_dict = {'kx': kxL, 'ky': kyL, 'kz': kzL, 'x': xL, 'y': yL, 'z': zL}
    attrs_dict = {'P': P, 'aIBi': aIBi}
    interp_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    interp_ds.to_netcdf(interpdatapath + '/InterpDat_P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    # All Plotting: (a) 2D ky=0 slice of |Bk|^2, (b) 2D slice of position distribution

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_xlim([-1 * linDimMajor, linDimMajor])
    axes[0].set_ylim([-1 * linDimMinor, linDimMinor])
    axes[1].set_xlim([-1 * linDimMajor, linDimMajor])
    axes[1].set_ylim([-1 * linDimMinor, linDimMinor])

    quad1 = axes[0].pcolormesh(kzg_Sph, kxg_Sph, PhDen_Sph[:-1, :-1], vmin=vmin, vmax=vmax)
    quad1m = axes[0].pcolormesh(kzg_Sph, -1 * kxg_Sph, PhDen_Sph[:-1, :-1], vmin=vmin, vmax=vmax)
    fig.colorbar(quad1, ax=axes[0], extend='both')
    quad2 = axes[1].pcolormesh(kzLg_ky0slice, kxLg_ky0slice, PhDen_Lg_ky0slice[:-1, :-1], vmin=vmin, vmax=vmax)
    fig.colorbar(quad2, ax=axes[1], extend='both')

    fig2, ax2 = plt.subplots()
    quad3 = ax2.pcolormesh(zLg_y0slice, xLg_y0slice, nzxy_y0slice, vmin=0, vmax=np.max(nzxy_y0slice))
    # ax2.set_xlim([-200, 200])
    # ax2.set_ylim([-3e3, 3e3])
    fig2.colorbar(quad3, ax=ax2, extend='both')

    plt.show()
