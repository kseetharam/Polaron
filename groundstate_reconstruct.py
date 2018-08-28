import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pf_dynamic_cart as pfc
import Grid
from scipy import interpolate
from timeit import default_timer as timer


if __name__ == "__main__":

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (105, 105, 105)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'imaginary', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon'}

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

    # # Analysis of Total Dataset
    interpdatapath = innerdatapath + '/interp'
    aIBi = -10
    Pind = 10

    # linDimMajor = 6.5
    # linDimMinor = 6.5
    linDimMajor = 0.2
    linDimMinor = 0.02
    dkxL = 1e-4; dkyL = 1e-4; dkzL = 1e-3

    qds = xr.open_dataset(innerdatapath + '/quench_Dataset_aIBi_{:.2f}.nc'.format(aIBi))
    qds_aIBi = qds

    PVals = qds['P'].values
    tVals = qds['t'].values
    n0 = qds.attrs['n0']
    gBB = qds.attrs['gBB']
    nu = pfc.nu(gBB)
    mI = qds.attrs['mI']
    mB = qds.attrs['mB']
    P = PVals[Pind]

    # # # FULL RECONSTRUCTION OF 3D CARTESIAN BETA_K FROM 2D SPHERICAL BETA_K (doing actual interpolation in 2D spherical instead of 3D nonlinear cartesian)

    # CSAmp_ds = (qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']).isel(t=-1)
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')

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

    # # Create linear 3D cartesian grid and reinterpolate Bk_3D onto this grid
    # kxL_pos = np.arange(1e-10, linDimMinor, dkxL); kxL = np.concatenate((1e-10 - 1 * np.flip(kxL_pos[1:], axis=0), kxL_pos))
    # kyL_pos = np.arange(1e-10, linDimMinor, dkyL); kyL = np.concatenate((1e-10 - 1 * np.flip(kyL_pos[1:], axis=0), kyL_pos))
    # kzL_pos = np.arange(1e-10, linDimMajor, dkzL); kzL = np.concatenate((1e-10 - 1 * np.flip(kzL_pos[1:], axis=0), kzL_pos))

    # print('size - kxL: {0}, kyL: {1}, kzL: {2}'.format(kxL.size, kyL.size, kzL.size))

    # kxLg_3D, kyLg_3D, kzLg_3D = np.meshgrid(kxL, kyL, kzL, indexing='ij')

    # print('dkxL: {0}, dkyL: {1}, dkzL: {2}'.format(dkxL, dkyL, dkzL))

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
    # interpend = timer()
    # print('Interp Time: {0}'.format(interpend - interpstart))
    # BkLg_3D_flat = Bk_2D_CartInt[tups_inverse]
    # BkLg_3D = BkLg_3D_flat.reshape(kg_3Di.shape)

    # BkLg_3D[np.isnan(BkLg_3D)] = 0
    # PhDenLg_3D = ((1 / Nph) * np.abs(BkLg_3D)**2).real.astype(float)
    # BkLg_3D_norm = np.sum(dkxL * dkyL * dkzL * (2 * np.pi)**(-3) * PhDenLg_3D)
    # print('Interpolated (1/Nph)|Bk|^2 normalization (Linear Cartesian 3D): {0}'.format(BkLg_3D_norm))

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

    # # Consistency check: use 2D ky=0 slice of |Bk|^2 to calculate phonon density and compare it to phonon density from original spherical interpolated data
    # Nx = len(kxL); Ny = len(kyL); Nz = len(kzL)
    # PhDenLg_xz_slice = PhDenLg_3D[:, Ny // 2, :]
    # PhDenLg_xy_slice = PhDenLg_3D[:, :, Nz // 2]
    # PhDenLg_yz_slice = PhDenLg_3D[Nx // 2, :, :]
    # np_xyz = nxyz
    # np_xz_slice = np_xyz[:, Ny // 2, :]
    # np_xy_slice = np_xyz[:, :, Nz // 2]
    # np_yz_slice = np_xyz[Nx // 2, :, :]
    # na_xyz = na_xyz_norm
    # na_xz_slice = na_xyz[:, Ny // 2, :]
    # na_xy_slice = na_xyz[:, :, Nz // 2]
    # na_yz_slice = na_xyz[Nx // 2, :, :]

    # # Create DataSet for 3D Betak and position distribution slices
    # PhDen_xz_slice_da = xr.DataArray(PhDenLg_xz_slice, coords=[kxL, kzL], dims=['kx', 'kz'])
    # PhDen_xy_slice_da = xr.DataArray(PhDenLg_xy_slice, coords=[kxL, kyL], dims=['kx', 'ky'])
    # PhDen_yz_slice_da = xr.DataArray(PhDenLg_yz_slice, coords=[kyL, kzL], dims=['ky', 'kz'])
    # np_xz_slice_da = xr.DataArray(np_xz_slice, coords=[xL, zL], dims=['x', 'z'])
    # np_xy_slice_da = xr.DataArray(np_xy_slice, coords=[xL, yL], dims=['x', 'y'])
    # np_yz_slice_da = xr.DataArray(np_yz_slice, coords=[yL, zL], dims=['y', 'z'])
    # na_xz_slice_da = xr.DataArray(na_xz_slice, coords=[xL, zL], dims=['x', 'z'])
    # na_xy_slice_da = xr.DataArray(na_xy_slice, coords=[xL, yL], dims=['x', 'y'])
    # na_yz_slice_da = xr.DataArray(na_yz_slice, coords=[yL, zL], dims=['y', 'z'])

    # data_dict = {'PhDen_xz': PhDen_xz_slice_da, 'PhDen_xy': PhDen_xy_slice_da, 'PhDen_yz': PhDen_yz_slice_da, 'np_xz': np_xz_slice_da, 'np_xy': np_xy_slice_da, 'np_yz': np_yz_slice_da, 'na_xz': na_xz_slice_da, 'na_xy': na_xy_slice_da, 'na_yz': na_yz_slice_da, }
    # coords_dict = {'kx': kxL, 'ky': kyL, 'kz': kzL, 'x': xL, 'y': yL, 'z': zL}
    # attrs_dict = {'P': P, 'aIBi': aIBi}
    # interp_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    # interp_ds.to_netcdf(interpdatapath + '/InterpDat_P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    # Plot

    interp_ds = xr.open_dataset(interpdatapath + '/InterpDat_P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    kxL = interp_ds['kx'].values
    kzL = interp_ds['kz'].values
    xL = interp_ds['x'].values
    zL = interp_ds['z'].values
    kxLg_xz_slice, kzLg_xz_slice = np.meshgrid(kxL, kzL, indexing='ij')
    xLg_xz_slice, zLg_xz_slice = np.meshgrid(xL, zL, indexing='ij')

    # # Take 2D slices of coordinates
    # kxLg_xz_slice = kxLg_3D[:, Ny // 2, :]
    # kzLg_xz_slice = kzLg_3D[:, Ny // 2, :]
    # zLg_xz_slice = zLg_3D[:, Ny // 2, :]
    # xLg_xz_slice = xLg_3D[:, Ny // 2, :]

    # All Plotting: (a) 2D ky=0 slice of |Bk|^2, (b) 2D slice of position distribution

    # fig1, ax1 = plt.subplots()
    # quad1 = ax1.pcolormesh(kzg_Sph, kxg_Sph, PhDen_Sph[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(PhDen_Sph)), vmax=np.max(PhDen_Sph)), cmap='inferno')
    # quad1m = ax1.pcolormesh(kzg_Sph, -1 * kxg_Sph, PhDen_Sph[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(PhDen_Sph)), vmax=np.max(PhDen_Sph)), cmap='inferno')
    # ax1.set_xlim([-1 * linDimMajor, linDimMajor])
    # ax1.set_ylim([-1 * linDimMinor, linDimMinor])
    # ax1.set_xlabel('kz (Impurity Propagation Direction)')
    # ax1.set_ylabel('kx')
    # ax1.set_title('Individual Phonon Momentum Distribution (Data)')
    # fig1.colorbar(quad1, ax=ax1, extend='both')

    fig2, ax2 = plt.subplots()
    quad2 = ax2.pcolormesh(kzLg_xz_slice, kxLg_xz_slice, interp_ds['PhDen_xz'].values[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(interp_ds['PhDen_xz'].values)), vmax=np.max(interp_ds['PhDen_xz'].values)), cmap='inferno')
    ax2.set_xlim([-1 * linDimMajor, linDimMajor])
    ax2.set_ylim([-1 * linDimMinor, linDimMinor])
    ax2.set_xlabel('kz (Impurity Propagation Direction)')
    ax2.set_ylabel('kx')
    ax2.set_title('Individual Phonon Momentum Distribution (Interp)')
    fig2.colorbar(quad2, ax=ax2, extend='both')

    fig3, ax3 = plt.subplots()
    quad3 = ax3.pcolormesh(zLg_xz_slice, xLg_xz_slice, interp_ds['np_xz'].values[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(interp_ds['np_xz'].values)), vmax=np.max(interp_ds['np_xz'].values)), cmap='inferno')
    ax3.set_xlabel('z (Impurity Propagation Direction)')
    ax3.set_ylabel('x')
    ax3.set_title('Individual Phonon Position Distribution (Interp)')
    fig3.colorbar(quad3, ax=ax3, extend='both')

    fig4, ax4 = plt.subplots()
    quad4 = ax4.pcolormesh(zLg_xz_slice, xLg_xz_slice, interp_ds['na_xz'].values[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(interp_ds['na_xz'].values)), vmax=np.max(interp_ds['na_xz'].values)), cmap='inferno')
    ax4.set_xlabel('z (Impurity Propagation Direction)')
    ax4.set_ylabel('x')
    ax4.set_title('Individual Atom Position Distribution (Interp)')
    fig4.colorbar(quad4, ax=ax4, extend='both')

    plt.show()
