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
    # (dx, dy, dz) = (0.25, 0.25, 0.25)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    # NGridPoints_cart = 1.37e5

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'real', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon', 'ReducedInterp': 'false', 'kGrid_ext': 'false'}

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

    # # # # Analysis of Total Dataset

    aIBi = -10
    qds = xr.open_dataset(innerdatapath + '/quench_Dataset.nc')
    qds_aIBi = qds.sel(aIBi=aIBi)

    PVals = qds['P'].values
    tVals = qds['t'].values
    n0 = qds.attrs['n0']
    gBB = qds.attrs['gBB']
    nu = pfc.nu(gBB)
    mI = qds.attrs['mI']
    mB = qds.attrs['mB']

    # # FULL RECONSTRUCTION OF 3D CARTESIAN BETA_K FROM 2D SPHERICAL BETA_K (doing actual interpolation in 2D spherical instead of 3D nonlinear cartesian)

    CSAmp_ds = (qds_aIBi['Real_CSAmp'] + 1j * qds_aIBi['Imag_CSAmp']).isel(t=-1)
    kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    kVec = kgrid.getArray('k')
    thVec = kgrid.getArray('th')

    print(PVals)
    Pind = 1
    P = PVals[Pind]
    print('P: {0}'.format(P))
    print('dk: {0}'.format(kVec[1] - kVec[0]))

    CSAmp_Vals = CSAmp_ds.sel(P=P).values
    Nph = qds_aIBi.isel(t=-1).sel(P=P)['Nph'].values
    Bk_2D_vals = CSAmp_Vals.reshape((len(kVec), len(thVec)))

    kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    kxg_Sph = kg * np.sin(thg)
    kzg_Sph = kg * np.cos(thg)
    # Normalization of the original data array - this checks out
    dk = kg[1, 0] - kg[0, 0]
    dth = thg[0, 1] - thg[0, 0]
    PhDen_Sph = ((1 / Nph) * np.abs(Bk_2D_vals)**2).real.astype(float)
    Bk_norm = np.sum(dk * dth * (2 * np.pi)**(-2) * kg**2 * np.sin(thg) * PhDen_Sph)
    print('Original (1/Nph)|Bk|^2 normalization (Spherical 2D): {0}'.format(Bk_norm))

    # Set reduced bounds of k-space and other things dependent on subsonic or supersonic
    if P < 0.9:
        [vmin, vmax] = [0, 500]
        # linDimMajor = 1.5
        # linDimMinor = 1.5
        linDimMajor = 2
        linDimMinor = 2
        ext_major_rat = 0.35
        ext_minor_rat = 0.35
        poslinDim = 10
        Npoints = 400  # actual number of points will be ~Npoints-1
    else:
        # [vmin, vmax] = [0, 9.2e13]
        # linDimMajor = 0.1  # For the worse grid value data, there is some dependancy on the final FFT on what range of k we pick...(specifically the z-axis = lindimMajor changing in range 0.1 - 0.4), For better data grid, FFT still vanishes after lindimMajor >=0.4
        # linDimMinor = 0.01
        linDimMajor = 2
        linDimMinor = 2
        ext_major_rat = 0.025
        ext_minor_rat = 0.0025
        poslinDim = 10
        Npoints = 400

    # Remove k values outside reduced k-space bounds (as |Bk|~0 there) and save the average of these values to add back in later before FFT
    kred_ind = np.argwhere(kg[:, 0] > (1.5 * linDimMajor))[0][0]
    kg_red = np.delete(kg, np.arange(kred_ind, kVec.size), 0)
    thg_red = np.delete(thg, np.arange(kred_ind, kVec.size), 0)
    Bk_red = np.delete(Bk_2D_vals, np.arange(kred_ind, kVec.size), 0)
    Bk_remainder = np.delete(Bk_2D_vals, np.arange(0, kred_ind), 0)
    Bk_rem_ave = np.average(Bk_remainder)
    kVec_red = kVec[0:kred_ind]
    kmax_rem = np.max(kVec)

    if toggleDict['ReducedInterp'] == 'true':
        kVec = kVec_red
        kg = kg_red
        thg = thg_red
        Bk_2D_vals = Bk_red

    # CHECK WHY ALL BK AMPLITUDES HAVE ZERO IMAGINARY PART, EVEN FOR SUPERSONIC CASE? IS THIS BECAUSE ITS THE GROUNDSTATE?
    # print(np.imag(Bk_2D_vals))

    # Create linear 3D cartesian grid and reinterpolate Bk_3D onto this grid
    kxL_pos, dkxL = np.linspace(1e-10, linDimMinor, Npoints // 2, retstep=True, endpoint=False); kxL = np.concatenate((1e-10 - 1 * np.flip(kxL_pos[1:], axis=0), kxL_pos))
    kyL_pos, dkyL = np.linspace(1e-10, linDimMinor, Npoints // 2, retstep=True, endpoint=False); kyL = np.concatenate((1e-10 - 1 * np.flip(kyL_pos[1:], axis=0), kyL_pos))
    kzL_pos, dkzL = np.linspace(1e-10, linDimMajor, Npoints // 2, retstep=True, endpoint=False); kzL = np.concatenate((1e-10 - 1 * np.flip(kzL_pos[1:], axis=0), kzL_pos))
    kxLg_3D, kyLg_3D, kzLg_3D = np.meshgrid(kxL, kyL, kzL, indexing='ij')

    # Re-interpret grid points of linear 3D Cartesian as nonlinear 3D spherical grid, find unique (k,th) points
    kg_3Di = np.sqrt(kxLg_3D**2 + kyLg_3D**2 + kzLg_3D**2)
    thg_3Di = np.arccos(kzLg_3D / kg_3Di)
    phig_3Di = np.arctan2(kyLg_3D, kxLg_3D)

    kg_3Di_flat = kg_3Di.reshape(kg_3Di.size)
    thg_3Di_flat = thg_3Di.reshape(thg_3Di.size)
    tups_3Di = np.column_stack((kg_3Di_flat, thg_3Di_flat))
    tups_3Di_unique, tups_inverse = np.unique(tups_3Di, return_inverse=True, axis=0)

    # Perform interpolation on 2D projection and reconstruct full matrix on 3D linear cartesian grid
    print('3D Cartesian grid Ntot: {:1.2E}'.format(kzLg_3D.size))
    print('Unique interp points: {:1.2E}'.format(tups_3Di_unique[:, 0].size))
    interpstart = timer()
    Bk_2D_CartInt = interpolate.griddata((kg.flatten(), thg.flatten()), Bk_2D_vals.flatten(), tups_3Di_unique, method='cubic')
    # Bk_2D_Rbf = interpolate.Rbf(kg, thg, Bk_2D.values)
    # Bk_2D_CartInt = Bk_2D_Rbf(tups_3Di_unique)
    interpend = timer()
    print('Interp Time: {0}'.format(interpend - interpstart))
    BkLg_3D_flat = Bk_2D_CartInt[tups_inverse]
    BkLg_3D = BkLg_3D_flat.reshape(kg_3Di.shape)

    BkLg_3D[np.isnan(BkLg_3D)] = 0
    PhDenLg_3D = ((1 / Nph) * np.abs(BkLg_3D)**2).real.astype(float)
    BkLg_3D_norm = np.sum(dkxL * dkyL * dkzL * (2 * np.pi)**(-3) * PhDenLg_3D)
    print('Interpolated (1/Nph)|Bk|^2 normalization (Linear Cartesian 3D): {0}'.format(BkLg_3D_norm))

    # Add the remainder of Bk back in (values close to zero for large k) (Note: can also do this more easily by setting a fillvalue in griddata and interpolating)
    if toggleDict['ReducedInterp'] == 'true' and toggleDict['kGrid_ext'] == 'true':
        kL_max_major = ext_major_rat * kmax_rem / np.sqrt(2)
        kL_max_minor = ext_minor_rat * kmax_rem / np.sqrt(2)
        print('kL_red_max_major: {0}, kL_ext_max_major: {1}, dkL_major: {2}'.format(np.max(kzL), kL_max_major, dkzL))
        print('kL_red_max_minor: {0}, kL_ext_max_minor: {1}, dkL_minor: {2}'.format(np.max(kxL), kL_max_minor, dkxL))
        kx_addon = np.arange(linDimMinor, kL_max_minor, dkxL); ky_addon = np.arange(linDimMinor, kL_max_minor, dkyL); kz_addon = np.arange(linDimMajor, kL_max_major, dkzL)
        print('kL_ext_addon size -  major: {0}, minor: {1}'.format(2 * kz_addon.size, 2 * kx_addon.size))
        kxL_ext = np.concatenate((1e-10 - 1 * np.flip(kx_addon, axis=0), np.concatenate((kxL, kx_addon))))
        kyL_ext = np.concatenate((1e-10 - 1 * np.flip(ky_addon, axis=0), np.concatenate((kyL, kx_addon))))
        kzL_ext = np.concatenate((1e-10 - 1 * np.flip(kz_addon, axis=0), np.concatenate((kzL, kx_addon))))

        ax = kxL.size; ay = kyL.size; az = kzL.size
        mx = kx_addon.size; my = ky_addon.size; mz = kz_addon.size

        BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones((mz, ax, ay)), np.concatenate((BkLg_3D, Bk_rem_ave * np.ones((mx, ay, az))), axis=0)), axis=0)
        BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones(((az + 2 * mz), mx, ay)), np.concatenate((BkLg_3D_ext, Bk_rem_ave * np.ones(((ax + 2 * mx), my, az))), axis=1)), axis=1)
        BkLg_3D_ext = np.concatenate((Bk_rem_ave * np.ones(((az + 2 * mz), (ax + 2 * mx), my)), np.concatenate((BkLg_3D_ext, Bk_rem_ave * np.ones(((ax + 2 * mx), (ay + 2 * my), mz))), axis=2)), axis=2)

        kxL = kxL_ext; kyL = kyL_ext; kzL = kzL_ext
        BkLg_3D = BkLg_3D_ext
        print('Cartesian Interp Extended Grid Shape: {0}'.format(BkLg_3D.shape))

    # Fourier Transform to get 3D position distribution
    xL = np.fft.fftshift(np.fft.fftfreq(kxL.size) * 2 * np.pi / dkxL)
    yL = np.fft.fftshift(np.fft.fftfreq(kyL.size) * 2 * np.pi / dkyL)
    zL = np.fft.fftshift(np.fft.fftfreq(kzL.size) * 2 * np.pi / dkzL)
    dxL = xL[1] - xL[0]; dyL = yL[1] - yL[0]; dzL = zL[1] - zL[0]
    dVxyz = dxL * dyL * dzL
    # print(dzL, 2 * np.pi / (kzL.size * dkzL))

    xLg_3D, yLg_3D, zLg_3D = np.meshgrid(xL, yL, zL, indexing='ij')
    beta_kxkykz = np.fft.ifftshift(BkLg_3D)
    amp_beta_xyz_preshift = np.fft.ifftn(beta_kxkykz) / dVxyz
    amp_beta_xyz = np.fft.fftshift(amp_beta_xyz_preshift)
    nxyz = ((1 / Nph) * np.abs(amp_beta_xyz)**2).real.astype(float)
    nxyz_norm = np.sum(dVxyz * nxyz)
    print('Linear grid (1/Nph)*n(x,y,z) normalization (Cartesian 3D): {0}'.format(nxyz_norm))

    # Calculate real space distribution of atoms in the BEC
    uk2 = 0.5 * (1 + (pfc.epsilon(kxLg_3D, kyLg_3D, kzLg_3D, mB) + gBB * n0) / pfc.omegak(kxLg_3D, kyLg_3D, kzLg_3D, mB, n0, gBB))
    vk2 = uk2 - 1
    uk = np.sqrt(uk2); vk = np.sqrt(vk2)

    uB_kxkykz = np.fft.ifftshift(uk * BkLg_3D)
    uB_xyz = np.fft.fftshift(np.fft.ifftn(uB_kxkykz) / dVxyz)
    vB_kxkykz = np.fft.ifftshift(vk * BkLg_3D)
    vB_xyz = np.fft.fftshift(np.fft.ifftn(vB_kxkykz) / dVxyz)
    # na_xyz = np.sum(vk2 * dkxL * dkyL * dkzL) + np.abs(uB_xyz - np.conjugate(vB_xyz))**2
    na_xyz = np.abs(uB_xyz - np.conjugate(vB_xyz))**2
    na_xyz_norm = na_xyz / np.sum(na_xyz * dVxyz)
    print(np.sum(vk2 * dkxL * dkyL * dkzL), np.max(np.abs(uB_xyz - np.conjugate(vB_xyz))**2))

    # # Create DataSet for 3D Betak and position distribution slices
    # PhDen_da = xr.DataArray(PhDenLg_3D, coords=[kxL, kyL, kzL], dims=['kx', 'ky', 'kz'])
    # nxyz_da = xr.DataArray(nxyz, coords=[xL, yL, zL], dims=['x', 'y', 'z'])

    # data_dict = {'PhDen': PhDen_da, 'nxyz': nxyz_da}
    # coords_dict = {'kx': kxL, 'ky': kyL, 'kz': kzL, 'x': xL, 'y': yL, 'z': zL}
    # attrs_dict = {'P': P, 'aIBi': aIBi}
    # interp_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    # interp_ds.to_netcdf(interpdatapath + '/InterpDat_P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    # Consistency check: use 2D ky=0 slice of |Bk|^2 to calculate phonon density and compare it to phonon density from original spherical interpolated data
    kxL_0ind = kxL.size // 2; kyL_0ind = kyL.size // 2; kzL_0ind = kzL.size // 2  # find position of zero of each axis: kxL=0, kyL=0, kzL=0
    kxLg_ky0slice = kxLg_3D[:, kyL_0ind, :]
    kzLg_ky0slice = kzLg_3D[:, kyL_0ind, :]
    PhDenLg_ky0slice = PhDenLg_3D[:, kyL_0ind, :]

    # Take 2D slices of position distribution
    zLg_y0slice = zLg_3D[:, yL.size // 2, :]
    xLg_y0slice = xLg_3D[:, yL.size // 2, :]
    nxyz_y0slice = nxyz[:, yL.size // 2, :]

    # Interpolate 2D slice of position distribution
    posmult = 5
    zL_y0slice_interp = np.linspace(-1 * poslinDim, poslinDim, posmult * zL.size); xL_y0slice_interp = np.linspace(-1 * poslinDim, poslinDim, posmult * xL.size)
    xLg_y0slice_interp, zLg_y0slice_interp = np.meshgrid(xL_y0slice_interp, zL_y0slice_interp, indexing='ij')
    nxyz_y0slice_interp = interpolate.griddata((xLg_y0slice.flatten(), zLg_y0slice.flatten()), nxyz_y0slice.flatten(), (xLg_y0slice_interp, zLg_y0slice_interp), method='cubic')

    # Take 2D slices of atom position distribution and interpolate
    na_xyz_y0slice = na_xyz_norm[:, yL.size // 2, :]
    na_xyz_y0slice_interp = interpolate.griddata((xLg_y0slice.flatten(), zLg_y0slice.flatten()), na_xyz_y0slice.flatten(), (xLg_y0slice_interp, zLg_y0slice_interp), method='cubic')

    # All Plotting: (a) 2D ky=0 slice of |Bk|^2, (b) 2D slice of position distribution
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_xlim([-1 * linDimMajor, linDimMajor])
    axes[0].set_ylim([-1 * linDimMinor, linDimMinor])
    axes[1].set_xlim([-1 * linDimMajor, linDimMajor])
    axes[1].set_ylim([-1 * linDimMinor, linDimMinor])

    # if P > 0.9:
    #     vmax = np.max(PhDen_Sph)
    # vmax = np.max(PhDen_Sph)
    # vmin = 1e-16

    quad1 = axes[0].pcolormesh(kzg_Sph, kxg_Sph, PhDen_Sph[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(PhDen_Sph)), vmax=np.max(PhDen_Sph)), cmap='plasma')
    quad1m = axes[0].pcolormesh(kzg_Sph, -1 * kxg_Sph, PhDen_Sph[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(PhDen_Sph)), vmax=np.max(PhDen_Sph)), cmap='plasma')
    fig.colorbar(quad1, ax=axes[0], extend='both')
    quad2 = axes[1].pcolormesh(kzLg_ky0slice, kxLg_ky0slice, PhDenLg_ky0slice[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(PhDen_Sph)), vmax=np.max(PhDen_Sph)), cmap='plasma')
    fig.colorbar(quad2, ax=axes[1], extend='both')
    axes[0].set_xlabel('kz (Impurity Propagation Direction)')
    axes[0].set_xlabel('kx')
    axes[1].set_xlabel('kz (Impurity Propagation Direction)')
    axes[1].set_xlabel('kx')
    axes[0].set_title('Individual Phonon Momentum Distribution (Data)')
    axes[1].set_title('Individual Phonon Momentum Distribution (Interp)')

    fig2, ax2 = plt.subplots()
    quad3 = ax2.pcolormesh(zLg_y0slice_interp, xLg_y0slice_interp, nxyz_y0slice_interp[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(nxyz_y0slice_interp)), vmax=np.max(nxyz_y0slice_interp)), cmap='plasma')
    ax2.set_xlabel('z (Impurity Propagation Direction)')
    ax2.set_ylabel('x')
    ax2.set_title('Individual Phonon Position Distribution (Interp)')
    fig2.colorbar(quad3, ax=ax2, extend='both')

    fig3, ax3 = plt.subplots()
    quad4 = ax3.pcolormesh(zLg_y0slice_interp, xLg_y0slice_interp, na_xyz_y0slice_interp[:-1, :-1], norm=colors.LogNorm(vmin=np.abs(np.min(na_xyz_y0slice_interp)), vmax=np.max(na_xyz_y0slice_interp)), cmap='plasma')
    ax3.set_xlabel('z (Impurity Propagation Direction)')
    ax3.set_ylabel('x')
    ax3.set_title('Individual Atom Position Distribution (Interp)')
    fig3.colorbar(quad4, ax=ax3, extend='both')

    plt.show()

    # # # # Analysis of Total Dataset

    # qds = xr.open_dataset(innerdatapath + '/quench_Dataset.nc')
    # # # qds = xr.open_dataset(innerdatapath + '/P_0.900_aIBi_-6.23.nc')
    # # tVals = qds['t'].values
    # # dt = tVals[1] - tVals[0]
    # # PVals = qds['P'].values
    # # aIBiVals = qds.coords['aIBi'].values
    # # n0 = qds.attrs['n0']
    # # gBB = qds.attrs['gBB']
    # # nu = pfc.nu(gBB)
    # # mI = qds.attrs['mI']
    # # mB = qds.attrs['mB']

    # # # aIBi = -10
    # # # qds_aIBi = qds.sel(aIBi=aIBi)

    # # # fig, ax = plt.subplots()
    # # # for P in PVals:
    # # #     Nph = qds_aIBi.sel(P=P)['Nph'].values
    # # #     dNph = np.diff(Nph)
    # # #     ax.plot(dNph / dt)

    # # plt.show()

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

    # # # INDIVDUAL PHONON MOMENTUM DISTRIBUTION DATASET ANALYSIS

    # nk_ds = xr.open_dataset(innerdatapath + '/nk_ind_Dataset.nc')
    # # nk_ds = xr.open_dataset(innerdatapath + '/nk_ind_Dataset_withjac.nc')
    # tVals = nk_ds['t'].values
    # dt = tVals[1] - tVals[0]
    # PVals = nk_ds['P'].values
    # aIBiVals = nk_ds.coords['aIBi'].values
    # n0 = nk_ds.attrs['n0']
    # gBB = nk_ds.attrs['gBB']
    # nu = pfc.nu(gBB)
    # mI = nk_ds.attrs['mI']
    # mB = nk_ds.attrs['mB']
    # aBB = gBB * mB / (4 * np.pi)
    # xi = (8 * np.pi * n0 * aBB)**(-1 / 2)
    # tscale = xi / nu

    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', nk_ds.coords['k'].values); kgrid.initArray_premade('th', nk_ds.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')

    # # kMat, thMat = np.meshgrid(kVec, thVec, indexing='ij')
    # # xMat = kMat * np.sin(thMat)
    # # zMat = kMat * np.cos(thMat)

    # # # SINGLE PLOT

    # # aIBi = -10
    # # Pind = -3
    # # tind = 100

    # # nk = nk_ds.sel(aIBi=aIBi).isel(P=Pind, t=tind)['nk_ind']
    # # Delta_nk = nk_ds.sel(aIBi=aIBi).isel(P=Pind, t=tind)['Delta_nk_ind']
    # # Delta_nk_2 = (nk_ds.sel(aIBi=aIBi).isel(P=Pind, t=tind)['nk_ind'] - nk_ds.sel(aIBi=aIBi).isel(P=Pind, t=tind - 1)['nk_ind']) / dt

    # # nk_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(nk, 'k', 'th', 5)
    # # xg_interp = kg_interp * np.sin(thg_interp)
    # # zg_interp = kg_interp * np.cos(thg_interp)

    # # fig1, ax1 = plt.subplots()
    # # quad1 = ax1.pcolormesh(zg_interp, xg_interp, nk_interp_vals)
    # # ax1.pcolormesh(zg_interp, -1 * xg_interp, nk_interp_vals)
    # # ax1.set_title('aIBi={0}, P={1}, t={2}'.format(aIBi, PVals[Pind], tVals[tind]))
    # # ax1.set_xlim([-3, 3])
    # # ax1.set_ylim([-3, 3])
    # # fig1.colorbar(quad1, ax=ax1, extend='both')

    # # # fig2, ax2 = plt.subplots()
    # # # quad2 = ax2.pcolormesh(zMat, xMat, Delta_nk_2.values)
    # # # ax2.pcolormesh(zMat, -1 * xMat, Delta_nk_2.values)
    # # # ax2.set_title('aIBi={0}, P={1}, t={2}'.format(aIBi, PVals[Pind], tVals[tind]))
    # # # # ax2.set_xlim([-3, 3])
    # # # # ax2.set_ylim([-3, 3])
    # # # fig2.colorbar(quad2, ax=ax2, extend='both')

    # # plt.show()

    # # ANIMATIONS

    # aIBi = -10
    # # P = 0.4
    # # P = 1.4
    # P = 5

    # tmax = 40
    # tmax_ind = (np.abs(tVals - tmax)).argmin()

    # nk = nk_ds.sel(aIBi=aIBi).sel(P=P, method='nearest')['nk_ind']
    # Delta_nk = nk_ds.sel(aIBi=aIBi).sel(P=P, method='nearest')['Delta_nk_ind']

    # P = 1 * nk['P'].values

    # Pph = qds.sel(aIBi=aIBi).sel(P=P)['Pph'].values
    # Pimp = P - Pph

    # # Phonon probability

    # fig1, ax1 = plt.subplots()
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

    # vmin = 0
    # vmax = 700
    # # vmax = 4

    # nk0_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(nk.isel(t=0), 'k', 'th', 5)
    # xg_interp = kg_interp * np.sin(thg_interp)
    # zg_interp = kg_interp * np.cos(thg_interp)

    # quad1 = ax1.pcolormesh(zg_interp, xg_interp, nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # quad1m = ax1.pcolormesh(zg_interp, -1 * xg_interp, nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # curve1 = ax1.plot(Pph[0], 0, marker='x', markersize=10, color="magenta", label=r'$P_{ph}$')[0]
    # curve1m = ax1.plot(Pimp[0], 0, marker='o', markersize=10, color="red", label=r'$P_{imp}$')[0]

    # t_text = ax1.text(0.81, 0.9, r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[0] / tscale), transform=ax1.transAxes, color='r')
    # # ax1.set_xlim([-2, 2])
    # # ax1.set_ylim([-2, 2])
    # ax1.set_xlim([-3, 3])
    # ax1.set_ylim([-3, 3])
    # ax1.legend(loc=2)
    # ax1.grid(True, linewidth=0.5)
    # ax1.set_title('Ind Phonon Distribution (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$P=$' + '{:.2f})'.format(P))
    # ax1.set_xlabel(r'$k_{z}$')
    # ax1.set_ylabel(r'$k_{x}$')
    # fig1.colorbar(quad1, ax=ax1, extend='both')

    # def animate1(i):
    #     nk_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(nk.isel(t=i), 'k', 'th', 5)
    #     quad1.set_array(nk_interp_vals[:-1, :-1].ravel())
    #     quad1m.set_array(nk_interp_vals[:-1, :-1].ravel())
    #     curve1.set_xdata(Pph[i])
    #     curve1m.set_xdata(Pimp[i])
    #     t_text.set_text(r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[i] / tscale))
    # anim1 = FuncAnimation(fig1, animate1, interval=1e-5, frames=range(tmax_ind + 1), blit=False)
    # anim1.save(animpath + '/aIBi_{:d}_P_{:.2f}'.format(aIBi, P) + '_indPhononDist_2D.gif', writer='imagemagick')

    # # Change in phonon probability

    # fig2, ax2 = plt.subplots()

    # # vmin = 1
    # # vmax = 0
    # # for Pind, Pv in enumerate(PVals):
    # #     for tind, t in enumerate(tVals):
    # #         if t == tVals[-1]:
    # #             break
    # #         vec = nk_ds.sel(aIBi=aIBi, P=Pv).isel(t=tind + 1)['nk_ind'].values - nk_ds.sel(aIBi=aIBi, P=Pv).isel(t=tind)['nk_ind'].values
    # #         vec = vec / dt
    # #         # vec = nk.sel(t=t).values
    # #         if np.min(vec) < vmin:
    # #             vmin = np.min(vec)
    # #         if np.max(vec) > vmax:
    # #             vmax = np.max(vec)

    # # vmin = -0.9
    # # vmax = 0.82
    # # vmax = 0.6
    # vmin = -25
    # vmax = 110

    # dnk0 = (nk.isel(t=1) - nk.isel(t=0)) / dt
    # dnk0_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(dnk0, 'k', 'th', 5)
    # xg_interp = kg_interp * np.sin(thg_interp)
    # zg_interp = kg_interp * np.cos(thg_interp)

    # quad2 = ax2.pcolormesh(zg_interp, xg_interp, dnk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # quad2m = ax2.pcolormesh(zg_interp, -1 * xg_interp, dnk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # curve2 = ax2.plot(Pph[0], 0, marker='x', markersize=10, color="magenta", label=r'$P_{ph}$')[0]
    # curve2m = ax2.plot(Pimp[0], 0, marker='o', markersize=10, color="red", label=r'$P_{imp}$')[0]
    # t_text = ax2.text(0.81, 0.9, r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[0] / tscale), transform=ax2.transAxes, color='r')
    # # ax2.set_xlim([-2, 2])
    # # ax2.set_ylim([-2, 2])
    # ax2.set_xlim([-3, 3])
    # ax2.set_ylim([-3, 3])
    # ax2.legend(loc=2)
    # ax2.grid(True, linewidth=0.5)
    # ax2.set_title('Ind Phonon Distribution Time Derivative (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$P=$' + '{:.2f})'.format(P))
    # ax2.set_xlabel(r'$k_{z}$')
    # ax2.set_ylabel(r'$k_{x}$')
    # fig2.colorbar(quad2, ax=ax2, extend='both')

    # def animate2(i):
    #     dnk = (nk.isel(t=i + 1) - nk.isel(t=i)) / dt
    #     dnk_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(dnk, 'k', 'th', 5)
    #     quad2.set_array(dnk_interp_vals[:-1, :-1].ravel())
    #     quad2m.set_array(dnk_interp_vals[:-1, :-1].ravel())
    #     curve2.set_xdata(Pph[i])
    #     curve2m.set_xdata(Pimp[i])
    #     t_text.set_text(r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[i] / tscale))
    # anim2 = FuncAnimation(fig2, animate2, interval=1e-5, frames=range(tmax_ind + 1), blit=False)
    # anim2.save(animpath + '/aIBi_{:d}_P_{:.2f}'.format(aIBi, P) + '_indPhononDistDeriv_2D_num.gif', writer='imagemagick')

    # # fig3, ax3 = plt.subplots()
    # # # vmin = 1
    # # # vmax = 0
    # # # for Pind, Pv in enumerate(PVals):
    # # #     for tind, t in enumerate(tVals):
    # # #         vec = nk_ds.sel(aIBi=aIBi, P=Pv, t=t)['nk_ind'].values
    # # #         # vec = nk.sel(t=t).values
    # # #         if np.min(vec) < vmin:
    # # #             vmin = np.min(vec)
    # # #         if np.max(vec) > vmax:
    # # #             vmax = np.max(vec)

    # # vmin = -0.9
    # # vmax = 0.82

    # # Delta_nk0_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(Delta_nk.isel(t=0), 'k', 'th', 5)
    # # xg_interp = kg_interp * np.sin(thg_interp)
    # # zg_interp = kg_interp * np.cos(thg_interp)

    # # quad3 = ax3.pcolormesh(zg_interp, xg_interp, Delta_nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # # quad3m = ax3.pcolormesh(zg_interp, -1 * xg_interp, Delta_nk0_interp_vals[:-1, :-1], vmin=vmin, vmax=vmax)
    # # curve3 = ax3.plot(Pph[0], 0, marker='x', markersize=10, color="magenta", label=r'$P_{ph}$')[0]
    # # curve3m = ax3.plot(Pimp[0], 0, marker='o', markersize=10, color="red", label=r'$P_{imp}$')[0]
    # # t_text = ax3.text(0.81, 0.9, r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[0] / tscale), transform=ax3.transAxes, color='r')
    # # ax3.set_xlim([-2, 2])
    # # ax3.set_ylim([-2, 2])
    # # ax3.legend(loc=2)
    # # ax3.grid(True, linewidth=0.5)
    # # ax3.set_title('Ind Phonon Distribution Time Derivative (' + r'$aIB^{-1}=$' + '{0}, '.format(aIBi) + r'$P=$' + '{:.2f})'.format(P))
    # # ax3.set_xlabel(r'$k_{z}$')
    # # ax3.set_ylabel(r'$k_{x}$')
    # # fig3.colorbar(quad3, ax=ax3, extend='both')

    # # def animate3(i):
    # #     Delta_nk_interp_vals, kg_interp, thg_interp = pfc.xinterp2D(Delta_nk.isel(t=i), 'k', 'th', 5)
    # #     quad3.set_array(Delta_nk_interp_vals[:-1, :-1].ravel())
    # #     quad3m.set_array(Delta_nk_interp_vals[:-1, :-1].ravel())
    # #     curve3.set_xdata(Pph[i])
    # #     curve3m.set_xdata(Pimp[i])
    # #     t_text.set_text(r'$t$ [$\frac{\xi}{c}$]: ' + '{:.1f}'.format(tVals[i] / tscale))
    # # anim3 = FuncAnimation(fig3, animate3, interval=1e-5, frames=range(tmax_ind + 1), blit=False)
    # # anim3.save(animpath + '/aIBi_{:d}_P_{:.2f}'.format(aIBi, P) + '_indPhononDistDeriv_2D_an.gif', writer='imagemagick')

    # # plt.draw()
    # # plt.show()
