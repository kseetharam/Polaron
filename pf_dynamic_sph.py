import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import quad
from timeit import default_timer as timer
import os
from scipy import interpolate
import Grid


# ---- HELPER FUNCTIONS ----


def kcos_func(kgrid):
    #
    names = list(kgrid.arrays.keys())
    functions_kcos = [lambda k: k, np.cos]
    return kgrid.function_prod(names, functions_kcos)


def kpow2_func(kgrid):
    #
    names = list(kgrid.arrays.keys())
    functions_kpow2 = [lambda k: k**2, lambda th: 0 * th + 1]
    return kgrid.function_prod(names, functions_kpow2)


# ---- BASIC FUNCTIONS ----


def ur(mI, mB):
    return (mB * mI) / (mB + mI)


def nu(gBB):
    return np.sqrt(gBB)  # NOTE: this assumes n0=1 and mB = 1 (real formula is np.sqrt(n0*gBB/mB))


def epsilon(k, mB):
    return k**2 / (2 * mB)


def omegak(k, mB, n0, gBB):
    ep = epsilon(k, mB)
    return np.sqrt(ep * (ep + 2 * gBB * n0))


def Omega(kgrid, DP, mI, mB, n0, gBB):
    names = list(kgrid.arrays.keys())  # ***need to have arrays added as k, th when kgrid is created
    if names[0] != 'k':
        print('CREATED kgrid IN WRONG ORDER')
    functions_omega0 = [lambda k: omegak(k, mB, n0, gBB) + (k**2 / (2 * mI)), lambda th: 0 * th + 1]
    omega0 = kgrid.function_prod(names, functions_omega0)
    return omega0 - kcos_func(kgrid) * DP / mI


def Wk(kgrid, mB, n0, gBB):
    names = list(kgrid.arrays.keys())
    functions_Wk = [lambda k: np.sqrt(epsilon(k, mB) / omegak(k, mB, n0, gBB)), lambda th: 0 * th + 1]
    return kgrid.function_prod(names, functions_Wk)


def gIB(kgrid, aIBi, mI, mB, n0, gBB):
    # gives bare interaction strength constant gIB
    k_max = kgrid.getArray('k')[-1]
    mR = ur(mI, mB)
    return 1 / ((mR / (2 * np.pi)) * aIBi - (mR / np.pi**2) * k_max)


# ---- SPECTRUM RELATED FUNCTIONS ----


# def PCrit_inf(kcutoff, aIBi, mI, mB, n0, gBB):
#     #
#     DP = mI * nu(mB, n0, gBB)  # condition for critical momentum is P-PB = mI*nu where nu is the speed of sound
#     # non-grid helper function

#     def Wk(k, gBB, mB, n0):
#         return np.sqrt(eB(k, mB) / w(k, gBB, mB, n0))

#     # calculate aSi
#     def integrand(k): return (4 * ur(mI, mB) / (k**2) - ((Wk(k, gBB, mB, n0)**2) / (DP * k / mI)) * np.log((w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) / (w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)))) * (k**2)
#     val, abserr = quad(integrand, 0, kcutoff, epsabs=0, epsrel=1.49e-12)
#     aSi = (1 / (2 * np.pi * ur(mI, mB))) * val
#     # calculate PB (phonon momentum)

#     def integrand(k): return ((2 * (w(k, gBB, mB, n0) + (k**2) / (2 * mI)) * (DP * k / mI) + (w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) * (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) * np.log((w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) / (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)))) / ((w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) * (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) * (DP * k / mI)**2)) * (Wk(k, gBB, mB, n0)**2) * (k**3)
#     val, abserr = quad(integrand, 0, kcutoff, epsabs=0, epsrel=1.49e-12)
#     PB = n0 / (ur(mI, mB)**2 * (aIBi - aSi)**2) * val

#     return DP + PB

def dirRF(dataset, kgrid, cParams, sParams):
    CSAmp = dataset['Real_CSAmp'] + 1j * dataset['Imag_CSAmp']
    Phase = dataset['Phase']
    dVk = kgrid.dV()
    tgrid = CSAmp.coords['t'].values
    CSA0 = CSAmp.isel(t=0).values; CSA0 = CSA0.reshape(CSA0.size)
    Phase0 = Phase.isel(t=0).values
    DynOv_Vec = np.zeros(tgrid.size, dtype=complex)

    for tind, t in enumerate(tgrid):
        CSAt = CSAmp.sel(t=t).values; CSAt = CSAt.reshape(CSAt.size)
        Phaset = Phase.sel(t=t).values
        exparg = np.dot(np.abs(CSAt)**2 + np.abs(CSA0)**2 - 2 * CSA0.conjugate() * CSAt, dVk)
        DynOv_Vec[tind] = np.exp(-1j * (Phaset - Phase0)) * np.exp((-1 / 2) * exparg)

    # calculate polaron energy (energy of initial state CSA0)
    [P, aIBi] = cParams
    [mI, mB, n0, gBB] = sParams
    dVk = kgrid.dV()
    kzg_flat = kcos_func(kgrid)
    g_IB = gIB(kgrid, aIBi, mI, mB, n0, gBB)
    PB0 = np.dot(kzg_flat * np.abs(CSA0)**2, dVk).real.astype(float)
    DP0 = P - PB0
    Energy0 = (P**2 - PB0**2) / (2 * mI) + np.dot(Omega(kgrid, DP0, mI, mB, n0, gBB) * np.abs(CSA0)**2, dVk) + g_IB * (np.dot(Wk(kgrid, mB, n0, gBB) * CSA0, dVk) + np.sqrt(n0))**2

    # calculate full dynamical overlap
    DynOv_Vec = np.exp(1j * Energy0) * DynOv_Vec
    ReDynOv_da = xr.DataArray(np.real(DynOv_Vec), coords=[tgrid], dims=['t'])
    ImDynOv_da = xr.DataArray(np.imag(DynOv_Vec), coords=[tgrid], dims=['t'])
    # DynOv_ds = xr.Dataset({'Real_DynOv': ReDynOv_da, 'Imag_DynOv': ImDynOv_da}, coords={'t': tgrid}, attrs=dataset.attrs)
    DynOv_ds = dataset[['Real_CSAmp', 'Imag_CSAmp', 'Phase']]; DynOv_ds['Real_DynOv'] = ReDynOv_da; DynOv_ds['Imag_DynOv'] = ImDynOv_da; DynOv_ds.attrs = dataset.attrs
    return DynOv_ds


# def spectFunc(t_Vec, S_Vec, tdecay):
#     # spectral function (Fourier Transform of dynamical overlap) using convention A(omega) = 2*Re[\int {S(t)*e^(-i*omega*t)}]
#     dt = t_Vec[1] - t_Vec[0]
#     Nt = t_Vec.size
#     decayFactor = np.exp(-1 * t_Vec / tdecay)
#     Sarg = S_Vec * decayFactor
#     sf_preshift = 2 * np.real(dt * np.fft.fft(Sarg))
#     sf = np.fft.fftshift(sf_preshift)
#     omega = np.fft.fftshift((2 * np.pi / dt) * np.fft.fftfreq(Nt))
#     return omega, sf


def spectFunc(t_Vec, S_Vec, tdecay):
    # spectral function (Fourier Transform of dynamical overlap) using convention A(omega) = 2*Re[\int {S(t)*e^(i*omega*t)}]
    dt = t_Vec[1] - t_Vec[0]
    Nt = t_Vec.size
    domega = 2 * np.pi / (Nt * dt)
    decayFactor = np.exp(-1 * t_Vec / tdecay)
    Sarg = S_Vec * decayFactor
    sf_preshift = np.real((2 * np.pi / domega) * np.fft.ifft(Sarg))
    # sf_preshift = 2 * np.real((2 * np.pi / domega) * np.fft.ifft(Sarg))
    sf = np.fft.fftshift(sf_preshift)
    omega = np.fft.fftshift((2 * np.pi / dt) * np.fft.fftfreq(Nt))
    return omega, sf


def Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB):
    dVk = kgrid.dV()
    kzg_flat = kcos_func(kgrid)
    Wk_grid = Wk(kgrid, mB, n0, gBB)
    Wki_grid = 1 / Wk_grid

    amplitude = CSAmp.reshape(CSAmp.size)
    PB = np.dot(kzg_flat * np.abs(amplitude)**2, dVk).real.astype(float)
    DP = P - PB
    Omega_grid = Omega(kgrid, DP, mI, mB, n0, gBB)
    gnum = gIB(kgrid, aIBi, mI, mB, n0, gBB)

    xp = 0.5 * np.dot(Wk_grid, amplitude * dVk)
    xm = 0.5 * np.dot(Wki_grid, amplitude * dVk)
    En = ((P**2 - PB**2) / (2 * mI) +
          np.dot(dVk * Omega_grid, np.abs(amplitude)**2) +
          gnum * (2 * np.real(xp) + np.sqrt(n0))**2 -
          gnum * (2 * np.imag(xm))**2)

    return En.real.astype(float)

# ---- OTHER HELPER FUNCTIONS AND DYNAMICS ----


# def PCrit_inf(kcutoff, aIBi, mI, mB, n0, gBB):
#     #
#     DP = mI * nu(gBB)  # condition for critical momentum is P-PB = mI*nu where nu is the speed of sound
#     # non-grid helper function

#     def Wk(k, gBB, mB, n0):
#         return np.sqrt(eB(k, mB) / w(k, gBB, mB, n0))

#     # calculate aSi
#     def integrand(k): return (4 * ur(mI, mB) / (k**2) - ((Wk(k, gBB, mB, n0)**2) / (DP * k / mI)) * np.log((w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) / (w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)))) * (k**2)
#     val, abserr = quad(integrand, 0, kcutoff, epsabs=0, epsrel=1.49e-12)
#     aSi = (1 / (2 * np.pi * ur(mI, mB))) * val
#     # calculate PB (phonon momentum)

#     def integrand(k): return ((2 * (w(k, gBB, mB, n0) + (k**2) / (2 * mI)) * (DP * k / mI) + (w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) * (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) * np.log((w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) / (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)))) / ((w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) * (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) * (DP * k / mI)**2)) * (Wk(k, gBB, mB, n0)**2) * (k**3)
#     val, abserr = quad(integrand, 0, kcutoff, epsabs=0, epsrel=1.49e-12)
#     PB = n0 / (ur(mI, mB)**2 * (aIBi - aSi)**2) * val

#     return DP + PB


# def spectFunc(t_Vec, S_Vec):
#     # spectral function (Fourier Transform of dynamical overlap)
#     tstep = t_Vec[1] - t_Vec[0]
#     N = t_Vec.size
#     tdecay = 3
#     decayFactor = np.exp(-1 * t_Vec / tdecay)
#     # decayFactor = 1
#     sf = 2 * np.real(np.fft.ifft(S_Vec * decayFactor))
#     omega = 2 * np.pi * np.fft.fftfreq(N, d=tstep)
#     return omega, sf


# def Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB):
#     dVk = kgrid.dV()
#     kzg_flat = kcos_func(kgrid)
#     Wk_grid = Wk(kgrid, mB, n0, gBB)
#     Wki_grid = 1 / Wk_grid

#     amplitude = CSAmp.reshape(CSAmp.size)
#     PB = np.dot(kzg_flat * np.abs(amplitude)**2, dVk).real.astype(float)
#     DP = P - PB
#     Omega_grid = Omega(kgrid, DP, mI, mB, n0, gBB)
#     gnum = gIB(kgrid, aIBi, mI, mB, n0, gBB)

#     xp = 0.5 * np.dot(Wk_grid, amplitude * dVk)
#     xm = 0.5 * np.dot(Wki_grid, amplitude * dVk)
#     En = ((P**2 - PB**2) / (2 * mI) +
#           np.dot(dVk * Omega_grid, np.abs(amplitude)**2) +
#           gnum * (2 * np.real(xp) + np.sqrt(n0))**2 -
#           gnum * (2 * np.imag(xm))**2)

#     return En.real.astype(float)


def CSAmp_timederiv(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB):
    # takes coherent state amplitude CSAmp in flattened form and returns d(CSAmp)/dt in flattened form
    dVk = kgrid.dV()
    kzg_flat = kcos_func(kgrid)
    Wk_grid = Wk(kgrid, mB, n0, gBB)
    Wki_grid = 1 / Wk_grid
    Omega0_grid = Omega(kgrid, 0, mI, mB, n0, gBB)
    gnum = gIB(kgrid, aIBi, mI, mB, n0, gBB)

    PB = np.dot(kzg_flat * np.abs(CSAmp)**2, dVk)
    betaSum = CSAmp + np.conjugate(CSAmp)
    xp = 0.5 * np.dot(Wk_grid, betaSum * dVk)
    betaDiff = CSAmp - np.conjugate(CSAmp)
    xm = 0.5 * np.dot(Wki_grid, betaDiff * dVk)

    damp = -1j * (gnum * np.sqrt(n0) * Wk_grid +
                  CSAmp * (Omega0_grid - kzg_flat * (P - PB) / mI) +
                  gnum * (Wk_grid * xp + Wki_grid * xm))

    # DeltaAmp = damp.reshape(len(kgrid.getArray('k')), len(kgrid.getArray('th')))
    return damp


def reconstructDistributions(CSAmp_ds, linDimMajor, linDimMinor, dkxL, dkyL, dkzL):
    import pf_dynamic_cart as pfc
    # Set up
    P = CSAmp_ds['P'].values
    aIBi = CSAmp_ds.attrs['aIBi']
    n0 = CSAmp_ds.attrs['n0']; gBB = CSAmp_ds.attrs['gBB']; mI = CSAmp_ds.attrs['mI']; mB = CSAmp_ds.attrs['mB']

    kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    kVec = kgrid.getArray('k')
    thVec = kgrid.getArray('th')
    print('P: {0}'.format(P))
    print('dk: {0}'.format(kVec[1] - kVec[0]))
    Bk_2D_vals = CSAmp_ds.values
    Nph = CSAmp_ds.attrs['Nph']
    # Bk_2D_vals = CSAmp_Vals.reshape((len(kVec), len(thVec)))
    kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # kxg_Sph = kg * np.sin(thg)
    # kzg_Sph = kg * np.cos(thg)
    # Normalization of the original data array - this checks out
    dk = kg[1, 0] - kg[0, 0]
    dth = thg[0, 1] - thg[0, 0]
    PhDen_Sph = ((1 / Nph) * np.abs(Bk_2D_vals)**2).real.astype(float)
    Bk_norm = np.sum(dk * dth * (2 * np.pi)**(-2) * kg**2 * np.sin(thg) * PhDen_Sph)
    print('Original (1/Nph)|Bk|^2 normalization (Spherical 2D): {0}'.format(Bk_norm))

    # Create linear 3D cartesian grid and reinterpolate Bk_3D onto this grid
    kmin = 1e-10
    # kmin = 0.14
    kxL_pos = np.arange(kmin, linDimMinor, dkxL); kxL = np.concatenate((kmin - 1 * np.flip(kxL_pos[1:], axis=0), kxL_pos))
    kyL_pos = np.arange(kmin, linDimMinor, dkyL); kyL = np.concatenate((kmin - 1 * np.flip(kyL_pos[1:], axis=0), kyL_pos))
    kzL_pos = np.arange(kmin, linDimMajor, dkzL); kzL = np.concatenate((kmin - 1 * np.flip(kzL_pos[1:], axis=0), kzL_pos))
    print('size - kxL: {0}, kyL: {1}, kzL: {2}'.format(kxL.size, kyL.size, kzL.size))
    kxLg_3D, kyLg_3D, kzLg_3D = np.meshgrid(kxL, kyL, kzL, indexing='ij')
    print('dkxL: {0}, dkyL: {1}, dkzL: {2}'.format(dkxL, dkyL, dkzL))
    # Re-interpret grid points of linear 3D Cartesian as nonlinear 3D spherical grid, find unique (k,th) points
    kg_3Di = np.sqrt(kxLg_3D**2 + kyLg_3D**2 + kzLg_3D**2)
    thg_3Di = np.arccos(kzLg_3D / kg_3Di)
    # phig_3Di = np.arctan2(kyLg_3D, kxLg_3D)
    kg_3Di_flat = kg_3Di.reshape(kg_3Di.size)
    thg_3Di_flat = thg_3Di.reshape(thg_3Di.size)
    tups_3Di = np.column_stack((kg_3Di_flat, thg_3Di_flat))
    tups_3Di_unique, tups_inverse = np.unique(tups_3Di, return_inverse=True, axis=0)
    # Perform interpolation on 2D projection and reconstruct full matrix on 3D linear cartesian grid
    print('3D Cartesian grid Ntot: {:1.2E}'.format(kzLg_3D.size))
    print('Unique interp points: {:1.2E}'.format(tups_3Di_unique[:, 0].size))
    interpstart = timer()
    Bk_2D_CartInt = interpolate.griddata((kg.flatten(), thg.flatten()), Bk_2D_vals.flatten(), tups_3Di_unique, method='cubic')
    # Bk_2D_CartInt = interpolate.griddata((kg.flatten(), thg.flatten()), Bk_2D_vals.flatten(), tups_3Di_unique, method='linear')
    interpend = timer()
    print('Interp Time: {0}'.format(interpend - interpstart))
    # print('Interpolated (1/Nph)|Bk|^2 normalization (Still in 2D): {0}'.format())
    BkLg_3D_flat = Bk_2D_CartInt[tups_inverse]
    BkLg_3D = BkLg_3D_flat.reshape(kg_3Di.shape)
    BkLg_3D[np.isnan(BkLg_3D)] = 0
    PhDenLg_3D = ((1 / Nph) * np.abs(BkLg_3D)**2).real.astype(float)
    BkLg_3D_norm = np.sum(dkxL * dkyL * dkzL * (2 * np.pi)**(-3) * PhDenLg_3D)
    print('Interpolated (1/Nph)|Bk|^2 normalization (Linear Cartesian 3D): {0}'.format(BkLg_3D_norm))
    # Fourier Transform to get 3D position distribution of phonons
    xL = np.fft.fftshift(np.fft.fftfreq(kxL.size) * 2 * np.pi / dkxL)
    yL = np.fft.fftshift(np.fft.fftfreq(kyL.size) * 2 * np.pi / dkyL)
    zL = np.fft.fftshift(np.fft.fftfreq(kzL.size) * 2 * np.pi / dkzL)
    dxL = xL[1] - xL[0]; dyL = yL[1] - yL[0]; dzL = zL[1] - zL[0]
    dVxyz = dxL * dyL * dzL
    xLg_3D, yLg_3D, zLg_3D = np.meshgrid(xL, yL, zL, indexing='ij')
    beta_kxkykz = np.fft.ifftshift(BkLg_3D)
    amp_beta_xyz_preshift = np.fft.ifftn(beta_kxkykz) / dVxyz
    amp_beta_xyz = np.fft.fftshift(amp_beta_xyz_preshift)
    np_xyz = ((1 / Nph) * np.abs(amp_beta_xyz)**2).real.astype(float)
    np_xyz_norm = np.sum(dVxyz * np_xyz)
    print('Linear grid (1/Nph)*n(x,y,z) normalization (Cartesian 3D): {0}'.format(np_xyz_norm))
    # Calculate total phonon momentum distribution
    beta2_kxkykz = np.abs(beta_kxkykz)**2
    beta2_xyz_preshift = np.fft.ifftn(beta2_kxkykz) / dVxyz
    beta2_xyz = np.fft.fftshift(beta2_xyz_preshift)
    decay_length = 5
    decay_xyz = np.exp(-1 * (xLg_3D**2 + yLg_3D**2 + zLg_3D**2) / (2 * decay_length**2))
    fexp = (np.exp(beta2_xyz - Nph) - np.exp(-Nph)) * decay_xyz
    nPB_preshift = np.fft.fftn(fexp) * dVxyz
    nPB_complex = np.fft.fftshift(nPB_preshift) / ((2 * np.pi)**3)  # this is the phonon momentum distribution in 3D Cartesian coordinates
    nPB = np.abs(nPB_complex)
    nPB_deltaK0 = np.exp(-Nph)
    # Produce impurity momentum and total phonon momentum magnitude distributions
    kgrid_L = Grid.Grid('CARTESIAN_3D')
    kgrid_L.initArray_premade('kx', kxL); kgrid_L.initArray_premade('ky', kyL); kgrid_L.initArray_premade('kz', kzL)
    PIgrid = pfc.ImpMomGrid_from_PhononMomGrid(kgrid_L, P)
    PB_x = kxL; PB_y = kyL; PB_z = kzL
    PI_x = PIgrid.getArray('kx'); PI_y = PIgrid.getArray('ky'); PI_z = PIgrid.getArray('kz')
    [PBm, nPBm, PIm, nPIm] = pfc.xyzDist_To_magDist(kgrid_L, nPB, P)
    # Calculate real space distribution of atoms in the BEC
    uk2 = 0.5 * (1 + (pfc.epsilon(kxLg_3D, kyLg_3D, kzLg_3D, mB) + gBB * n0) / pfc.omegak(kxLg_3D, kyLg_3D, kzLg_3D, mB, n0, gBB))
    vk2 = uk2 - 1
    uk = np.sqrt(uk2); vk = np.sqrt(vk2)
    uB_kxkykz = np.fft.ifftshift(uk * BkLg_3D)
    uB_xyz = np.fft.fftshift(np.fft.ifftn(uB_kxkykz) / dVxyz)
    vB_kxkykz = np.fft.ifftshift(vk * BkLg_3D)
    vB_xyz = np.fft.fftshift(np.fft.ifftn(vB_kxkykz) / dVxyz)
    # na_xyz = np.sum(vk2 * dkxL * dkyL * dkzL) + np.abs(uB_xyz - np.conjugate(vB_xyz))**2
    na_xyz_prenorm = (np.abs(uB_xyz - np.conjugate(vB_xyz))**2).real.astype(float)
    na_xyz = na_xyz_prenorm / np.sum(na_xyz_prenorm * dVxyz)
    print(np.sum(vk2 * dkxL * dkyL * dkzL), np.max(np.abs(uB_xyz - np.conjugate(vB_xyz))**2))
    # Create DataSet for 3D Betak and position distribution slices
    Nx = len(kxL); Ny = len(kyL); Nz = len(kzL)
    PhDen_xz_slice_da = xr.DataArray(PhDenLg_3D[:, Ny // 2, :], coords=[kxL, kzL], dims=['kx', 'kz'])
    PhDen_xy_slice_da = xr.DataArray(PhDenLg_3D[:, :, Nz // 2], coords=[kxL, kyL], dims=['kx', 'ky'])
    PhDen_yz_slice_da = xr.DataArray(PhDenLg_3D[Nx // 2, :, :], coords=[kyL, kzL], dims=['ky', 'kz'])
    np_xz_slice_da = xr.DataArray(np_xyz[:, Ny // 2, :], coords=[xL, zL], dims=['x', 'z'])
    np_xy_slice_da = xr.DataArray(np_xyz[:, :, Nz // 2], coords=[xL, yL], dims=['x', 'y'])
    np_yz_slice_da = xr.DataArray(np_xyz[Nx // 2, :, :], coords=[yL, zL], dims=['y', 'z'])
    na_xz_slice_da = xr.DataArray(na_xyz[:, Ny // 2, :], coords=[xL, zL], dims=['x', 'z'])
    na_xy_slice_da = xr.DataArray(na_xyz[:, :, Nz // 2], coords=[xL, yL], dims=['x', 'y'])
    na_yz_slice_da = xr.DataArray(na_xyz[Nx // 2, :, :], coords=[yL, zL], dims=['y', 'z'])
    nPB_xz_slice = nPB[:, Ny // 2, :]; nPB_xz_slice_da = xr.DataArray(nPB_xz_slice, coords=[PB_x, PB_z], dims=['PB_x', 'PB_z'])
    nPB_xy_slice = nPB[:, :, Nz // 2]; nPB_xy_slice_da = xr.DataArray(nPB_xy_slice, coords=[PB_x, PB_y], dims=['PB_x', 'PB_y'])
    nPB_yz_slice = nPB[Nx // 2, :, :]; nPB_yz_slice_da = xr.DataArray(nPB_yz_slice, coords=[PB_y, PB_z], dims=['PB_y', 'PB_z'])
    nPI_xz_slice = np.flip(np.flip(nPB_xz_slice, 0), 1); nPI_xz_slice_da = xr.DataArray(nPI_xz_slice, coords=[PI_x, PI_z], dims=['PI_x', 'PI_z'])
    nPI_xy_slice = np.flip(np.flip(nPB_xy_slice, 0), 1); nPI_xy_slice_da = xr.DataArray(nPI_xy_slice, coords=[PI_x, PI_y], dims=['PI_x', 'PI_y'])
    nPI_yz_slice = np.flip(np.flip(nPB_yz_slice, 0), 1); nPI_yz_slice_da = xr.DataArray(nPI_yz_slice, coords=[PI_y, PI_z], dims=['PI_y', 'PI_z'])
    nPBm_da = xr.DataArray(nPBm, coords=[PBm], dims=['PB_mag'])
    nPIm_da = xr.DataArray(nPIm, coords=[PIm], dims=['PI_mag'])
    data_dict = ({'PhDen_xz': PhDen_xz_slice_da, 'PhDen_xy': PhDen_xy_slice_da, 'PhDen_yz': PhDen_yz_slice_da,
                  'np_xz': np_xz_slice_da, 'np_xy': np_xy_slice_da, 'np_yz': np_yz_slice_da,
                  'na_xz': na_xz_slice_da, 'na_xy': na_xy_slice_da, 'na_yz': na_yz_slice_da,
                  'nPB_xz_slice': nPB_xz_slice_da, 'nPB_xy_slice': nPB_xy_slice_da, 'nPB_yz_slice': nPB_yz_slice_da, 'nPB_mag': nPBm_da,
                  'nPI_xz_slice': nPI_xz_slice_da, 'nPI_xy_slice': nPI_xy_slice_da, 'nPI_yz_slice': nPI_yz_slice_da, 'nPI_mag': nPIm_da})
    coords_dict = {'kx': kxL, 'ky': kyL, 'kz': kzL, 'x': xL, 'y': yL, 'z': zL, 'PB_x': PB_x, 'PB_y': PB_y, 'PB_z': PB_z, 'PI_x': PI_x, 'PI_y': PI_y, 'PI_z': PI_z, 'PB_mag': PBm, 'PI_mag': PIm}
    attrs_dict = {'P': P, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'mom_deltapeak': nPB_deltaK0}
    interp_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    # interp_ds.to_netcdf(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, linDimMajor, linDimMinor))
    return interp_ds

    # # compare grids
    # fig, ax = plt.subplots()
    # kxLg, kzLg = np.meshgrid(kxL, kzL, indexing='ij')
    # kxg_Sph = kg * np.sin(thg)
    # kzg_Sph = kg * np.cos(thg)
    # ax.scatter(kzg_Sph, kxg_Sph, c='b')
    # ax.scatter(kzg_Sph, -1 * kxg_Sph, c='b')
    # ax.scatter(kzLg, kxLg, c='r')
    # plt.show()
    # # remove references to objects to free up memory
    # kxLg_3D, kyLg_3D, kzLg_3D, kg_3Di, thg_3Di, phig_3Di, kg_3Di_flat, thg_3Di_flat
    # tups_3Di, tups_3Di_unique, tups_inverse
    # Bk_2D_CartInt, BkLg_3D_flat, BkLg_3D, PhDenLg_3D
    # xLg_3D, yLg_3D, zLg_3D
    # beta_kxkykz, amp_beta_xyz_preshift, amp_beta_xyz
    # uk2, vk2, uk, vk, uB_kxkykz, uB_xyz, vB_kxkykz, vB_xyz
    # nxyz, np_xyz, na_xyz, na_xyz_norm
    # PhDenLg_xz_slice, PhDenLg_xy_slice, PhDenLg_yz_slice
    # np_xz_slice, np_xy_slice, np_yz_slice
    # na_xz_slice, na_xy_slice, na_yz_slice
    # PhDen_xz_slice_da, PhDen_xy_slice_da, PhDen_yz_slice_da
    # np_xz_slice_da, np_xy_slice_da, np_yz_slice_da, na_xz_slice_da, na_xy_slice_da, na_yz_slice_da, interp_ds


def reconstructMomDists(CSAmp_ds, linDimMajor, linDimMinor, dkxL, dkyL, dkzL):
    import pf_dynamic_cart as pfc
    # Set up
    P = CSAmp_ds['P'].values
    aIBi = CSAmp_ds.attrs['aIBi']
    n0 = CSAmp_ds.attrs['n0']; gBB = CSAmp_ds.attrs['gBB']; mI = CSAmp_ds.attrs['mI']; mB = CSAmp_ds.attrs['mB']

    kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    kVec = kgrid.getArray('k')
    thVec = kgrid.getArray('th')
    print('P: {0}'.format(P))
    print('dk: {0}'.format(kVec[1] - kVec[0]))
    Bk_2D_vals = CSAmp_ds.values
    Nph = CSAmp_ds.attrs['Nph']
    kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # Normalization of the original data array - this checks out
    dk = kg[1, 0] - kg[0, 0]
    dth = thg[0, 1] - thg[0, 0]
    PhDen_Sph = ((1 / Nph) * np.abs(Bk_2D_vals)**2).real.astype(float)
    Bk_norm = np.sum(dk * dth * (2 * np.pi)**(-2) * kg**2 * np.sin(thg) * PhDen_Sph)
    print('Original (1/Nph)|Bk|^2 normalization (Spherical 2D): {0}'.format(Bk_norm))
    Bk2_2D = (np.abs(Bk_2D_vals)**2).real.astype(float)
    # jac = (kg**2) * np.sin(thg) * dk * dth  # dphi missing somehow? or should just divide by 2pi when doing interpolation?
    jac = 1
    Bk2Jac_2D = jac * Bk2_2D
    # Create linear 3D cartesian grid and reinterpolate Bk_3D onto this grid
    kmin = np.min(kVec)
    kxL_pos = np.arange(kmin, linDimMinor, dkxL); kxL = np.concatenate((kmin - 1 * np.flip(kxL_pos[1:], axis=0), kxL_pos))
    kyL_pos = np.arange(kmin, linDimMinor, dkyL); kyL = np.concatenate((kmin - 1 * np.flip(kyL_pos[1:], axis=0), kyL_pos))
    kzL_pos = np.arange(kmin, linDimMajor, dkzL); kzL = np.concatenate((kmin - 1 * np.flip(kzL_pos[1:], axis=0), kzL_pos))
    print('size - kxL: {0}, kyL: {1}, kzL: {2}'.format(kxL.size, kyL.size, kzL.size))
    kxLg_3D, kyLg_3D, kzLg_3D = np.meshgrid(kxL, kyL, kzL, indexing='ij')
    print('dkxL: {0}, dkyL: {1}, dkzL: {2}'.format(dkxL, dkyL, dkzL))
    # Re-interpret grid points of linear 3D Cartesian as nonlinear 3D spherical grid, find unique (k,th) points
    kg_3Di = np.sqrt(kxLg_3D**2 + kyLg_3D**2 + kzLg_3D**2)
    thg_3Di = np.arccos(kzLg_3D / kg_3Di)
    # phig_3Di = np.arctan2(kyLg_3D, kxLg_3D)
    kg_3Di_flat = kg_3Di.reshape(kg_3Di.size)
    thg_3Di_flat = thg_3Di.reshape(thg_3Di.size)
    tups_3Di = np.column_stack((kg_3Di_flat, thg_3Di_flat))
    tups_3Di_unique, tups_inverse = np.unique(tups_3Di, return_inverse=True, axis=0)
    # Perform interpolation on 2D projection and reconstruct full matrix on 3D linear cartesian grid
    print('3D Cartesian grid Ntot: {:1.2E}'.format(kzLg_3D.size))
    print('Unique interp points: {:1.2E}'.format(tups_3Di_unique[:, 0].size))
    interpstart = timer()
    PhDen_2D_CartInt = interpolate.griddata((kg.flatten(), thg.flatten()), PhDen_Sph.flatten(), tups_3Di_unique, method='linear')
    # Bk2Jac_2D_CartInt = interpolate.griddata((kg.flatten(), thg.flatten()), Bk2Jac_2D.flatten(), tups_3Di_unique, method='nearest')
    # Bk_2D_CartInt = interpolate.griddata((kg.flatten(), thg.flatten()), Bk_2D_vals.flatten(), tups_3Di_unique, method='linear')
    interpend = timer()
    print('Interp Time: {0}'.format(interpend - interpstart))
    # Bk2Lg_3D_flat = Bk2Jac_2D_CartInt[tups_inverse]
    # Bk2Lg_3D = Bk2Lg_3D_flat.reshape(kg_3Di.shape)
    # Bk2Lg_3D[np.isnan(Bk2Lg_3D)] = 0
    # PhDenLg_3D = ((1 / Nph) * Bk2Lg_3D).real.astype(float)
    PhDen_3D_flat = PhDen_2D_CartInt[tups_inverse]
    PhDenLg_3D = PhDen_3D_flat.reshape(kg_3Di.shape).real.astype(float)
    PhDenLg_3D[np.isnan(PhDenLg_3D)] = 0
    PhDenLg_3D_norm = np.sum(dkxL * dkyL * dkzL * (2 * np.pi)**(-3) * PhDenLg_3D)

    cart_mask = kg <= linDimMajor
    kg_red = kg[cart_mask]
    thg_red = thg[cart_mask]
    Bk2_2D_red = Bk2_2D[cart_mask]
    Nph_red = np.sum(dk * dth * (2 * np.pi)**(-2) * kg_red**2 * np.sin(thg_red) * Bk2_2D_red)
    print('Rough percentage of phonons in reduced Cartesian grid (Calculated from Spherical 2D): {0}'.format(Nph_red / Nph))

    Bk2Lg_3D = (Nph_red / PhDenLg_3D_norm) * PhDenLg_3D  # SHOULD ACTUALLY MULTIPLY BY WEIGHT OF N_PH LIMITED TO REGION OF CARTESIAN GRID
    Bk2Lg_3D_norm = (1 / Nph) * np.sum(dkxL * dkyL * dkzL * (2 * np.pi)**(-3) * Bk2Lg_3D)
    print('Interpolated (1/Nph)|Bk|^2 normalization (Linear Cartesian 3D): {0}'.format(PhDenLg_3D_norm))
    print('Interpolated (1/Nph)|Bk|^2 forced normalization (Linear Cartesian 3D): {0}'.format(Bk2Lg_3D_norm))
    # Calculate total phonon momentum distribution
    xL = np.fft.fftshift(np.fft.fftfreq(kxL.size) * 2 * np.pi / dkxL)
    yL = np.fft.fftshift(np.fft.fftfreq(kyL.size) * 2 * np.pi / dkyL)
    zL = np.fft.fftshift(np.fft.fftfreq(kzL.size) * 2 * np.pi / dkzL)
    dxL = xL[1] - xL[0]; dyL = yL[1] - yL[0]; dzL = zL[1] - zL[0]
    dVxyz = dxL * dyL * dzL
    xLg_3D, yLg_3D, zLg_3D = np.meshgrid(xL, yL, zL, indexing='ij')
    beta2_kxkykz = Bk2Lg_3D
    beta2_xyz_preshift = np.fft.ifftn(beta2_kxkykz) / dVxyz
    beta2_xyz = np.fft.fftshift(beta2_xyz_preshift)
    decay_length = 5
    decay_xyz = np.exp(-1 * (xLg_3D**2 + yLg_3D**2 + zLg_3D**2) / (2 * decay_length**2))
    fexp = (np.exp(beta2_xyz - Nph_red) - np.exp(-Nph_red)) * decay_xyz
    nPB_preshift = np.fft.fftn(fexp) * dVxyz
    nPB_complex = np.fft.fftshift(nPB_preshift) / ((2 * np.pi)**3)  # this is the phonon momentum distribution in 3D Cartesian coordinates
    nPB = np.abs(nPB_complex)
    nPB_deltaK0 = np.exp(-Nph_red)
    # Produce impurity momentum and total phonon momentum magnitude distributions
    kgrid_L = Grid.Grid('CARTESIAN_3D')
    kgrid_L.initArray_premade('kx', kxL); kgrid_L.initArray_premade('ky', kyL); kgrid_L.initArray_premade('kz', kzL)
    PIgrid = pfc.ImpMomGrid_from_PhononMomGrid(kgrid_L, P)
    PB_x = kxL; PB_y = kyL; PB_z = kzL
    PI_x = PIgrid.getArray('kx'); PI_y = PIgrid.getArray('ky'); PI_z = PIgrid.getArray('kz')
    [PBm, nPBm, PIm, nPIm] = pfc.xyzDist_To_magDist(kgrid_L, nPB, P)
    # Create DataSet for 3D Betak and position distribution slices
    Nx = len(kxL); Ny = len(kyL); Nz = len(kzL)
    PhDen_xz_slice_da = xr.DataArray(PhDenLg_3D[:, Ny // 2, :], coords=[kxL, kzL], dims=['kx', 'kz'])
    PhDen_xy_slice_da = xr.DataArray(PhDenLg_3D[:, :, Nz // 2], coords=[kxL, kyL], dims=['kx', 'ky'])
    PhDen_yz_slice_da = xr.DataArray(PhDenLg_3D[Nx // 2, :, :], coords=[kyL, kzL], dims=['ky', 'kz'])
    nPB_xz_slice = nPB[:, Ny // 2, :]; nPB_xz_slice_da = xr.DataArray(nPB_xz_slice, coords=[PB_x, PB_z], dims=['PB_x', 'PB_z'])
    nPB_xy_slice = nPB[:, :, Nz // 2]; nPB_xy_slice_da = xr.DataArray(nPB_xy_slice, coords=[PB_x, PB_y], dims=['PB_x', 'PB_y'])
    nPB_yz_slice = nPB[Nx // 2, :, :]; nPB_yz_slice_da = xr.DataArray(nPB_yz_slice, coords=[PB_y, PB_z], dims=['PB_y', 'PB_z'])
    nPI_xz_slice = np.flip(np.flip(nPB_xz_slice, 0), 1); nPI_xz_slice_da = xr.DataArray(nPI_xz_slice, coords=[PI_x, PI_z], dims=['PI_x', 'PI_z'])
    nPI_xy_slice = np.flip(np.flip(nPB_xy_slice, 0), 1); nPI_xy_slice_da = xr.DataArray(nPI_xy_slice, coords=[PI_x, PI_y], dims=['PI_x', 'PI_y'])
    nPI_yz_slice = np.flip(np.flip(nPB_yz_slice, 0), 1); nPI_yz_slice_da = xr.DataArray(nPI_yz_slice, coords=[PI_y, PI_z], dims=['PI_y', 'PI_z'])
    nPBm_da = xr.DataArray(nPBm, coords=[PBm], dims=['PB_mag'])
    nPIm_da = xr.DataArray(nPIm, coords=[PIm], dims=['PI_mag'])
    data_dict = ({'PhDen_xz': PhDen_xz_slice_da, 'PhDen_xy': PhDen_xy_slice_da, 'PhDen_yz': PhDen_yz_slice_da,
                  'nPB_xz_slice': nPB_xz_slice_da, 'nPB_xy_slice': nPB_xy_slice_da, 'nPB_yz_slice': nPB_yz_slice_da, 'nPB_mag': nPBm_da,
                  'nPI_xz_slice': nPI_xz_slice_da, 'nPI_xy_slice': nPI_xy_slice_da, 'nPI_yz_slice': nPI_yz_slice_da, 'nPI_mag': nPIm_da})
    coords_dict = {'kx': kxL, 'ky': kyL, 'kz': kzL, 'x': xL, 'y': yL, 'z': zL, 'PB_x': PB_x, 'PB_y': PB_y, 'PB_z': PB_z, 'PI_x': PI_x, 'PI_y': PI_y, 'PI_z': PI_z, 'PB_mag': PBm, 'PI_mag': PIm}
    attrs_dict = {'P': P, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'mom_deltapeak': nPB_deltaK0}
    interp_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    # interp_ds.to_netcdf(interpdatapath + '/InterpDat_P_{:.2f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, linDimMajor, linDimMinor))
    return interp_ds


def quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict):
    #
    # do not run this inside CoherentState or PolaronHamiltonian
    import CoherentState
    import PolaronHamiltonian
    # takes parameters, performs dynamics, and outputs desired observables
    [P, aIBi] = cParams
    [xgrid, kgrid, tgrid] = gParams
    [mI, mB, n0, gBB] = sParams

    NGridPoints = kgrid.size()
    k_max = kgrid.getArray('k')[-1]
    kVec = kgrid.getArray('k')
    thVec = kgrid.getArray('th')

    # calculate some parameters
    nu_const = nu(gBB)
    gnum = gIB(kgrid, aIBi, mI, mB, n0, gBB)

    # Initialization CoherentState
    cs = CoherentState.CoherentState(kgrid, xgrid)

    # Initialization PolaronHamiltonian
    Params = [P, aIBi, mI, mB, n0, gBB]
    ham = PolaronHamiltonian.PolaronHamiltonian(cs, Params, toggleDict)

    # Prepare coarse grained time grid where we save data
    tgrid_coarse = tgrid[0:-1:toggleDict['CoarseGrainRate']]
    if tgrid_coarse[-1] != tgrid[-1]:
        tgrid_coarse = np.concatenate((tgrid_coarse, np.array([tgrid[-1]])))

    # Time evolution

    # Initialize observable Data Arrays

    PB_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    NB_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    ReDynOv_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    ImDynOv_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Phase_da = xr.DataArray(np.full(tgrid_coarse.size, np.nan, dtype=float), coords=[tgrid_coarse], dims=['tc'])
    ReAmp_da = xr.DataArray(np.full((tgrid_coarse.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid_coarse, kVec, thVec], dims=['tc', 'k', 'th'])
    ImAmp_da = xr.DataArray(np.full((tgrid_coarse.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid_coarse, kVec, thVec], dims=['tc', 'k', 'th'])

    # PB_da = xr.DataArray(np.full(tgrid_coarse.size, np.nan, dtype=float), coords=[tgrid_coarse], dims=['t'])
    # NB_da = xr.DataArray(np.full(tgrid_coarse.size, np.nan, dtype=float), coords=[tgrid_coarse], dims=['t'])
    # ReDynOv_da = xr.DataArray(np.full(tgrid_coarse.size, np.nan, dtype=float), coords=[tgrid_coarse], dims=['t'])
    # ImDynOv_da = xr.DataArray(np.full(tgrid_coarse.size, np.nan, dtype=float), coords=[tgrid_coarse], dims=['t'])
    # Phase_da = xr.DataArray(np.full(tgrid_coarse.size, np.nan, dtype=float), coords=[tgrid_coarse], dims=['t'])
    # ReAmp_da = xr.DataArray(np.full((tgrid_coarse.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid_coarse, kVec, thVec], dims=['t', 'k', 'th'])
    # ImAmp_da = xr.DataArray(np.full((tgrid_coarse.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid_coarse, kVec, thVec], dims=['t', 'k', 'th'])
    # # ReDeltaAmp_da = xr.DataArray(np.full((tgrid_coarse.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid_coarse, kVec, thVec], dims=['t', 'k', 'th'])
    # # ImDeltaAmp_da = xr.DataArray(np.full((tgrid_coarse.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid_coarse, kVec, thVec], dims=['t', 'k', 'th'])

    start = timer()
    for ind, t in enumerate(tgrid):
        if ind == 0:
            dt = t
            cs.evolve(dt, ham)
        else:
            dt = t - tgrid[ind - 1]
            cs.evolve(dt, ham)

        if t in tgrid_coarse:
            tc_ind = np.nonzero(tgrid_coarse == t)[0][0]
            # PB_da[tc_ind] = cs.get_PhononMomentum()
            # NB_da[tc_ind] = cs.get_PhononNumber()
            # DynOv = cs.get_DynOverlap()
            # ReDynOv_da[tc_ind] = np.real(DynOv)
            # ImDynOv_da[tc_ind] = np.imag(DynOv)
            Phase_da[tc_ind] = cs.get_Phase()
            Amp = cs.get_Amplitude().reshape(len(kVec), len(thVec))
            ReAmp_da[tc_ind] = np.real(Amp)
            ImAmp_da[tc_ind] = np.imag(Amp)

        PB_da[ind] = cs.get_PhononMomentum()
        NB_da[ind] = cs.get_PhononNumber()
        DynOv = cs.get_DynOverlap()
        ReDynOv_da[ind] = np.real(DynOv)
        ImDynOv_da[ind] = np.imag(DynOv)

        # PB_da[ind] = cs.get_PhononMomentum()
        # NB_da[ind] = cs.get_PhononNumber()
        # DynOv = cs.get_DynOverlap()
        # ReDynOv_da[ind] = np.real(DynOv)
        # ImDynOv_da[ind] = np.imag(DynOv)
        # Phase_da[ind] = cs.get_Phase()
        # Amp = cs.get_Amplitude().reshape(len(kVec), len(thVec))
        # ReAmp_da[ind] = np.real(Amp)
        # ImAmp_da[ind] = np.imag(Amp)

        # amplitude = cs.get_Amplitude()
        # PB = np.dot(ham.kz * np.abs(amplitude)**2, cs.dVk)
        # betaSum = amplitude + np.conjugate(amplitude)
        # xp = 0.5 * np.dot(ham.Wk_grid, betaSum * cs.dVk)
        # betaDiff = amplitude - np.conjugate(amplitude)
        # xm = 0.5 * np.dot(ham.Wki_grid, betaDiff * cs.dVk)

        # damp = -1j * (ham.gnum * np.sqrt(n0) * ham.Wk_grid +
        #               amplitude * (ham.Omega0_grid - ham.kz * (P - PB) / mI) +
        #               ham.gnum * (ham.Wk_grid * xp + ham.Wki_grid * xm))

        # DeltaAmp = damp.reshape(len(kVec), len(thVec))
        # ReDeltaAmp_da[ind] = np.real(DeltaAmp)
        # ImDeltaAmp_da[ind] = np.imag(DeltaAmp)

        end = timer()
        print('t: {:.2f}, cst: {:.2f}, dt: {:.3f}, runtime: {:.3f}'.format(t, cs.time, dt, end - start))
        start = timer()

    # Create Data Set

    data_dict = {'Pph': PB_da, 'Nph': NB_da, 'Real_DynOv': ReDynOv_da, 'Imag_DynOv': ImDynOv_da, 'Phase': Phase_da, 'Real_CSAmp': ReAmp_da, 'Imag_CSAmp': ImAmp_da}
    # data_dict = {'Pph': PB_da, 'Nph': NB_da, 'Real_DynOv': ReDynOv_da, 'Imag_DynOv': ImDynOv_da, 'Phase': Phase_da, 'Real_CSAmp': ReAmp_da, 'Imag_CSAmp': ImAmp_da, 'Real_Delta_CSAmp': ReDeltaAmp_da, 'Imag_Delta_CSAmp': ImDeltaAmp_da}
    coords_dict = {'t': tgrid, 'tc': tgrid_coarse}
    attrs_dict = {'NGridPoints': NGridPoints, 'k_mag_cutoff': k_max, 'P': P, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'nu': nu_const, 'gIB': gnum}

    dynsph_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)

    return dynsph_ds
