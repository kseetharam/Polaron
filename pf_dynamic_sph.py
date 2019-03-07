import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import quad
from timeit import default_timer as timer
import os

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


def spectFunc(t_Vec, S_Vec):
    # spectral function (Fourier Transform of dynamical overlap)
    tstep = t_Vec[1] - t_Vec[0]
    N = t_Vec.size
    tdecay = 3
    decayFactor = np.exp(-1 * t_Vec / tdecay)
    # decayFactor = 1
    sf = 2 * np.real(np.fft.ifft(S_Vec * decayFactor))
    omega = 2 * np.pi * np.fft.fftfreq(N, d=tstep)
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
    PB_da = xr.DataArray(np.full(tgrid_coarse.size, np.nan, dtype=float), coords=[tgrid_coarse], dims=['t'])
    NB_da = xr.DataArray(np.full(tgrid_coarse.size, np.nan, dtype=float), coords=[tgrid_coarse], dims=['t'])
    ReDynOv_da = xr.DataArray(np.full(tgrid_coarse.size, np.nan, dtype=float), coords=[tgrid_coarse], dims=['t'])
    ImDynOv_da = xr.DataArray(np.full(tgrid_coarse.size, np.nan, dtype=float), coords=[tgrid_coarse], dims=['t'])
    Phase_da = xr.DataArray(np.full(tgrid_coarse.size, np.nan, dtype=float), coords=[tgrid_coarse], dims=['t'])
    ReAmp_da = xr.DataArray(np.full((tgrid_coarse.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid_coarse, kVec, thVec], dims=['t', 'k', 'th'])
    ImAmp_da = xr.DataArray(np.full((tgrid_coarse.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid_coarse, kVec, thVec], dims=['t', 'k', 'th'])
    # ReDeltaAmp_da = xr.DataArray(np.full((tgrid_coarse.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid_coarse, kVec, thVec], dims=['t', 'k', 'th'])
    # ImDeltaAmp_da = xr.DataArray(np.full((tgrid_coarse.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid_coarse, kVec, thVec], dims=['t', 'k', 'th'])

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
            PB_da[tc_ind] = cs.get_PhononMomentum()
            NB_da[tc_ind] = cs.get_PhononNumber()
            DynOv = cs.get_DynOverlap()
            ReDynOv_da[tc_ind] = np.real(DynOv)
            ImDynOv_da[tc_ind] = np.imag(DynOv)
            Phase_da[tc_ind] = cs.get_Phase()
            Amp = cs.get_Amplitude().reshape(len(kVec), len(thVec))
            ReAmp_da[tc_ind] = np.real(Amp)
            ImAmp_da[tc_ind] = np.imag(Amp)

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
    coords_dict = {'t': tgrid_coarse}
    attrs_dict = {'NGridPoints': NGridPoints, 'k_mag_cutoff': k_max, 'P': P, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'nu': nu_const, 'gIB': gnum}

    dynsph_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)

    return dynsph_ds
