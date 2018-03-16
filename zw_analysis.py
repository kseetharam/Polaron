import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import itertools
import Grid
import pf_dynamic_sph as pfs

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta)

    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    # k_max = np.sqrt((np.pi / dx)**2 + (np.pi / dy)**2 + (np.pi / dz)**2)
    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)

    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    if dk < k_min:
        print('k ARRAY GENERATION ERROR')

    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray_premade('k', kArray)
    kgrid.initArray_premade('th', thetaArray)
    dVk = kgrid.dV()

    # Basic parameters

    mI = 1.7
    mB = 1
    n0 = 1
    aBB = 0.075
    gBB = (4 * np.pi / mB) * aBB

    names = list(kgrid.arrays.keys())  # ***need to have arrays added as k, th when kgrid is created
    if names[0] != 'k':
        print('CREATED kgrid IN WRONG ORDER')
    functions_wk = [lambda k: pfs.omegak(k, mB, n0, gBB), lambda th: 0 * th + 1]
    wk = kgrid.function_prod(names, functions_wk)

    # datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    innerdatapath = datapath + '/imdyn_spherical'
    outputdatapath = datapath + '/mm'

    # # Individual Datasets

    tGrid = np.linspace(0, 50, 100)

    for ind, filename in enumerate(os.listdir(innerdatapath)):
        if filename == 'quench_Dataset_sph.nc':
            continue
        ds = xr.open_dataset(innerdatapath + '/' + filename)
        print(filename)
        aIBi = ds.attrs['aIBi']
        P = ds.attrs['P']
        aIBiVec = aIBi * np.ones(tGrid.size)
        PVec = P * np.ones(tGrid.size)

        CSAmp = (ds['Real_CSAmp'] + 1j * ds['Imag_CSAmp']).values
        CSAmp = CSAmp.reshape(CSAmp.size)

        DynOv_Vec = np.zeros(tGrid.size, dtype=complex)
        for tind, t in enumerate(tGrid):
            amplitude = CSAmp * np.exp(-1j * wk * t)
            DynOv_Vec[tind] = np.exp(-1j * t * P / (2 * mI)) * np.exp((-1 / 2) * np.dot(np.abs(amplitude)**2, dVk).real.astype(float))

        data = np.concatenate((PVec[:, np.newaxis], aIBiVec[:, np.newaxis], tGrid[:, np.newaxis], np.real(DynOv_Vec)[:, np.newaxis], np.imag(DynOv_Vec)[:, np.newaxis]), axis=1)
        np.savetxt(outputdatapath + '/quench_P_{:.3f}_aIBi_{:.2f}.dat'.format(P, aIBi), data)
