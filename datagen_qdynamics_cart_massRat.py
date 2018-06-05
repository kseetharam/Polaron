import numpy as np
import pandas as pd
import xarray as xr
import Grid
import pf_dynamic_cart
import os
import sys
from timeit import default_timer as timer
# import pf_static_cart


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    (Nx, Ny, Nz) = (len(xgrid.getArray('x')), len(xgrid.getArray('y')), len(xgrid.getArray('z')))

    kxfft = np.fft.fftfreq(Nx) * 2 * np.pi / dx; kyfft = np.fft.fftfreq(Nx) * 2 * np.pi / dy; kzfft = np.fft.fftfreq(Nx) * 2 * np.pi / dz

    kgrid = Grid.Grid('CARTESIAN_3D')
    kgrid.initArray_premade('kx', np.fft.fftshift(kxfft)); kgrid.initArray_premade('ky', np.fft.fftshift(kyfft)); kgrid.initArray_premade('kz', np.fft.fftshift(kzfft))

    kx = kgrid.getArray('kx')

    tMax = 1000
    dt = 10

    # tMax = 100
    # dt = 0.2

    tgrid = np.arange(0, tMax + dt, dt)

    gParams = [xgrid, kgrid, tgrid]

    # NGridPoints = (2 * Lx / dx) * (2 * Ly / dy) * (2 * Lz / dz)
    NGridPoints = xgrid.size()

    kx = kgrid.getArray('kx'); ky = kgrid.getArray('ky'); kz = kgrid.getArray('kz')
    k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 + np.max(kz)**2)

    print('datagen_qdynamics_cart_massRat')
    print('Total time steps: {0}'.format(tgrid.size))
    print('UV cutoff: {0}'.format(k_max))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Basic parameters

    # Toggle parameters

    toggleDict = {'Location': 'cluster', 'Dynamics': 'imaginary', 'Coupling': 'twophonon', 'Grid': 'cartesian'}

    # ---- SET PARAMS ----

    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05

    Params_List = []
    mI_Vals = np.array([1, 2, 5, 10])
    aIBi_Vals = np.array([-10.0, -5.0, -2.0])
    # P_Vals = np.array([0.1, 0.4, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 2.4, 2.7, 3.0, 4.0, 5.0])
    P_Vals = np.array([3.2, 3.4, 3.6, 3.8, 3.9, 4.1, 4.2, 4.4, 4.6, 4.8, 5.2, 5.4, 5.6, 5.8, 6.0])

    for mI in mI_Vals:
        for aIBi in aIBi_Vals:
            for P in P_Vals:
                sParams = [mI, mB, n0, gBB]
                cParams = [P, aIBi]
                if toggleDict['Location'] == 'home':
                    datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints, mI / mB)
                elif toggleDict['Location'] == 'work':
                    datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints, mI / mB)
                elif toggleDict['Location'] == 'cluster':
                    datapath = '/n/regal/demler_lab/kis/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints, mI / mB)
                if toggleDict['Dynamics'] == 'real':
                    innerdatapath = datapath + '/redyn'
                elif toggleDict['Dynamics'] == 'imaginary':
                    innerdatapath = datapath + '/imdyn'
                if toggleDict['Grid'] == 'cartesian':
                    innerdatapath = innerdatapath + '_cart'
                elif toggleDict['Grid'] == 'spherical':
                    innerdatapath = innerdatapath + '_spherical'
                if toggleDict['Coupling'] == 'frohlich':
                    innerdatapath = innerdatapath + '_froh'
                elif toggleDict['Coupling'] == 'twophonon':
                    innerdatapath = innerdatapath
                Params_List.append([sParams, cParams, innerdatapath])

    # # ---- COMPUTE DATA ON COMPUTER ----

    # runstart = timer()

    # for ind, Params in enumerate(Params_List):
    #     loopstart = timer()
    #     [sParams, cParams, innerdatapath] = Params_List[ind]
    #     [mI, mB, n0, gBB] = sParams
    #     [P, aIBi] = cParams
    #     dyncart_ds = pf_dynamic_cart.quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict)
    #     dyncart_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    #     loopend = timer()
    #     print('Index: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(ind, P, aIBi, loopend - loopstart))

    # end = timer()
    # print('Total Time: {:.2f}'.format(end - runstart))

    # ---- COMPUTE DATA ON CLUSTER ----

    runstart = timer()

    taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    if(taskCount > len(Params_List)):
        print('ERROR: TASK COUNT MISMATCH')
        P = float('nan')
        aIBi = float('nan')
        sys.exit()
    else:
        [sParams, cParams, innerdatapath] = Params_List[taskID]
        [mI, mB, n0, gBB] = sParams
        [P, aIBi] = cParams

    dyncart_ds = pf_dynamic_cart.quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict)
    dyncart_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    end = timer()
    print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
