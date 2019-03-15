import numpy as np
import Grid
import pf_dynamic_sph
import os
import sys
from timeit import default_timer as timer
from pf_static_sph import aSi_grid


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    # Ntheta = 250
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta)

    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    # for realdyn evolution
    tMax = 100
    dt = 0.2
    CoarseGrainRate = 1

    tgrid = np.arange(0, tMax + dt, dt)

    NGridPoints = (Nk * Ntheta).astype(int)

    print('Total time steps: {0}'.format(tgrid.size))
    # print('UV cutoff: {0}'.format(k_max))
    # print('dk: {0}'.format(dk))
    print('dtheta: {0}'.format(dtheta))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Basic parameters

    mI = 1
    # mI = 10
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05

    sParams = [mI, mB, n0, gBB]

    # Toggle parameters

    toggleDict = {'Location': 'cluster', 'Dynamics': 'real', 'Coupling': 'twophonon', 'IRcuts': 'true', 'Grid': 'spherical', 'Longtime': 'false', 'CoarseGrainRate': CoarseGrainRate}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, mI / mB)
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, mI / mB)
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/scratchlfs/demler_lab/kis/genPol_data/NGridPoints_{:.2E}/massRatio={:.1f}'.format(NGridPoints_cart, mI / mB)

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

    if toggleDict['Longtime'] == 'true':
        innerdatapath = innerdatapath + '_longtime'
    elif toggleDict['Longtime'] == 'false':
        innerdatapath = innerdatapath

    if toggleDict['IRcuts'] == 'true':
        innerdatapath = innerdatapath + '_IRcuts'
    elif toggleDict['IRcuts'] == 'false':
        innerdatapath = innerdatapath

    # if os.path.isdir(datapath[0:-14]) is False:
    #     os.mkdir(datapath[0:-14])

    # if os.path.isdir(datapath) is False:
    #     os.mkdir(datapath)

    # if os.path.isdir(innerdatapath) is False:
    #     os.mkdir(innerdatapath)

    # # # ---- SINGLE FUNCTION RUN ----

    # runstart = timer()

    # P = 1.4
    # aIBi = -0.1

    # print(innerdatapath)
    # # aSi = aSi_grid(kgrid, 0, mI, mB, n0, gBB); aIBi = aIBi - aSi
    # # print(aIBi)

    # cParams = [P, aIBi]

    # dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict)
    # dynsph_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    # end = timer()
    # print('Time: {:.2f}'.format(end - runstart))

    # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi, P VALUES) ----

    allParams_List = []

    aIBi_Vals = np.array([-10.0, -5.0, -2.0])
    P_Vals = np.concatenate((np.linspace(0.1, 0.8, 5, endpoint=False), np.linspace(0.8, 3.0, 22, endpoint=False), np.linspace(3.0, 5.0, 3)))
    IRrat_Vals = np.array([2, 5, 10, 100, 4e3])
    innerdatapath = innerdatapath + '_IRcuts'

    # if os.path.isdir(innerdatapath) is False:
    #     os.mkdir(innerdatapath)
    # for IRrat in IRrat_Vals:
    #     IRdatapath = innerdatapath + '/IRratio_{:.1E}'.format(IRrat)
    #     if os.path.isdir(IRdatapath) is False:
    #         os.mkdir(IRdatapath)

    for ind, aIBi in enumerate(aIBi_Vals):
        for P in P_Vals:
            for IRrat in IRrat_Vals:
                allParams_List.append([P, aIBi, IRrat])

    print(len(allParams_List))

    # # ---- COMPUTE DATA ON COMPUTER ----
    # print(innerdatapath)

    # runstart = timer()

    # for ind, cParams in enumerate(cParams_List):
    #     loopstart = timer()
    #     [P, aIBi] = cParams
    #     dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict)
    #     # dynsph_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    #     loopend = timer()
    #     print('Index: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(ind, P, aIBi, loopend - loopstart))

    # end = timer()
    # print('Total Time: {:.2f}'.format(end - runstart))

    # ---- COMPUTE DATA ON CLUSTER ----

    runstart = timer()

    taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # taskCount = len(allParams_List)
    # taskID = 332

    if(taskCount != len(allParams_List)):
        print('ERROR: TASK COUNT MISMATCH')
        print(taskCount, len(allParams_List))
        P = float('nan')
        aIBi = float('nan')
        sys.exit()
    else:
        allParams = allParams_List[taskID]
        [P, aIBi, IRrat] = allParams
        cParams = [P, aIBi]

        k_min = IRrat * 1e-5
        k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)
        kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
        kgrid = Grid.Grid("SPHERICAL_2D")
        kgrid.initArray_premade('k', kArray)
        kgrid.initArray_premade('th', thetaArray)
        gParams = [xgrid, kgrid, tgrid]
        IRdatapath = innerdatapath + '/IRratio_{:.1E}'.format(IRrat)

    dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict)
    dynsph_ds.to_netcdf(IRdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    end = timer()
    print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f}, IRratio: {:.1E}, Time: {:.2f}'.format(taskID, P, aIBi, IRrat, end - runstart))
