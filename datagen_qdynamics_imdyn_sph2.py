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

    # # (Lx, Ly, Lz) = (30, 30, 30)
    # (Lx, Ly, Lz) = (21, 21, 21)
    # # (Lx, Ly, Lz) = (12, 12, 12)
    # # (dx, dy, dz) = (0.75, 0.75, 0.75)
    # (dx, dy, dz) = (0.375, 0.375, 0.375)
    # # (dx, dy, dz) = (0.25, 0.25, 0.25)

    (Lx, Ly, Lz) = (105, 105, 105)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    Ntheta = 250
    Nk = np.ceil(NGridPoints_desired / Ntheta)

    # # adjust num theta points
    # Ntheta = 25
    # Nk = 85
    # NGridPoints_cart = 0.5 * NGridPoints_cart

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

    # for imdyn evolution
    tMax = 2e4
    dt = 10
    CoarseGrainRate = 100

    # # for realdyn evolution
    # tMax = 200
    # dt = 0.2
    # CoarseGrainRate = 1

    tgrid = np.arange(0, tMax + dt, dt)

    gParams = [xgrid, kgrid, tgrid]
    NGridPoints = kgrid.size()

    print('Total time steps: {0}'.format(tgrid.size))
    print('UV cutoff: {0}'.format(k_max))
    print('dk: {0}'.format(dk))
    print('dtheta: {0}'.format(dtheta))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Basic parameters

    mI = 1
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05

    sParams = [mI, mB, n0, gBB]

    # Toggle parameters

    toggleDict = {'Location': 'cluster', 'Dynamics': 'imaginary', 'Coupling': 'twophonon', 'Grid': 'spherical', 'Longtime': 'false', 'CoarseGrainRate': CoarseGrainRate}

    # ---- SET OUTPUT DATA FOLDER ----

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
        innerdatapath = innerdatapath + '_spherical2'

    if toggleDict['Coupling'] == 'frohlich':
        innerdatapath = innerdatapath + '_froh'
    elif toggleDict['Coupling'] == 'twophonon':
        innerdatapath = innerdatapath

    if toggleDict['Longtime'] == 'true':
        innerdatapath = innerdatapath + '_longtime'
    elif toggleDict['Longtime'] == 'false':
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
    # aIBi = -2

    # # aSi = aSi_grid(kgrid, 0, mI, mB, n0, gBB); aIBi = aIBi - aSi
    # # print(aIBi)

    # cParams = [P, aIBi]

    # dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict)
    # dynsph_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    # end = timer()
    # print('Time: {:.2f}'.format(end - runstart))

    # # # ---- SINGLE FUNCTION RUN GAUSSIAN COMPARISON ----

    # runstart = timer()

    # thetaArray = np.linspace(0, np.pi, 1e3)
    # kArray = np.arange(0.1, 5.1, 0.1)
    # kgrid = Grid.Grid("SPHERICAL_2D")
    # kgrid.initArray_premade('k', kArray)
    # kgrid.initArray_premade('th', thetaArray)
    # # print('{:.2E}'.format(kgrid.size()))
    # tMax = 20
    # dt = 0.1
    # tgrid = np.arange(0, tMax + dt, dt)
    # gParams = [xgrid, kgrid, tgrid]
    # mI = 1e9
    # mB = 1
    # n0 = 1
    # gBB = (4 * np.pi / mB) * 0.065
    # sParams = [mI, mB, n0, gBB]
    # P = 0.05
    # aIBi = -1.2
    # cParams = [P, aIBi]

    # datapath = datapath[0:-22] + '{:.2E}/massRatio=inf'.format(kgrid.size())
    # if toggleDict['Dynamics'] == 'real':
    #     innerdatapath = datapath + '/redyn_spherical'
    #     filepath = innerdatapath + '/cs_mfrt_aIBi_{:.2f}.npy'.format(aIBi)
    # elif toggleDict['Dynamics'] == 'imaginary':
    #     innerdatapath = datapath + '/imdyn_spherical'
    #     filepath = innerdatapath + '/cs_mfit_aIBi_{:.2f}.npy'.format(aIBi)
    # if os.path.isdir(datapath) is False:
    #     os.mkdir(datapath)
    # if os.path.isdir(innerdatapath) is False:
    #     os.mkdir(innerdatapath)

    # dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict)
    # # dynsph_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    # energy_vec = np.zeros(tgrid.size)
    # CSAmp_ds = dynsph_ds['Real_CSAmp'] + 1j * dynsph_ds['Imag_CSAmp']
    # for ind, t in enumerate(tgrid):
    #     CSAmp = CSAmp_ds.sel(t=t).values
    #     energy_vec[ind] = pf_dynamic_sph.Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB)
    # NB_Vec = dynsph_ds['Nph'].values
    # Zfactor_Vec = np.abs((dynsph_ds['Real_DynOv'] + 1j * dynsph_ds['Imag_DynOv']).values)
    # tVec = tgrid
    # Params = [aIBi, mB, n0, gBB]
    # data = [Params, tVec, NB_Vec, Zfactor_Vec, energy_vec]
    # np.save(filepath, data)

    # end = timer()
    # print('Time: {:.2f}'.format(end - runstart))

    # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi, P VALUES) ----

    cParams_List = []

    aIBi_Vals = np.array([-10.0, -5.0, -2.0])
    # aSi = aSi_grid(kgrid, 0, mI, mB, n0, gBB); aIBi_Vals = aIBi_Vals - aSi

    # P_Vals = np.array([0.4])
    P_Vals = np.concatenate((np.array([0.1, 0.4, 0.6]), np.linspace(0.8, 2.8, 20), np.linspace(3.0, 5.0, 3)))

    for ind, aIBi in enumerate(aIBi_Vals):
        for P in P_Vals:
            cParams_List.append([P, aIBi])

    # # ---- COMPUTE DATA ON COMPUTER ----

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

    if(taskCount != len(cParams_List)):
        print('ERROR: TASK COUNT MISMATCH')
        P = float('nan')
        aIBi = float('nan')
        sys.exit()
    else:
        cParams = cParams_List[taskID]
        [P, aIBi] = cParams

    dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict)
    dynsph_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    end = timer()
    print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
