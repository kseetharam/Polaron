import numpy as np
import pandas as pd
import xarray as xr
import Grid
import pf_dynamic_sph
import os
import sys
from timeit import default_timer as timer
from copy import copy
import itertools


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    higherCutoff = False; cutoffRat = 1.5
    betterResolution = True; resRat = 0.5

    (Lx, Ly, Lz) = (60, 60, 60)
    (dx, dy, dz) = (0.25, 0.25, 0.25)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)

    # Toggle parameters

    toggleDict = {'Location': 'cluster', 'Dynamics': 'real', 'Interaction': 'on', 'Grid': 'spherical', 'Coupling': 'twophonon'}

    # ---- SET PARAMS ----

    tau = 5
    dkxL = 5e-2; dkyL = 5e-2; dkzL = 5e-2
    linDimMajor = 0.99 * (k_max * np.sqrt(2) / 2)
    linDimMinor = linDimMajor

    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05  # Dresher uses aBB ~ 0.2 instead of 0.5 here
    nu = np.sqrt(n0 * gBB / mB)

    Params_List = []
    mI_Vals = np.array([1.0])
    # mI_Vals = np.array([0.5, 1.0, 2, 5.0])
    # aIBi_Vals = np.array([-10.0, -5.0, -2.0, -1.5])
    # aIBi_Vals = np.array([-1.25, -1.0])
    aIBi_Vals = np.array([-10.0, -5.0, -2.0, -1.5, -1.25, -1.0])
    P_Vals_norm = np.concatenate((np.linspace(0.1, 0.8, 5, endpoint=False), np.linspace(0.8, 1.4, 10, endpoint=False), np.linspace(1.4, 3.0, 12, endpoint=False), np.linspace(3.0, 5.0, 10, endpoint=False), np.linspace(5.0, 9.0, 20)))

    for mI in mI_Vals:
        P_Vals = mI * nu * P_Vals_norm
        for aIBi in aIBi_Vals:
            for P in P_Vals:
                cParams = [P, aIBi]
                if toggleDict['Location'] == 'home':
                    datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
                elif toggleDict['Location'] == 'work':
                    datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
                elif toggleDict['Location'] == 'cluster':
                    datapath = '/n/scratchlfs/demler_lab/kis/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
                if higherCutoff is True:
                    datapath = datapath + '_cutoffRat_{:.2f}'.format(cutoffRat)
                if betterResolution is True:
                    datapath = datapath + '_resRat_{:.2f}'.format(resRat)
                gridpath = copy(datapath)
                datapath = datapath + '/massRatio={:.1f}'.format(mI / mB)
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
                Params_List.append([cParams, innerdatapath])

                # if os.path.isdir(innerdatapath + '/amp3D') is False:
                #     os.mkdir(innerdatapath + '/amp3D')

    # # Params_List = Params_List[0:2]
    # remList = []; remList.append(Params_List[9]); remList.append(Params_List[74]); remList.append(Params_List[79]); remList.append(Params_List[102]); remList.append(Params_List[198])
    # Params_List = remList
    print(len(Params_List))

    # # ---- COMPUTE DATA ON COMPUTER ----

    # runstart = timer()

    # for ind, Params in enumerate(Params_List):
    #     loopstart = timer()
    #     [cParams, innerdatapath] = Params_List[ind]
    #     [P, aIBi] = cParams

    #     qds_PaIBi = xr.open_dataset(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    #     qds_PaIBi.attrs['P'] = P
    #     tsVals = qds_PaIBi.coords['tc'].values
    #     tsVals = tsVals[tsVals <= tau]
    #     tsVals = tsVals[1::]

    #     ds_list = []
    #     t_list = []
    #     for tind, t in enumerate(tsVals):
    #         tloopstart = timer()
    #         CSAmp_ds = (qds_PaIBi['Real_CSAmp'] + 1j * qds_PaIBi['Imag_CSAmp']).sel(tc=t); CSAmp_ds.attrs = qds_PaIBi.attrs; CSAmp_ds.attrs['Nph'] = qds_PaIBi['Nph'].sel(t=t).values
    #         interp_ds = pf_dynamic_sph.reconstructMomDists(CSAmp_ds, linDimMajor, linDimMinor, dkxL, dkyL, dkzL)
    #         # interp_ds.to_netcdf(innerdatapath + '/amp3D/interp_P_{:.3f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}_tind_{:d}.nc'.format(P, aIBi, linDimMajor, linDimMinor,tind))
    #         ds_list.append(interp_ds); t_list.append(t)

    #         tloopend = timer()
    #         print('tLoop time: {:.2f}'.format(tloopend - tloopstart))

    #     s = sorted(zip(t_list, ds_list))
    #     g = itertools.groupby(s, key=lambda x: x[0])
    #     t_keys = []; t_ds_list = []
    #     for key, group in g:
    #         t_temp_list, ds_temp_list = zip(*list(group))
    #         t_keys.append(key)  # note that key = t_temp_list[0]
    #         t_ds_list.append(ds_temp_list[0])
    #     with xr.concat(t_ds_list, pd.Index(t_keys, name='t')) as ds_tot:
    #         filename = innerdatapath + '/amp3D/interp_P_{:.3f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, linDimMajor, linDimMinor)
    #         ds_tot.to_netcdf(filename)

    #     loopend = timer()
    #     print('Index: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(ind, P, aIBi, loopend - loopstart))

    # end = timer()
    # print('Total Time: {:.2f}'.format(end - runstart))

    # ---- COMPUTE DATA ON CLUSTER ----

    runstart = timer()

    taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # taskCount = len(Params_List); taskID = 72; print(Params_List[taskID])

    if(taskCount > len(Params_List)):
        print('ERROR: TASK COUNT MISMATCH')
        sys.exit()
    else:
        [cParams, innerdatapath] = Params_List[taskID]
        [P, aIBi] = cParams

    qds_PaIBi = xr.open_dataset(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    qds_PaIBi.attrs['P'] = P
    tsVals = qds_PaIBi.coords['tc'].values
    tsVals = tsVals[tsVals <= tau]
    tsVals = tsVals[1::]
    ds_list = []
    t_list = []
    for tind, t in enumerate(tsVals):
        tloopstart = timer()
        CSAmp_ds = (qds_PaIBi['Real_CSAmp'] + 1j * qds_PaIBi['Imag_CSAmp']).sel(tc=t); CSAmp_ds.attrs = qds_PaIBi.attrs; CSAmp_ds.attrs['Nph'] = qds_PaIBi['Nph'].sel(t=t).values
        interp_ds = pf_dynamic_sph.reconstructMomDists(CSAmp_ds, linDimMajor, linDimMinor, dkxL, dkyL, dkzL)
        # interp_ds.to_netcdf(innerdatapath + '/amp3D/interp_P_{:.3f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}_tind_{:d}.nc'.format(P, aIBi, linDimMajor, linDimMinor,tind))
        ds_list.append(interp_ds); t_list.append(t)
        tloopend = timer()
        print('tLoop time: {:.2f}'.format(tloopend - tloopstart))
    s = sorted(zip(t_list, ds_list))
    g = itertools.groupby(s, key=lambda x: x[0])
    t_keys = []; t_ds_list = []
    for key, group in g:
        t_temp_list, ds_temp_list = zip(*list(group))
        t_keys.append(key)  # note that key = t_temp_list[0]
        t_ds_list.append(ds_temp_list[0])
    with xr.concat(t_ds_list, pd.Index(t_keys, name='t')) as ds_tot:
        filename = innerdatapath + '/amp3D/interp_P_{:.3f}_aIBi_{:.2f}_lDM_{:.2f}_lDm_{:.2f}.nc'.format(P, aIBi, linDimMajor, linDimMinor)
        ds_tot.to_netcdf(filename)

    end = timer()
    print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
