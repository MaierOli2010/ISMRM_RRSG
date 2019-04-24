#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 1 2019

@author: omaier

Copyright 2019 Oliver Maier

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import os
import h5py
import pyopencl as cl
import pyopencl.array as clarray
import argparse
from rrsg_cgreco._helper_fun import goldcomp as goldcomp
from rrsg_cgreco._helper_fun import utils
from rrsg_cgreco._CG_reco import CGReco as CGReco
import sys

DTYPE = np.complex64
DTYPE_real = np.float32
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"
path = os.environ["TOOLBOX_PATH"] + "/python/"
sys.path.append(path)
from bart import bart


def run(config='default', InScale=1, denscor=1,
        data='rawdata_brain_radial_96proj_12ch.h5', acc=1,
        ogf='1.706'):
    '''
    Function to run the CG reco of radial data.
    '''
    parser = argparse.ArgumentParser(description='CG Sense Reconstruction')
    parser.add_argument(
      '--config', default=config, dest='config',
      help='Name of config file to use (assumed to be in the same folder). \
 If not specified, use default parameters.')
    parser.add_argument(
      '--InScale', default=InScale, type=int, dest='inscale',
      help='Perform Intensity Scaling.')
    parser.add_argument(
      '--denscor', default=denscor, type=int, dest='denscor',
      help='Perform density correction.')
    parser.add_argument(
      '--data', default=data, dest='data',
      help='Path to the h5 data file.')
    parser.add_argument(
      '--acc', default=acc, type=int, dest='acc',
      help='Desired acceleration factor.')
    parser.add_argument(
      '--ogf', default=ogf, type=str, dest='ogf',
      help='Overgridfactor. 1.706 for Brain, 1+1/3 for heart data.')
    args = parser.parse_args()
    _run_reco(args)


def _run_reco(args):
    np.seterr(divide='ignore', invalid='ignore')
# Create par struct to store parameters
    par = {}
###############################################################################
# Read Input data   ###########################################################
###############################################################################
    if args.data == '':
        raise ValueError("No data file specified")

    name = os.path.normpath(args.data)
    fname = name.split(os.sep)[-1]
    h5_dataset = h5py.File(name, 'r')
    par["file"] = h5_dataset
    h5_dataset_rawdata_name = 'rawdata'
    h5_dataset_trajectory_name = 'trajectory'

    if "heart" in args.data:
        if args.acc == 2:
            R = 33
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[
                :, :, :33]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[
                :, :, :33, :]
        elif args.acc == 3:
            R = 22
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[
                :, :, :22]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[
                :, :, :22, :]
        elif args.acc == 4:
            R = 11
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[
                :, :, :11]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[
                :, :, :11, :]
        else:
            R = 55
            trajectory = h5_dataset.get(h5_dataset_trajectory_name)[...]
            rawdata = h5_dataset.get(h5_dataset_rawdata_name)[...]
    else:
        R = args.acc
        trajectory = h5_dataset.get(h5_dataset_trajectory_name)[
            :, :, ::R]
        rawdata = h5_dataset.get(h5_dataset_rawdata_name)[
            :, :, ::R, :]

    [dummy, nFE, nSpokes, nCh] = rawdata.shape

###############################################################################
# Read Data ###################################################################
###############################################################################
    par["ogf"] = float(eval(args.ogf))
    dimX, dimY, NSlice = [int(nFE/par["ogf"]), int(nFE/par["ogf"]), 1]
    data = np.require(np.squeeze(rawdata.T)[None, :, None, ...],
                      requirements='C')
    par["traj"] = np.require((
        trajectory[0]/(2*np.max(trajectory[0])) +
        1j*trajectory[1]/(2*np.max(trajectory[0]))).T[None, ...],
                   requirements='C')

    par["dcf"] = np.sqrt(np.array(goldcomp.cmp(
                     par["traj"]), dtype=DTYPE_real)).astype(DTYPE_real)
    par["dcf"] = np.require(np.abs(par["dcf"]),
                            DTYPE_real, requirements='C')
    [NScan, NC, reco_Slices, Nproj, N] = data.shape
###############################################################################
# Set sequence related parameters #############################################
###############################################################################
    par["NC"] = NC
    par["dimY"] = dimY
    par["dimX"] = dimX
    par["NSlice"] = NSlice
    par["NScan"] = NScan
    par["N"] = N
    par["Nproj"] = Nproj
###############################################################################
# Create OpenCL Context and Queues ############################################
###############################################################################
    platforms = cl.get_platforms()
    par["GPU"] = False
    par["Platform_Indx"] = 0
    for j in range(len(platforms)):
        if platforms[j].get_devices(device_type=cl.device_type.GPU):
            print("GPU OpenCL platform <%s> found\
 with %i device(s) and OpenCL-version <%s>"
                  % (str(platforms[j].get_info(cl.platform_info.NAME)),
                     len(platforms[j].get_devices(
                        device_type=cl.device_type.GPU)),
                     str(platforms[j].get_info(cl.platform_info.VERSION))))
            par["GPU"] = True
            par["Platform_Indx"] = j
    if not par["GPU"]:
        print("No GPU OpenCL platform found. Returning.")

    par["ctx"] = []
    par["queue"] = []
    num_dev = len(platforms[par["Platform_Indx"]].get_devices())
    par["num_dev"] = num_dev
    for device in range(num_dev):
        dev = []
        dev.append(platforms[par["Platform_Indx"]].get_devices()[device])
        tmp = cl.Context(dev)
        par["ctx"].append(tmp)
        par["queue"].append(
         cl.CommandQueue(
          tmp,
          platforms[par["Platform_Indx"]].get_devices()[device],
          properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
          | cl.command_queue_properties.PROFILING_ENABLE))
###############################################################################
# Coil Sensitivity Estimation #################################################
###############################################################################
    img_igrid = bart(1, 'nufft -i -t', trajectory, rawdata)
    img_igrid_sos = bart(1, 'rss 8', img_igrid)
    img_igrid_sos = np.abs(img_igrid_sos).astype(DTYPE)

    try:
        slices_coils = par["file"]['Coils'][()].shape[1]
        print("Using precomputed coil sensitivities")
        par["C"] = par["file"]['Coils'][
            :, int(slices_coils/2) - int(np.floor((par["NSlice"])/2)):
            int(slices_coils/2) + int(np.ceil(par["NSlice"]/2)), ...]\
            .astype(DTYPE)

        par["InScale"] = par["file"]["InScale"][
         int(slices_coils/2)-int(np.floor((par["NSlice"])/2)):
         int(slices_coils/2)+int(np.ceil(par["NSlice"]/2)), ...]\
            .astype(DTYPE_real)
    except KeyError:
        img_igrid = bart(1, 'nufft -i -t', trajectory, rawdata)
        data_bart = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
            img_igrid.T, (-2, -1)), norm='ortho'), (-2, -1))
        data_bart = np.require(np.squeeze(data_bart.T).astype(DTYPE),
                               requirements='C')[None, ...]
        sens_maps = bart(1, 'ecalib -m1 -I', data_bart)
        sens_maps = np.require(np.squeeze(sens_maps).T, requirements='C')
        par["C"] = sens_maps[:, None, ...]
        par["C"] = np.require(np.transpose(par["C"], (0, 1, 3, 2)),
                              requirements='C')
        sumSqrC = np.sqrt(np.sum(np.abs(par["C"] * np.conj(par["C"])), 0))
        par["C"] = par["C"] / np.tile(sumSqrC, (par["NC"], 1, 1, 1))
        par["C"][~np.isfinite(par["C"])] = 0
#        #### Remove backfoled part at the top
#        par["C"][:, :, :20, :] = 0
        par["InScale"] = sumSqrC
        par["file"].create_dataset('Coils', shape=par["C"].shape,
                                   dtype=DTYPE, data=par["C"])
        par["file"].create_dataset('InScale', shape=sumSqrC.shape,
                                   dtype=DTYPE_real, data=sumSqrC)
        del sumSqrC
    par["file"].close()
###############################################################################
# Set Intensity and Density Scaling ###########################################
###############################################################################
    if args.inscale:
        pass
    else:
        par["C"] *= par["InScale"]
        par["InScale"] = np.ones_like(par["InScale"])
    if args.denscor:
        data = data*(par["dcf"])
    else:
        par["dcf"] = np.ones_like(par["dcf"])
###############################################################################
# generate nFFT  ##############################################################
###############################################################################
    FFT = utils.NUFFT(par)

    def nFTH(x, fft, par):
        siz = np.shape(x)
        result = np.require(np.zeros((par["NC"], par["NSlice"], par["NScan"],
                            par["dimY"], par["dimX"]), dtype=DTYPE),
                            requirements='C')
        tmp_result = clarray.empty(fft.queue, (par["NScan"], 1, 1,
                                   par["dimY"], par["dimX"]), dtype=DTYPE)
        for j in range(siz[1]):
            for k in range(siz[2]):
                inp = clarray.to_device(fft.queue,
                                        np.require(x[:, j, k, ...][
                                                     :, None, None, ...],
                                                   requirements='C'))
                fft.adj_NUFFT(tmp_result, inp)
                result[j, k, ...] = np.squeeze(tmp_result.get())
        return np.transpose(result, (2, 0, 1, 3, 4))
    images_coils = nFTH(data, FFT, par)
    images = np.require(np.sum(images_coils *
                               (np.conj(par["C"])), axis=1),
                        requirements='C')
    del FFT, nFTH

    opt = CGReco(par)
    opt.data = data
###############################################################################
# Start Reco ##################################################################
###############################################################################
    opt.reco_par = utils.read_config(args.config, "DEFAULT")
    opt.execute()
    result = (opt.result)
    res = opt.res
    del opt
###############################################################################
# New .hdf5 save files ########################################################
###############################################################################
    outdir = ""
    if "heart" in args.data:
        outdir += "/heart"
    elif "brain" in args.data:
        outdir += "/brain"
    if not os.path.exists('./output'):
        os.makedirs('output')
    if not os.path.exists('./output' + outdir):
        os.makedirs("./output" + outdir)
    cwd = os.getcwd()
    os.chdir("./output" + outdir)
    f = h5py.File("CG_reco_InScale_" + str(args.inscale) + "_denscor_"
                  + str(args.denscor) + "_reduction_" + str(R) +
                  "_acc_" + str(args.acc) + "_" + fname, "w")
    f.create_dataset("images_ifft_", images.shape, dtype=DTYPE,
                     data=images)
    f.create_dataset("images_ifft_coils_", images_coils.shape, dtype=DTYPE,
                     data=images_coils)
    f.create_dataset("CG_reco", result.shape,
                     dtype=DTYPE, data=result)
    f.create_dataset('InScale', shape=par["InScale"].shape,
                     dtype=DTYPE_real, data=par["InScale"])
    f.create_dataset('Bart_ref', shape=img_igrid_sos.shape,
                     dtype=DTYPE, data=img_igrid_sos)
    f.attrs['res'] = res
    f.flush()
    f.close()
    os.chdir(cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CG Sense Reconstruction')
    parser.add_argument(
      '--config', default='default', dest='config',
      help='Name of config file to use (assumed to be in the same folder). \
 If not specified, use default parameters.')
    parser.add_argument(
      '--InScale', default=1, type=int, dest='inscale',
      help='Perform Intensity Scaling.')
    parser.add_argument(
      '--denscor', default=1, type=int, dest='denscor',
      help='Perform density correction.')
    parser.add_argument(
      '--data', default='rawdata_brain_radial_96proj_12ch.h5', dest='data',
      help='Path to the h5 data file.')
    parser.add_argument(
      '--acc', default=1, type=int, dest='acc',
      help='Desired acceleration factor.')
    parser.add_argument(
      '--ogf', default="1.706", type=str, dest='ogf',
      help='Overgridfactor. 1.706 for Brain, 1+1/3 for heart data.')
    args = parser.parse_args()
    _run_reco(args)
