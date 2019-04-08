#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 1 2019

@author: omaier
"""
import numpy as np
import time
import pyopencl as cl
import pyopencl.array as clarray
from transforms.pyopencl_nufft import PyOpenCLNUFFT as NUFFT

DTYPE = np.complex64
DTYPE_real = np.float32


class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build("-cl-mad-enable -cl-fast-relaxed-math")
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel


class CGReco:
    def __init__(self, par):
        self.C = par["C"]
        self.traj = par["traj"]
        self.NSlice = par["NSlice"]
        self.NScan = par["NScan"]
        self.dimX = par["dimX"]
        self.dimY = par["dimY"]
        self.NC = par["NC"]
        self.fval_min = 0
        self.fval = 0
        self.ctx = par["ctx"][0]
        self.queue = par["queue"][0]
        self.res = []
        self.N = par["N"]
        self.Nproj = par["Nproj"]
        self.incor = par["InScale"].astype(DTYPE)
        self.coil_buf = cl.Buffer(self.ctx,
                                  cl.mem_flags.READ_ONLY |
                                  cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=self.C.data)
        self.tmp_sino = clarray.empty(self.queue,
                                      (self.NScan, self.NC, self.NSlice,
                                       self.Nproj, self.N), DTYPE, "C")
        self.tmp_result = clarray.empty(self.queue,
                                        (self.NScan, self.NC, self.NSlice,
                                         self.dimY, self.dimX), DTYPE, "C")
        self.NUFFT = NUFFT(self.ctx, self.queue, par,
                           overgridfactor=par["ogf"])

        self.prg = Program(self.ctx,
                           open('./kernels/opencl_operator_kernels.c').read())

    def eval_fwd_kspace(self, y, x, wait_for=[]):
        return self.prg.operator_fwd(self.queue,
                                     (self.NSlice, self.dimY, self.dimX), None,
                                     y.data, x.data, self.coil_buf,
                                     np.int32(self.NC), np.int32(self.NSlice),
                                     np.int32(self.NScan),
                                     wait_for=wait_for)

    def operator_lhs(self, out, x):
        self.tmp_result.add_event(self.eval_fwd_kspace(
            self.tmp_result, x, wait_for=self.tmp_result.events+x.events))
        self.tmp_sino.add_event(self.NUFFT.fwd_NUFFT(
            self.tmp_sino, self.tmp_result))
        return self.operator_rhs(out, self.tmp_sino)

    def operator_rhs(self, out, x, wait_for=[]):
        self.tmp_result.add_event(self.NUFFT.adj_NUFFT(
            self.tmp_result, x, wait_for=wait_for+x.events))
        return self.prg.operator_ad(self.queue,
                                    (self.NSlice, self.dimY, self.dimX), None,
                                    out.data, self.tmp_result.data,
                                    self.coil_buf, np.int32(self.NC),
                                    np.int32(self.NSlice),
                                    np.int32(self.NScan),
                                    wait_for=(self.tmp_result.events +
                                              out.events+wait_for))

    def kspace_filter(self, x):
        print("Performing k-space filtering")
        beta = 100
        kc = 25
        kpoints = np.arange(-np.floor(self.dimX/2), np.ceil(self.dimX/2))
        filter_vec = 1/2+1/np.pi*np.arctan(beta*(kc - np.abs(kpoints)/kc))
        filter_kspace = np.outer(filter_vec, filter_vec)
        x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
            np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
                    x, (-2, -1)), norm='ortho'),
                    (-2, -1))*filter_kspace, (-2, -1)),
                    norm='ortho'), (-2, -1))
        return x

###############################################################################
#   Start a Reconstruction ####################################################
#   Call inner optimization ###################################################
#   output: optimal value of x ################################################
###############################################################################
    def execute(self):
        iters = self.reco_par["max_iters"]
        guess = np.zeros((iters+1, 1, 1, self.NSlice, self.dimY, self.dimX),
                         dtype=DTYPE)
        start = time.time()
        result = self.cg_solve(guess, iters)
        result = result/self.incor
        result[~np.isfinite(result)] = 0
        end = time.time()-start
        print("-"*80)
        print("Elapsed time: %f seconds" % (end))
        print("-"*80)
        self.result = self.kspace_filter(result)
        print("done")

###############################################################################
#   Conjugate Gradient optimization ###########################################
#   input: initial guess x ####################################################
#          number of iterations iters #########################################
#   output: optimal value of x ################################################
###############################################################################
    def cg_solve(self, x, iters):
        x = clarray.to_device(self.queue, np.require(x, requirements="C"))
        b = clarray.empty(self.queue,
                          (self.NScan, 1, self.NSlice, self.dimY, self.dimX),
                          DTYPE, "C")
        Ax = clarray.empty(self.queue,
                           (self.NScan, 1, self.NSlice, self.dimY, self.dimX),
                           DTYPE, "C")
        data = clarray.to_device(self.queue, self.data)

        self.operator_rhs(b, data)
        res = b
        p = res
        delta = np.linalg.norm(res.get())**2/np.linalg.norm(b.get())**2
        self.res.append(delta)
        print("Initial Residuum: ", delta)

        for i in range(iters):
            self.operator_lhs(Ax, p)
            Ax = Ax + self.reco_par["lambd"]*p
            alpha = (clarray.vdot(res, res)/(clarray.vdot(p, Ax))).real.get()
            x[i+1] = (x[i] + alpha*p)
            res_new = res - alpha*Ax
            delta = np.linalg.norm(res_new.get())**2/np.linalg.norm(b.get())**2
            self.res.append(delta)
            if delta < self.reco_par["tol"]:
                print("Converged after %i iterations to %1.3e." % (i, delta))
                return x.get()[:i+1, ...]
            if not np.mod(i, 1):
                print("Residuum at iter %i : %1.3e" % (i, delta), end='\r')

            beta = (clarray.vdot(res_new, res_new) /
                    clarray.vdot(res, res)).real.get()
            p = res_new+beta*p
            (res, res_new) = (res_new, res)
        return x.get()
