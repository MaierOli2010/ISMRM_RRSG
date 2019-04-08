#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:10:07 2019

@author: omaier
"""
import numpy as np
import configparser
from transforms.pyopencl_nufft import PyOpenCLNUFFT

DTYPE = np.complex64
DTYPE_real = np.float32


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def NUFFT(par):
    NC = par["NC"]
    NSlice = par["NSlice"]
    par["NC"] = 1
    par["NSlice"] = 1
    FFT = PyOpenCLNUFFT(par["ctx"][0], par["queue"][0], par,
                        overgridfactor=par["ogf"])
    par["NC"] = NC
    par["NSlice"] = NSlice
    return FFT


def gen_default_config():

    config = configparser.ConfigParser()
    config['DEFAULT'] = {}
    config['DEFAULT']["max_iters"] = '300'
    config['DEFAULT']["display_iterations"] = 'True'
    config['DEFAULT']["tol"] = '5e-3'
    config['DEFAULT']["lambd"] = '0'
    with open('default.ini', 'w') as configfile:
        config.write(configfile)


def read_config(conf_file, reg_type="DEFAULT"):
    config = configparser.ConfigParser()
    try:
        with open(conf_file+'.ini', 'r') as f:
            config.read_file(f)
    except IOError:
        print("Config file not readable or not found. \
              Generating default config.")
        gen_default_config()
        with open(conf_file+'.ini', 'r') as f:
            config.read_file(f)
    finally:
        params = {}
        for key in config[reg_type]:
            if key in {'max_gn_it', 'max_iters', 'start_iters'}:
                params[key] = int(config[reg_type][key])
            elif key == 'display_iterations':
                params[key] = config[reg_type].getboolean(key)
            else:
                params[key] = float(config[reg_type][key])
    return params
