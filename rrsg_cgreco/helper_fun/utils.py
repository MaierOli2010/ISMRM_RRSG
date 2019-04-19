#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:10:07 2019

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
import configparser
from transforms.pyopencl_nufft import PyOpenCLNUFFT

DTYPE = np.complex64
DTYPE_real = np.float32

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
    config['DEFAULT']["tol"] = '1e-30'
    config['DEFAULT']["lambd"] = '5e-1'
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
