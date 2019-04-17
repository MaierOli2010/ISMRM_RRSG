ISMRM_RRSG
===================================

* Requires [PyOpenCL](https://github.com/inducer/pyopencl) package
* Requires [GPyFFT](https://github.com/geggo/gpyfft) package 
* Requires [bart](https://github.com/mrirecon/bart)

Currently runs only on GPUs due to a limitation in the GPyFFT package.

Installing dependencies:
---------------
First make sure that you have a working OpenCL installation
  - OpenCL is usually shipped with GPU driver (Nvidia/AMD)
  - Install the ocl_icd and the OpenCL-Headers
  ```
    apt-get install ocl_icd* opencl-headers
  ```  
Possible restart of system after installing new drivers
  - Build [clinfo](https://github.com/Oblomov/clinfo)
  - Run clinfo in terminal and check for errors
  - Build or download binarys of [clFFT](https://github.com/clMathLibraries/clFFT)
    - Please refer to the [clFFT](https://github.com/clMathLibraries/clFFT) docs regarding building
    - If build from source symlink clfft libraries from lib64 to the lib folder and run ``` ldconfig ```
  - Build GPyFFT](https://github.com/geggo/gpyfft) 
    ```
    python setup.py build_ext bdist_wheel
    pip install ./dist/YOUR-WHEEL-NAME.whl
    ```
  - install pyfftw
  ```
    apt-get install pyfftw
  ```    
  - Finally install [PyOpenCL](https://github.com/inducer/pyopencl)
    ```
    pip install pyopencl
    ```
Please refer to the documentaiton of [bart](https://github.com/mrirecon/bart) for a detailed explanation on how to set up the toolbox.


Running the recosntruction:
-------------------------
Navigate to the root folder of ISMRM_RRSG and simply type:
```
./run_acc
```
to run the reconstruction for brain and heart data with increasing accelaration.
After reconstruction is finished, the required plots will be automatically generated and saved in the root folder.


Regularization can be changed or turned off by changing the value of ```lambd``` in ```default.ini```. The .ini file will be automatically generated the first time the code is run. The ```tol``` parameter can be used to change the desired toleranze of the optimization scheme. ```max_iters``` defines the maximum number of CG iterations.
