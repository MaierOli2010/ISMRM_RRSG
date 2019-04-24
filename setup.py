from setuptools import setup

setup(name='rrsg_cgreco',
      version='0.1',
      description='ISMRM RRSG Paper Initiative',
      url='https://github.com/MaierOli2010/ISMRM_RRSG',
      author='Oliver Maier',
      author_email='oliver.maier@tugraz.at',
      license='Apache-2.0',
      packages=setuptools.find_packages(),
      setup_requires=["cython"],
      install_requires=[
        'cython',
        'pyopencl',
        'numpy',
        'h5py',
        'mako',
        'matplotlib',
        'gpyfft @ git+https://github.com/geggo/gpyfft.git#egg=gpyfft'],
      entry_points={
        'console_scripts': ['rrsg_cgreco = rrsg_cgreco._Init_CG:run',
                            'rrsg_plotresults = rrsg_cgreco._plot_results:run'],
        },
        package_data={'rrsg_cgreco':['kernels/*.c']},
        include_package_data=True,
      zip_safe=False) 
