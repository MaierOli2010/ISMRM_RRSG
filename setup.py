from setuptools import setup

setup(name='rrsg_cgreco',
      version='0.1',
      description='ISMRM RRSG Paper Initiative',
      url='https://github.com/MaierOli2010/ISMRM_RRSG',
      author='Oliver Maier',
      author_email='oliver.maier@tugraz.at',
      license='Apache-2.0',
      packages=['rrsg_cgreco'],
      install_requires=[
        'pyopencl',
        'numpy',
        'h5py',
	'mako',
	'matplotlib'],
      zip_safe=False) 
