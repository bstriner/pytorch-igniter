from setuptools import find_packages, setup

setup(name='pytorch-igniter',
      version='0.0.1',
      author='Ben Striner',
      url='https://github.com/bstriner/pytorch-igniter',
      install_requires=[
          'pytorch-ignite',
          'pyyaml',
          'mlflow'
      ],
      packages=find_packages())
