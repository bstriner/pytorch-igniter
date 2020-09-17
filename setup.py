from setuptools import find_packages, setup

setup(name='pytorch-igniter',
      version='0.0.9',
      author='Ben Striner',
      url='https://github.com/bstriner/pytorch-igniter',
      install_requires=[
          'pytorch-ignite',
          'pyyaml'
      ],
      extras_require={
          "mlflow":  ["mlflow"],
          "sagemaker": ["sagemaker"],
      },
      packages=find_packages())

# python setup.py bdist_wheel sdist && twine upload dist\*
