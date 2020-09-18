from setuptools import find_packages, setup

setup(name='pytorch-igniter',
      version='0.0.10',
      author='Ben Striner',
      author_email="bstriner@gmail.com",
      url='https://github.com/bstriner/pytorch-igniter',
      description="Simplify running pytorch training with fully-configured pytorch-ignite",
      install_requires=[
          'pytorch-ignite',
          'pyyaml',
          'tqdm'
      ],
      extras_require={
          "mlflow":  ["mlflow"],
          "sagemaker": ["sagemaker"],
      },
      packages=find_packages())
