from setuptools import find_packages, setup
import os
with open(os.path.abspath(os.path.join(__file__, '../README.rst')), encoding='utf-8') as f:
    long_description = f.read()
setup(name='pytorch-igniter',
      version='0.0.32',
      author='Ben Striner',
      author_email="bstriner@gmail.com",
      url='https://github.com/bstriner/pytorch-igniter',
      description="Simplify running pytorch training with fully-configured pytorch-ignite",
      install_requires=[
          'pytorch-ignite',
          'pyyaml',
          'tqdm',
          'mlflow',
          'aws-sagemaker-remote'
      ],
      packages=find_packages(),
      long_description=long_description,
      long_description_content_type='text/x-rst')
