from distutils.core import setup
from setuptools import find_packages
import os

setup_path = os.path.abspath(os.path.dirname(__file__))

setup(name='slpt',
      version='1.0',

      python_requires='>=3',
      packages=['slpt', 'slpt.SLPT', 'slpt.config', 'slpt.backbone', 'slpt.facedetector', 'slpt.utils'],
      include_package_data=True,
      )
