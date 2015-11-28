#!/usr/bin/env python

from os.path import exists
from setuptools import setup
import ctd2xray

setup(name='ctd2xray',
      version=ctd2xray.__version__,
      description='Python utilities for hydrographic data',
      url='http://github.com/rabernat/ctd2xray',
      maintainer='Ryan Abernathey',
      maintainer_email='rpa@ldeo.columbia.edu',
      license='MIT',
      #keywords='task-scheduling parallelism',
      packages=['ctd2xray'],
      long_description=(open('README.md').read() if exists('README.md')
                        else ''),
      install_requires=['numpy','scipy','xray'],
      zip_safe=False)
