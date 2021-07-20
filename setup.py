#!/usr/bin/env python
# Install script for labdata tools

#  wfield - tools to transfer and copy data in the lab
# Copyright (C) 2021 Joao Couto - jpcouto@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup
from setuptools.command.install import install
import os
from os.path import join as pjoin

labdata_dir = pjoin(os.path.expanduser('~'),'.labdatatools')
if not os.path.isdir(labdata_dir):
    print('Creating {0}'.format(labdata_dir))
    os.makedirs(labdata_dir)


description = '''Utilities for copying and managing lab data'''

setup(
    name = 'labdatatools',
    version = '0.1',
    author = 'Joao Couto',
    author_email = 'jpcouto@gmail.com',
    description = (description),
    long_description = description,
    long_description_content_type='text/markdown',
    license = 'GPL',
    install_requires = requirements,
    url = "https://github.com/jpcouto/labdatatools",
    packages = ['labdatatools'],
    #entry_points = {
    #    'console_scripts': [
    #        'wfield = wfield.cli:main',
    #        'wfield-ncaas = wfield.ncaas_gui:main',
    #    ]
    #},
)
