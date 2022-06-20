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
import sys
from os.path import join as pjoin
import shutil

            
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
    install_requires = [],
    url = "https://github.com/jpcouto/labdatatools",
    packages = ['labdatatools'],
    entry_points = {
        'console_scripts': [
            'labdata = labdatatools.cli:main',
        ]
    },
)

if 'install' in sys.argv or 'develop' in sys.argv:
    from labdatatools.utils import labdata_preferences
    plugins = labdata_preferences['plugins_folder']

    # if the config directory tree doesn't exist, create it
    if not os.path.exists(plugins):
        os.makedirs(plugins)

    # copy every file from given location to the specified ``CONFIG_PATH``
    for fname in os.listdir('analysis'):
        fpath = os.path.join('analysis', fname)
        if not os.path.exists(pjoin(plugins,fname)):
            shutil.copy(fpath, plugins)
        else:
            print('File already exists.')
