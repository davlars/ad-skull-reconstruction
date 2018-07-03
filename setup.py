# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Setup script for adutils.

Installation command::

    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import
from setuptools import setup
import os


root_path = os.path.dirname(__file__)


setup(
    name='adutils',
    version='0.1',
    description='ad-skull-reconstruction',
    url='https://github.com/davlars/ad-skull-reconstruction',
    package_dir={'adutils': 'adutils'},
    install_requires=['nibabel',
                      'tqdm',
                      'matplotlib',
                      'astra-toolbox',
                      'odl>=0.6.1.dev0']
)
