import os
from functools import lru_cache
from typing import Optional

import numpy as np
from h5py import File

from .EOS import TabulatedEOS


class StellarCollapseEOS(TabulatedEOS):
    """Equation of state format for tables from stellarcollapse.org"""
    path: str

    def post_init(self, path: str):
        """
        Parameters:

        path: str
            Path to the h5 file

        self.name is initialized to the name of the directory containing the h5 file.
        """

        self.table_keys = {'ye': 'ye',
                           'temp': 'temp',
                           'rho': 'rho',
                           }

        self.path = path
        self.name = os.path.basename(os.path.dirname(path))

    @lru_cache(maxsize=10)
    def get_key(self, key):
        """
        returns the dataset < key > from the h5 file
        keys without log can be used, i.e. temp returns 10 ^ logtemp
        rho is converted from n
        """

        units = {}

        self._check_initialized()

        with File(self.path, 'r') as hfile:
            if key in hfile:
                data = np.array(hfile[key])
            elif f'log{key}' in hfile:
                data = np.array(hfile[f'log{key}'])
                data = 10**data
            else:
                raise KeyError(f"{key} not found in {self}")

            if key == 'energy':
                data -= np.array(hfile['energy_shift'])

        if key in units:
            data *= units[key]
        return data

    def keys(self) -> list[str]:
        """
        returns the keys of the h5 file
        """
        with File(self.path, 'r') as hfile:
            return list(hfile.keys())
