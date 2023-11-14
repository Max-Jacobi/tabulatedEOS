from functools import lru_cache
import os
from typing import Optional

import numpy as np
from h5py import File

from .EOS import TabulatedEOS


class SROSNAEOS(TabulatedEOS):
    """EOS table produced by the SNA code of SROEOS."""
    path: str
    mass_factor: float = 1674927470925588.0

    def post_init(self, path: str, mass_factor: Optional[float] = None):
        """
        Parameters:

        path: str
            Path to the h5 file
        mass_factor: float
            Mass factor to scale the density (default: value used in MERGE SRO code)


        self.name is initialized to the name of the directory containing the h5 file.
        """

        self.table_keys = {'ye': 'ye',
                           'temp': 'temp',
                           'rho': 'rho',
                           }

        # mass factor in MeV
        if mass_factor is not None:
            self.mass_factor = mass_factor

        self.path = path
        self.name = os.path.basename(os.path.dirname(path))

    @lru_cache(maxsize=10)
    def get_key(self, key):
        """
        returns the dataset <key> from the h5 file
        keys without log can be used, i.e. temp returns 10^logtemp
        rho is converted from n
        """

        units = {}

        self._check_initialized()

        with File(self.path, 'r') as hfile:
            if key == 'rho':
                data = np.array(hfile['logn'])
                data = 10 ** data * self.mass_factor
            elif key in hfile:
                data = np.array(hfile[key])
            elif f'log{key}' in hfile:
                data = np.array(hfile[f'log{key}'])
                data = 10**data
            else:
                raise KeyError(f"{key} not found in {self}")

        if key in units:
            data *= units[key]
        return data

    def keys(self) -> list[str]:
        """
        returns the keys of the h5 file
        """
        with File(self.path, 'r') as hfile:
            return list(hfile.keys())
