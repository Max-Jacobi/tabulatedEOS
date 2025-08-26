import os
from functools import lru_cache
from typing import Optional

import numpy as np
from h5py import File, Dataset

from .EOS import TabulatedEOS
from .unit_system import Nuclear
from .unit_system import UnitSystem as US


class PyCompOSEEOS(TabulatedEOS):
    """Equation of state format for tables from pycompose"""
    path: str

    def post_init(self, path: str, **kwargs):
        """
        Parameters:

        path: str
            Path to the h5 file

        self.name is initialized to the name of the directory containing the h5 file.
        """

        self.table_keys = {
            'rho': 'rho',
            'ye': 'yq',
            'temp': 't',
            }
        self.eos_units = Nuclear

        self.conversions["rho"] = US.MassDensityConversion
        self.conversions["t"] = US.TemperatureConversion
        self.conversions["Q1"] = _conv_Q1
        self.conversions["Q2"] = US.EntropyConversion
        self.conversions["Q3"] = US.ChemicalPotentialConversion
        self.conversions["Q4"] = US.ChemicalPotentialConversion
        self.conversions["Q5"] = US.ChemicalPotentialConversion
        self.conversions["Q6"] = _conv_Q67
        self.conversions["Q7"] = _conv_Q67
        self.conversions["press"] = US.PressureConversion

        self.path = path
        with File(self.path, 'r') as hfile:
            self.mass_factor = float(hfile['mn'][()])

        self.name = os.path.basename(os.path.dirname(path))

    @lru_cache(maxsize=10)
    def get_key(self, key: str) -> np.ndarray:
        """
        returns the dataset < key > from the h5 file
        rho is converted from nb
        """

        self._check_initialized()

        if key == 'press':
            _key = 'Q1'
        elif key == 'eps':
            _key = 'Q7'
        elif key == 'entr':
            _key = 'Q2'
        elif key == 'rho':
            _key = 'nb'
        else:
            _key = key


        with File(self.path, 'r') as hfile:
            if _key in hfile:
                data = np.array(hfile[_key])
            else:
                raise KeyError(f"{_key} not found in {self}")

            if key == 'press':
                data *= np.array(hfile['nb'])[:, None, None]
            elif key == 'rho':
                data *= self.mass_factor

        return data

    @lru_cache
    def keys(self) -> list[str]:
        """
        returns the keys of the h5 file
        """
        with File(self.path, 'r') as hfile:
            return list(filter(
                lambda k: isinstance((dset := hfile[k]), Dataset) and len(dset.shape)==3,
                hfile.keys()
                ))




def _conv_Q1(*args: US) -> float:
    return US.PressureConversion(*args)/US.DensityConversion(*args)

def _conv_Q67(*args: US) -> float:
    return US.EnergyConversion(*args)/US.MassConversion(*args)
