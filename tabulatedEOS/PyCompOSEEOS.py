import os
from functools import lru_cache, cached_property
from typing import TYPE_CHECKING

import numpy as np
from h5py import File, Dataset

from .EOS import TabulatedEOS
from .unit_system import Nuclear
from .unit_system import UnitSystem as US

if TYPE_CHECKING:
    from .EOS import Array1D, Array3D



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
        self.conversions["Q1"] = _conv_Q1                        # P/n_B
        self.conversions["Q2"] = US.EntropyConversion            # S/n_B
        self.conversions["Q3"] = US.ChemicalPotentialConversion  # mu_B
        self.conversions["Q4"] = US.ChemicalPotentialConversion  # mu_Q
        self.conversions["Q5"] = US.ChemicalPotentialConversion  # mu_L
        self.conversions["Q6"] = _conv_Q67                       # h = (e + P)/(m_B*n_B)
        self.conversions["Q7"] = _conv_Q67                       # e/(m_B*n_B) - 1
        self.conversions["press"] = US.PressureConversion

        self.path = path
        with File(self.path, 'r') as hfile:
            self.mass_factor = float(hfile['mn'][()])

        self.name = os.path.basename(os.path.dirname(path))

    @lru_cache(maxsize=10)
    def get_key(self, key: str) -> "Array1D | Array3D":
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

    @cached_property
    def hinf(self) -> float:
        """returns the minimum enthalpy in the table"""
        return np.min(self.get_key_with_units('Q6'))


def _conv_Q1(*args: US) -> float:
    return US.PressureConversion(*args)/US.DensityConversion(*args)

def _conv_Q67(*args: US) -> float:
    return US.EnergyConversion(*args)/US.MassConversion(*args)
