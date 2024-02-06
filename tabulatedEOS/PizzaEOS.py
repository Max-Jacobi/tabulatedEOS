import os
from typing import Optional
from functools import cached_property, lru_cache
import numpy as np
from collections import namedtuple
from h5py import File

from .EOS import TabulatedEOS


class PizzaEOS(TabulatedEOS):
    """Realistic Tabluated EOS """
    hydro_path: Optional[str]
    weak_path: Optional[str]

    def post_init(self, path: str):
        """
        Path should be the directory containing the hydro and weak files
        self.name is initialized to the directory name.
        If no absolute path is given, and the CACTUS_BASEDIR environment variable is set, the path is
        assumed to be relative to '$CACTUS_BASEDIR/EOSs/'.
        """

        self.table_keys = {'ye': 'ye',
                           'temp': 'temperature',
                           'rho': 'density',
                           }

        # mass factor in MeV
        self._mass_fac: Optional[float] = None

        cactus_base = (os.environ['CACTUS_BASEDIR']
                       if "CACTUS_BASEDIR" in os.environ else None)
        if path[0] != '/' and cactus_base is not None:
            path = f"{cactus_base}/EOSs/{path}"

        self.hydro_path = f"{path}/hydro.h5"
        self.weak_path = f"{path}/weak.h5"
        self.name = path.split('/')[-1]

    @lru_cache(maxsize=10)
    def get_key(self, key):
        """returns the dataset <key> from the hydro.h5 or weak.h5 file"""

        self._check_initialized()

        units = dict(
            rho=RUnits['Rho'],
            pressure=RUnits["Press"],
            internalEnergy=RUnits["Eps"],
            density=RUnits["Rho"],
        )

        for path in [self.hydro_path, self.weak_path]:
            with File(path, 'r') as hfile:
                if key in hfile:
                    data = np.array(hfile[key])
                    if key in units:
                        data *= units[key]
                    break
        else:
            raise KeyError(f"{key} not found in EOS tables in {self}")
        return data

    def keys(self):
        """returns the keys in the hydro.h5 file"""
        self._check_initialized()
        with File(self.hydro_path, 'r') as hfile:
            keys = list(hfile.keys())
        with File(self.weak_path, 'r') as hfile:
            keys.extend(list(hfile.keys()))
        return keys

    @cached_property
    def mbary50(self,) -> float:
        mass_fac = float(self.get_key("mass_fac"))
        mev_50_msol = 8.962964431087716e-11
        return mass_fac*mev_50_msol


# Unit conversion from Pizza to geometric units in solar mass
Units = {
    "Rho": 6.175828477586656e+17,  # g/cm^3
    "Eps":  8.9875517873681764e+20,  # erg/g
    "Press":  5.550725674743868e+38,  # erg/cm^3
    "Mass": 1.988409870967742e+33,  # g
    "Energy": 1.7870936689836656e+3,  # 50 erg
    "Time": 0.004925490948309319,  # ms
    "Length":  1.4766250382504018,  # km
}
RUnits = {
    "Rho":  1.6192159539877191e-18,
    "Press":  1.8015662430410847e-39,
    "Eps":  1.1126500560536184e-21,
    "Mass":  5.028992139685286e-34,
    "Energy":  5.595508386114039e-55,
    "Time":  203.02544670054692,
    "Length":  0.6772199943086858,
}

# create a class from the dictionary of units for easy access
_lower_U = {k.lower(): v for k, v in Units.items()}
_lower_RU = {k.lower(): v for k, v in RUnits.items()}
U = namedtuple('Units', _lower_U.keys())(**_lower_U)
RU = namedtuple('RUnits', _lower_RU.keys())(**_lower_RU)
