import os
from typing import Optional
from dataclasses import replace
from functools import cached_property, lru_cache
import numpy as np
from h5py import File

from .EOS import TabulatedEOS
from .unit_system import CGS
from .unit_system import UnitSystem as US

def _conv_eps(*args: US) -> float:
    return US.EnergyConversion(*args)/US.MassConversion(*args)

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

        self.conversions["rho"] = US.MassDensityConversion
        self.conversions["density"] = US.MassDensityConversion
        self.conversions["temp"] = US.TemperatureConversion
        self.conversions["temperature"] = US.TemperatureConversion
        self.conversions["internalEnergy"] = _conv_eps
        self.conversions["entropy"] = US.EntropyConversion
        self.conversions["pressure"] = US.PressureConversion

        # mass factor in MeV
        self._mass_fac: Optional[float] = None

        cactus_base = (os.environ['CACTUS_BASEDIR']
                       if "CACTUS_BASEDIR" in os.environ else None)
        if path[0] != '/' and cactus_base is not None:
            path = f"{cactus_base}/EOSs/{path}"

        self.hydro_path = f"{path}/hydro.h5"
        self.weak_path = f"{path}/weak.h5"
        self.name = path.split('/')[-1]
        self.eos_units = replace(CGS, temperature=CGS.kb/CGS.MeV, kb=1.0)

    @lru_cache(maxsize=10)
    def get_key(self, key):
        """returns the dataset <key> from the hydro.h5 or weak.h5 file"""

        self._check_initialized()

        for path in [self.hydro_path, self.weak_path]:
            with File(path, 'r') as hfile:
                if key in hfile:
                    data = np.array(hfile[key])
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
