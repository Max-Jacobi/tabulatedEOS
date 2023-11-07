import os
from abc import ABC, abstractmethod
from typing import Callable, Optional, TYPE_CHECKING, List, Dict, Tuple
from functools import reduce
from warnings import warn
import numpy as np
from h5py import File  # type: ignore
import alpyne.uniform_interpolation as ui  # type: ignore

from .Utils import RUnits

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike


class TabulatedEOS(ABC):
    """
    EOS abstract base class for interpolation in tabulated equation of state tables.False
    To define a new format the following functions have to be set:
    set_path:
        Sets the path to hdf5 files.
        This has to overwrite self.name to something different than "Unitilialized"
    __post_init__:
        This function is called after the constructor and should set the following variables:
        self.ye_key: str
            Name of the ye key in the hdf5 file
        self.temp_key: str
            Name of the temp key in the hdf5 file
        self.rho_key: str
            Name of the rho key in the hdf5 file
    get_key:
        This function should return the data for the given key.
    """

    data: Dict[str, 'NDArray[np.float_]']
    name: str
    ye_key: str
    temp_key: str
    rho_key: str

    def __init__(self, path: Optional[str] = None):
        # Interpolation table setup for uniform interpolation
        self._table: Optional[List['NDArray[np.float_]']] = None
        self._table_cold: Optional[List['NDArray[np.float_]']] = None

        # data to be interpolated, loaded from file when needed and cached
        self.data = {}

        # Ranges of interpolation input
        self._ye_range: Optional[Tuple[float, float]] = None
        self._temp_range: Optional[Tuple[float, float]] = None
        self._rho_range: Optional[Tuple[float, float]] = None

        self.name = "Unitilialized"
        self.set_path(path)

        self.__post_init__()

        for key in "ye temp rho".split():
            if not hasattr(self, f"{key}_key"):
                raise ValueError(
                    f"__post_init__ has to initialize self.{key}_key"
                )

    @abstractmethod
    def __post_init__(self):
        """
        This has to set
         - self.ye_key
         - self.temp_key
         - self.rho_key
        """
        ...

    @abstractmethod
    def set_path(self, path: Optional[str]) -> None:
        """
        Sets the path to hdf5 files.
        This has to overwrite self.name to something different than "Unitilialized"
        """
        ...

    @abstractmethod
    def get_key(self, key: str) -> "NDArray[np.float_]":
        """
        Returns the data for the given key.
        """
        ...

    @property
    def table(self) -> List['NDArray[np.float_]']:
        self._check_initialized()
        if self._table is None:
            Ye = self.get_key(self.ye_key)
            ltemp = np.log10(self.get_key(self.temp_key))
            lrho = np.log10(self.get_key(self.rho_key))

            if len(Ye) > 1:
                iye = 1/(Ye[1]-Ye[0])
            else:
                iye = np.nan
            if len(ltemp) > 1:
                iltemp = 1/(ltemp[1]-ltemp[0])
            else:
                iltemp = np.nan
            if len(lrho) > 1:
                ilrho = 1/(lrho[1]-lrho[0])
            else:
                ilrho = np.nan
            self._table = [np.array([Ye[0], ltemp[0], lrho[0]]),
                           np.array([iye, iltemp, ilrho])]
        return self._table

    def get_caller(self,
                   keys: List[str],
                   func: Callable = lambda *args, **_: args[0]
                   ) -> Callable:
        def eos_caller(ye: 'NDArray[np.float_]',
                       temp: 'NDArray[np.float_]',
                       rho: 'NDArray[np.float_]',
                       *_, **kw) -> 'NDArray[np.float_]':

            nonlocal keys
            nonlocal func

            self._get_keys(keys)
            scalars = [kk for kk in keys if np.isscalar(self.data[kk])]

            shape = ye.shape
            fshape = (np.prod(shape), )

            for aa, name, (rl, ru) in zip([ye, temp, rho],
                                          'ye temp rho'.split(),
                                          [self.ye_range, self.temp_range, self.rho_range]):

                if isinstance(aa, np.ndarray):
                    aa = aa.copy()
                    aa[(lmask := aa < rl)] = rl
                    aa[(rmask := aa > ru)] = ru
                    if np.any(lmask):
                        warn(f"{name} below EOS range")
                    if np.any(rmask):
                        warn(f"{name} above EOS range")

            args = [ye.flatten(), np.log10(temp).flatten(),
                    np.log10(rho).flatten()]
            mask = reduce(np.logical_and, [np.isfinite(arg) for arg in args])
            args = [arg[mask] for arg in args]

            data = np.array([self.data[kk] for kk in keys])
            islog = np.array([np.all(dd > 0) for dd in data])
            data[islog] = np.log10(data[islog])

            res = ui.linterp3D(*args, *self.table, data)

            args = []
            i_int = 0
            for kk in keys:
                if kk in scalars:
                    tmp = self.data[kk]
                else:
                    tmp = np.zeros(fshape)*np.nan
                    tmp[mask] = 10**res[i_int] if islog[i_int] else res[i_int]
                    tmp = np.reshape(tmp, shape)
                    i_int += 1
                args.append(tmp)

            return func(*args, rho=rho, ye=ye, temp=temp, **kw)

        return eos_caller

    def get_cold_caller(self,
                        keys: List[str],
                        func: Callable = lambda *args, **_: args[0]
                        ) -> Callable:
        def eos_caller_cold(ye: 'NDArray[np.float_]',
                            rho: 'NDArray[np.float_]',
                            *_, **kw) -> 'NDArray[np.float_]':

            nonlocal keys
            nonlocal func

            self._get_keys(keys)
            scalars = [kk for kk in keys if np.isscalar(self.data[kk])]

            shape = ye.shape
            fshape = (np.prod(shape), )

            args = [ye.flatten(), np.log10(rho).flatten()]
            mask = reduce(np.logical_and, [np.isfinite(arg) for arg in args])
            args = [arg[mask] for arg in args]

            data = np.array([self.data[kk][:, 0] for kk in keys
                             if kk not in scalars])
            islog = np.array([np.all(dd > 0) for dd in data])
            data[islog] = np.log10(data[islog])

            res = ui.linterp2D(*args,
                               self.table[0][[0, 2]],
                               self.table[1][[0, 2]],
                               data)
            args = []
            i_int = 0
            for kk in keys:
                if kk in scalars:
                    tmp = self.data[kk]
                else:
                    tmp = np.zeros(fshape)*np.nan
                    tmp[mask] = 10**res[i_int] if islog[i_int] else res[i_int]
                    tmp = np.reshape(tmp, shape)
                    i_int += 1
                args.append(tmp)

            return func(*args, rho=rho, ye=ye, **kw)

        return eos_caller_cold

    def get_inf_caller(self,
                       keys: List[str],
                       func: Callable = lambda *args, **_: args[0]
                       ) -> Callable:
        def eos_caller_inf(ye: 'NDArray[np.float_]',
                           *_, **kw) -> 'NDArray[np.float_]':

            nonlocal keys
            nonlocal func

            self._get_keys(keys)
            scalars = [kk for kk in keys if np.isscalar(self.data[kk])]

            shape = ye.shape
            fshape = (np.prod(shape), )

            ye = ye.flatten()
            mask = np.isfinite(ye)
            ye = ye[mask]

            data = np.array([self.data[kk][:, 0, 0] for kk in keys
                            if kk not in scalars])
            islog = np.array([np.all(dd > 0) for dd in data])
            data[islog] = np.log10(data[islog])

            res = ui.linterp1D(ye,
                               self.table[0][0],
                               self.table[1][0],
                               data)

            args = []
            i_int = 0
            for kk in keys:
                if kk in scalars:
                    tmp = self.data[kk]
                else:
                    tmp = np.zeros(fshape)*np.nan
                    tmp[mask] = 10**res[i_int] if islog[i_int] else res[i_int]
                    tmp = np.reshape(tmp, shape)
                    i_int += 1
                args.append(tmp)

            return func(*args, ye=ye, **kw)

        return eos_caller_inf

    def get_weak_eq_caller(self,
                           keys: List[str],
                           func: Callable = lambda *args, **_: args[0]
                           ) -> Callable:
        def eos_caller(temp: 'NDArray[np.float_]',
                       rho: 'NDArray[np.float_]',
                       *_, **kw) -> 'NDArray[np.float_]':

            nonlocal keys
            nonlocal func

            self._get_keys(keys)
            scalars = [kk for kk in keys if np.isscalar(self.data[kk])]

            shape = rho.shape
            fshape = (np.prod(shape), )

            args = [np.log10(temp).flatten(),
                    np.log10(rho).flatten()]
            mask = reduce(np.logical_and, [np.isfinite(arg) for arg in args])
            args = [arg[mask] for arg in args]

            data = np.array([self.data[kk] for kk in keys])
            islog = np.array([np.all(dd > 0) for dd in data])
            data[islog] = np.log10(data[islog])

            res = ui.linterp3D(*args, *self.table, data)

            func_args = []
            i_int = 0
            for kk in keys:
                if kk in scalars:
                    tmp = self.data[kk]
                else:
                    tmp = np.zeros(fshape)*np.nan
                    tmp[mask] = 10**res[i_int] if islog[i_int] else res[i_int]
                    tmp = np.reshape(tmp, shape)
                    i_int += 1
                func_args.append(tmp)

            return func(*func_args, rho=rho, temp=temp, **kw)

        return eos_caller

    def _check_initialized(self):
        if self.name == 'Unitilialized':
            raise ValueError(F"Path to {self.__class__.__name__} file not given. "
                             F"Run {self.__class__.__name__}.set_path('path/to/eos/')")

    @property
    def ye_range(self) -> Tuple[np.float_, np.float_]:
        self._check_initialized()
        if self._ye_range is None:
            Ye = self.get_key(self.ye_key)
            self._ye_range = (Ye[0], Ye[-1])
        return self._ye_range

    @property
    def temp_range(self) -> Tuple[np.float_, np.float_]:
        self._check_initialized()
        if self._temp_range is None:
            Temp = self.get_key(self.temp_key)
            self._temp_range = (Temp[0], Temp[-1])
        return self._temp_range

    @property
    def rho_range(self) -> Tuple[np.float_, np.float_]:
        self._check_initialized()
        if self._rho_range is None:
            Rho = self.get_key(self.rho_key)
            self._rho_range = (Rho[0], Rho[-1])
        return self._rho_range

    def _get_keys(self, keys: List[str]):
        self._check_initialized()

        new_keys = [kk for kk in keys if kk not in self.data]

        if len(new_keys) > 0:
            for kk in new_keys:
                self.data[kk] = self.get_key(kk)

    def __call__(
        self,
        key: str,
        ye: "ArrayLike",
        temp: "ArrayLike",
        rho: "ArrayLike",
    ) -> "NDArray[np.float_]":
        if np.isscalar(ye):
            ye = np.array([ye])
            temp = np.array([temp])
            rho = np.array([rho])
            scalar = True
        else:
            ye = np.asarray(ye)
            temp = np.asarray(temp)
            rho = np.asarray(rho)
            scalar = False

        if ye.shape != temp.shape or ye.shape != rho.shape:
            raise ValueError("ye, rho, and temp must have the same shape")

        if scalar:
            return self.get_caller([key])(ye, temp, rho)[0]
        return self.get_caller([key])(ye, temp, rho)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with name {self.name}"


class PizzaEOS(TabulatedEOS):
    """Realistic Tabluated EOS """
    hydro_path: Optional[str]
    weak_path: Optional[str]

    def __post_init__(self):
        self.ye_key = "ye"
        self.temp_key = "temperature"
        self.rho_key = "density"

        # mass factor in MeV
        self._mass_fac: Optional[float] = None

    def set_path(self, path: Optional[str]):
        "EOS path setter"
        if path is None:
            self.hydro_path = None
            self.weak_path = None
            return

        cactus_base = (os.environ['CACTUS_BASEDIR']
                       if "CACTUS_BASEDIR" in os.environ else None)
        if path[0] != '/' and cactus_base is not None:
            path = f"{cactus_base}/EOSs/{path}"

        self.hydro_path = f"{path}/hydro.h5"
        self.weak_path = f"{path}/weak.h5"
        self.name = path.split('/')[-1]

    def get_key(self, key):
        """Get key from hydro or weak file"""
        if key in self.data:
            return self.data[key]

        self._check_initialized()

        scale = dict(
            rho=RUnits['Rho'],
            pressure=RUnits["Press"],
            only_P=RUnits["Press"],
            internalEnergy=RUnits["Eps"],
            only_E=RUnits["Eps"],
            density=RUnits["Rho"],
        )

        for path in [self.hydro_path, self.weak_path]:
            with File(path, 'r') as hfile:
                if key in hfile:
                    self.data[key] = np.array(hfile[key])
                    if key in scale:
                        self.data[key] *= scale[key]
                    break
        else:
            raise KeyError(f"{key} not found in EOS tables in {self}")
        return self.data[key]

    def get_mbary50(self,) -> float:
        self._check_initialized()
        with File(self.hydro_path, 'r') as hfile:
            mass_fac = hfile['mass_factor'][()]
        mev_50_msol = 8.962964431087716e-11
        return mass_fac*mev_50_msol
