from abc import ABC, abstractmethod
from typing import Callable, Optional, TYPE_CHECKING, Any, overload
from functools import reduce, cached_property
from warnings import warn
import numpy as np
import alpyne.uniform_interpolation as ui  # type: ignore

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    Mask = NDArray[np.bool_]


class TabulatedEOS(ABC):
    """
    EOS abstract base class for interpolation in tabulated equation of state tables.False
    To define a new format the following functions have to be set:
    post_init(path):
        Called after at initialization and should set the following variables:
          - self.table_keys: dict[str, str]
                Dictionary with the keys for electron fraction, temperature and density
                Keys should be 'ye', 'temp' and 'rho'
                The order has to be the the same in which they are needed for interpolation
          - self.name: str
              should be set to something different than "Unitilialized"
          - should also initialize variables based on path that will be used by get_keys
    get_key:
        This function should return the data for the given key.
    """

    name: str
    table_keys: dict[str, str]
    log_names: list[str] = ['rho', 'temp']

    def __init__(self, path: Optional[str] = None, **kwargs: Any) -> None:

        self.name = "Unitilialized"
        if path is None:
            return
        self.post_init(path, **kwargs)

        if not hasattr(self, "table_keys"):
            raise ValueError("post_init has to initialize self.table_keys")
        if self.name == "Unitilialized":
            raise ValueError("post_init has to initialize self.name")

        self._read_table()

    @abstractmethod
    def post_init(self, path: str, **kwargs):
        """
        The function should set the following variables:
          - self.table_keys: dict[str, str]
                Dictionary with the keys for electron fraction, temperature and density
                Keys should be 'ye', 'temp' and 'rho'
                The order has to be the the same in which they are needed for interpolation
          - self.name: str
              should be set to something different than "Unitilialized"

        Should also initialize the variables to needed by get_key based on input path and other keyword arguments.
        """
        ...

    @abstractmethod
    def get_key(self, key: str) -> "NDArray[np.float_]":
        """
        Returns the data for the given key.
        This will be wrapped by functools.lru_cache to speed up interpolation.
        """
        ...

    def _read_table(self) -> None:
        self._offsets = {}
        self._inv_steps = {}

        for name, key in self.table_keys.items():
            dat = self.get_key(key)
            if name in self.log_names:
                dat = np.log10(dat)
            self._offsets[name] = dat[0]
            if len(dat) > 1:
                self._inv_steps[name] = 1/(dat[1]-dat[0])
            else:
                self._inv_steps[name] = None

    def _get_caller(self,
                    arguments: list[str],
                    keys: list[str],
                    data_slice: Optional[dict[str, int]] = None,
                    func: Optional[Callable] = None,
                    ) -> Callable:
        self._check_initialized()

        if func is None:
            func = lambda *args, **_: args[0]
        if data_slice is None:
            data_slice = {}

        def eos_caller(
            *args: 'NDArray[np.float_]',
            **kwargs: 'NDArray[np.float_]',
        ) -> 'NDArray[np.float_]':

            nonlocal arguments
            nonlocal data_slice
            nonlocal keys
            nonlocal func

            kwargs = self._convert_args_to_kwargs(args, kwargs, arguments)
            inputs, kwargs = self._separate_inputs(kwargs, arguments)
            inputs, shape = self._check_inputs(inputs)

            inps, finite_mask = self._prepare_inputs(inputs)
            offsets, inv_steps = self._prepare_table(arguments)

            data = self._slice_data(keys, data_slice)
            data, islog = self._logspace_data(data)

            if len(inps) == 1:
                offsets = np.float64(offsets)
                inv_steps = np.float64(inv_steps)
                result = ui.linterp1D(inps[0], offsets, inv_steps, data)
            elif len(inps) == 2:
                result = ui.linterp2D(
                    inps[0], inps[1], offsets, inv_steps, data)
            elif len(inps) == 3:
                result = ui.linterp3D(
                    inps[0], inps[1], inps[2], offsets, inv_steps, data)
            else:
                raise ValueError("Too many arguments")

            result = self._reshape_result(result, shape, finite_mask, islog)

            return func(*result, **inputs, **kwargs)

        return eos_caller

    def get_caller(self, keys: list[str], func: Optional[Callable] = None,) -> Callable:
        return self._get_caller(
            arguments=['ye', 'temp', 'rho'],
            keys=keys,
            func=func
        )

    def get_cold_caller(self, keys: list[str], func: Optional[Callable] = None,) -> Callable:
        return self._get_caller(
            arguments=['ye', 'rho'],
            data_slice={'temp': 0},
            keys=keys,
            func=func
        )

    def get_inf_caller(self, keys: list[str], func: Optional[Callable] = None,) -> Callable:
        return self._get_caller(
            arguments=['ye'],
            data_slice={"temp": 0, "rho": 0},
            keys=keys,
            func=func
        )

    def _check_initialized(self):
        if self.name == 'Unitilialized':
            raise ValueError(F"Path to {self.__class__.__name__} file not given. "
                             F"Run {self.__class__.__name__}.set_path('path/to/eos/')")

    @cached_property
    def range(self) -> dict[str, tuple[float, float]]:
        self._check_initialized()
        table_range = {}
        for name, key in self.table_keys.items():
            dat = self.get_key(key)
            table_range[name] = (dat[0], dat[-1])
        return table_range

    @overload
    def __call__(
        self,
        key: str,
        ye: 'ArrayLike',
        temp: 'ArrayLike',
        rho: 'ArrayLike',
    ) -> "NDArray[np.float_]":
        ...

    @overload
    def __call__(
        self,
        key: str,
        ye: float,
        temp: float,
        rho: float,
    ) -> float:
        ...

    def __call__(self, key, ye, temp, rho):
        if np.isscalar(ye):
            ye = np.array([ye])
            temp = np.array([temp])
            rho = np.array([rho])
            return self.get_caller([key])(ye=ye, temp=temp, rho=rho)[0]

        ye = np.asarray(ye)
        temp = np.asarray(temp)
        rho = np.asarray(rho)
        return self.get_caller([key])(ye=ye, temp=temp, rho=rho)

    def __getitem__(self, key: str) -> "NDArray[np.float_]":
        return self.get_key(key)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with name {self.name}"

    def _check_inputs(
        self,
        inputs: "dict[str, NDArray[np.float_]]",
    ) -> tuple["dict[str, NDArray[np.float_]]", tuple[int, ...]]:
        inp_shape = iter(inputs.values()).__next__().shape
        clipped_inputs = {}
        for name, inp in inputs.items():
            if inp.shape != inp_shape:
                raise ValueError("Inputs have to have the same shape")
            for key, val_range in self.range.items():
                if name == key:
                    clipped = np.clip(inp, *val_range)
                    if np.any(clipped != inp):
                        outliers = inp[clipped != inp]
                        warn(
                            f"{name} = {outliers} out of range {val_range} of EOS {self.name}")
                    clipped_inputs[name] = clipped
                    break
            else:
                raise ValueError(f"Invalid argument {name}")
        return clipped_inputs, inp_shape

    def _prepare_inputs(
        self,
        inputs: dict[str, "NDArray[np.float_]"],
    ) -> tuple[list["NDArray[np.float_]"], "Mask"]:
        c_inputs = inputs.copy()
        for name in inputs:
            if name in self.log_names:
                c_inputs[name] = np.log10(inputs[name])

        args = [c_inputs[name]
                for name, key in self.table_keys.items()
                if name in c_inputs]

        finite_mask = reduce(
            np.logical_and,
            [np.isfinite(arg) for arg in args],
        )
        args = [arg[finite_mask] for arg in args]
        return args, finite_mask

    def _prepare_table(self, arguments: list[str]):
        offsets = np.array([self._offsets[name] for name in arguments])
        inv_steps = np.array([self._inv_steps[name] for name in arguments])
        return offsets, inv_steps

    def _slice_data(
        self,
        keys: list[str],
        data_slice: dict[str, int],
    ) -> "NDArray[np.float_]":

        data_slice = dict(sorted(data_slice.items(), key=lambda it: it[1]))

        def slice_data(data):
            for ii, tk in list(enumerate(self.table_keys))[::-1]:
                if tk in data_slice:
                    data = data.take(data_slice[tk], axis=ii)
            return data

        data = np.array([slice_data(self.get_key(kk)) for kk in keys])
        return data

    def _logspace_data(
        self,
        data: "NDArray[np.float_]",
    ) -> tuple["NDArray[np.float_]", "Mask"]:
        islog = np.array([np.all(dd > 0) for dd in data])
        data[islog] = np.log10(data[islog])
        return data, islog

    @staticmethod
    def _convert_args_to_kwargs(
        args: tuple["NDArray[np.float_]", ...],
        kwargs: dict[str, Any],
        arguments: list[str],
    ) -> dict[str, "NDArray[np.float_]"]:
        kwargs = kwargs.copy()
        args_list = list(args)
        for argument in arguments:
            if argument in kwargs:
                continue
            kwargs[argument] = args_list.pop(0)
        if args_list:
            raise ValueError("Too many arguments")
        return kwargs

    @staticmethod
    def _separate_inputs(
        kwargs: dict[str, Any],
        arguments: list[str],
    ) -> tuple[dict[str, "NDArray[np.float_]"], dict[str, Any]]:
        inputs = {key: kwargs.pop(key)
                  for key in arguments
                  if key in kwargs}

        return inputs, kwargs

    @staticmethod
    def _reshape_result(
        result: "NDArray[np.float_]",
        shape: tuple[int, ...],
        finite_mask: "Mask",
        islog: "Mask",
    ) -> list["NDArray[np.float_]"]:
        reshaped = []
        for log, res in zip(islog, result):
            tmp = np.zeros(shape)*np.nan
            if log:
                tmp[finite_mask] = 10**res
            else:
                tmp[finite_mask] = res
            reshaped.append(tmp)
        return reshaped
