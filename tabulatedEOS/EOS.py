from abc import ABC, abstractmethod
from typing import Callable, Optional, TYPE_CHECKING, Any
from functools import reduce, cached_property, lru_cache
from warnings import warn
import numpy as np
import alpyne.uniform_interpolation as ui  # type: ignore

from .unit_system import unit_systems, UnitSystem

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    ND = np.ndarray
    Mask = ND[tuple[int, ...], np.dtype[np.bool]]
    Array1D = ND[tuple[int], np.dtype[np.float64]]
    Array3D = ND[tuple[int, int, int], np.dtype[np.float64]]
    Scalar = float | np.floating



def _return_first(*args, **_):
    return args[0]

class TabulatedEOS(ABC):
    eos_units: UnitSystem
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
          - self.eos_units: UnitSyste,
              one of the unit systems defined in tabulatedEOS.unit_systems
          - self.conversions: dict[str, Callable]
              links key to unit conversion functions to be used

          - should also initialize variables based on path that will be used by get_keys
    get_key:
        This function should return the data for the given key.
    """

    name: str
    table_keys: dict[str, str]
    log_names: list[str] = ['rho', 'temp']

    def __init__(
        self,
        path: Optional[str] = None,
        code_units: str = "CGS",
        **kwargs):

        self.name = "Unitilialized"
        self.conversions: dict[str, Callable[[UnitSystem, UnitSystem], float]] = {}
        self.code_units = unit_systems[code_units]
        if path is None:
            return
        self.post_init(path, **kwargs)

        if not hasattr(self, "table_keys"):
            raise ValueError("post_init has to initialize self.table_keys")
        if not hasattr(self, "eos_units"):
            raise ValueError("post_init has to initialize self.eos_units")
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

        Should also initialize the variables needed by get_key
        based on input path and other keyword arguments.
        """
        ...

    @abstractmethod
    def get_key(self, key: str) -> "Array1D | Array3D":
        """
        Returns the raw data for the given key.
        This will be wrapped by functools.lru_cache to speed up interpolation.
        """
        ...

    def unit_conversion(self, key: str) -> float:
        """
        Returns the conversion factor from eos_units to code_units for a given key
        """
        if key not in self.conversions:
            return 1.0
        return self.conversions[key](self.eos_units, self.code_units)

    @lru_cache
    def get_key_with_units(self, key: str) -> "Array1D | Array3D":
        """
        Returns the data for the given key with unit conversion.
        """
        return self.get_key(key) * self.unit_conversion(key)


    def inverse_call(self, **kwargs: "ArrayLike") -> "ND":
        """
        Solve for the missing one among {'ye','temp','rho'} so that
        field_table(ye, temp, rho) == field_value.

        kwargs must include exactly two of {'ye','temp','rho'} and exactly one
        extra key that names the output field table to invert (e.g. 'pressure').
        Everything may be scalar or array-like and will be broadcast.
        """

        axes = tuple(self.table_keys.keys())

        field_key: str | None = None
        target_val: "ND | None" = None

        kw = dict(kwargs)
        for k in kw:
            if k in self.log_names:
                kw[k] = np.log10(kw[k])

        for k in list(kw.keys()):
            if k not in axes:
                field_key = k
                target_val = np.asarray(kw.pop(k), dtype=float)
                break
        if field_key is None or target_val is None or len(kw) > 2:
            raise ValueError("Provide exactly one output field key to invert (e.g. 'pressure').")

        missing = [ax for ax in axes if ax not in kw]
        if len(missing) != 1:
            raise ValueError("Exactly one of {'ye','temp','rho'} must be missing (the solve target).")
        target_ax = missing[0]
        stat_ax = tuple(ax for ax in axes if ax != target_ax)

        g = {ax: self.get_key_with_units(self.table_keys[ax]) for ax in axes}
        for k in self.log_names:
            g[k] = np.log10(g[k])
        T0 = self.get_key_with_units(field_key)

        if np.all(T0 > 0):
            target_val = np.log10(target_val)
            T0 = np.log10(T0)

        axis_index = {k: idx for idx, k in enumerate(axes)}
        A, B, C = stat_ax[0], stat_ax[1], target_ax
        perm = (axis_index[A], axis_index[B], axis_index[C])
        T = np.transpose(T0, perm)  # shape: (nA, nB, nC)
        gC = g[C]                   # 1D grid along target axis
        nA, nB, nC = T.shape

        def frac_index(val: "ArrayLike", grid: "ND", n: int) -> tuple["ND", "ND", "ND", "ND"]:
            """Uniform grid: map value -> (lower index, weight in [0,1])."""
            val = np.asarray(val, dtype=float)
            dx = grid[1] - grid[0]
            i_float = (val - grid[0]) / dx
            i0 = np.floor(i_float).astype(int)
            i0 = np.clip(i0, 0, n - 2)
            w = i_float - i0
            w = np.clip(w, 0.0, 1.0)
            return i0, i0+1, w, 1-w

        iA0, iA1, wA0, wA1 = frac_index(kw[A], g[A], nA)
        iB0, iB1, wB0, wB1 = frac_index(kw[B], g[B], nB)

        c00 = wA1 * wB1
        c01 = wA1 * wB0
        c10 = wA0 * wB1
        c11 = wA0 * wB0

        batch_shape = np.broadcast(c00, c01, c10, c11, target_val).shape
        tv = np.broadcast_to(target_val, batch_shape)

        def val_at_k(k: "ND | int") -> "ND":
            return (
                c00 * T[iA0, iB0, k] +
                c01 * T[iA0, iB1, k] +
                c10 * T[iA1, iB0, k] +
                c11 * T[iA1, iB1, k]
            ) - tv

        lo = np.zeros(batch_shape, dtype=int)
        hi = np.full(batch_shape, nC - 1, dtype=int)
        f_lo = val_at_k(lo)
        f_hi = val_at_k(hi)

        inc = f_hi > f_lo

        bad = f_lo * f_hi > 0
        if np.any(bad):
            raise ValueError(
                f"value out of range along '{C}' for some elements (cannot bracket)."
            )

        # Batched index bisection: O(log2(nC)) lookups on the target axis only
        max_iter = int(np.ceil(np.log2(nC))) + 2
        for _ in range(max_iter):
            active = (hi - lo) > 1
            if not np.any(active):
                break
            mid = (lo + hi) // 2
            f_mid = val_at_k(mid)

            less = f_mid <= 0
            go_right = np.where(inc, less, ~less) & active
            go_left  = (~go_right) & active

            lo = np.where(go_right, mid, lo)
            f_lo = np.where(go_right, f_mid, f_lo)

            hi = np.where(go_left, mid, hi)
            f_hi = np.where(go_left, f_mid, f_hi)

        x_lo = gC[lo]
        x_hi = gC[hi]
        denom = (f_lo - f_hi)
        t = np.divide(f_lo, denom, out=np.zeros_like(tv, dtype=float), where=denom != 0.0)
        t = np.clip(t, 0.0, 1.0)  # be safe near flat spots
        x = x_lo + t * (x_hi - x_lo)

        if C in self.log_names:
            return 10**x
        return x


    def _read_table(self) -> None:
        self._offsets = {}
        self._inv_steps = {}

        for name, key in self.table_keys.items():
            dat = self.get_key_with_units(key)
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
                    data_slice: dict[str, int] = {},
                    func: Callable = _return_first,
                    ) -> Callable:
        self._check_initialized()

        def eos_caller(
            *args: "ND",
            **kwargs: "ND",
        ) -> "ND":

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

    def get_caller(self, keys: list[str], func: Callable = _return_first,) -> Callable:
        return self._get_caller(
            arguments=['ye', 'temp', 'rho'],
            keys=keys,
            func=func
        )

    def get_cold_caller(self, keys: list[str], func: Callable = _return_first,) -> Callable:
        return self._get_caller(
            arguments=['ye', 'rho'],
            data_slice={'temp': 0},
            keys=keys,
            func=func
        )

    def get_inf_caller(self, keys: list[str], func: Callable = _return_first,) -> Callable:
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
            dat = self.get_key_with_units(key)
            table_range[name] = (dat[0], dat[-1])
        return table_range

    def __call__(
        self,
        key: str,
        ye: 'ArrayLike',
        temp: 'ArrayLike',
        rho: 'ArrayLike',
    ) -> 'ArrayLike':
        is_scalar = np.isscalar(ye)
        ye = np.atleast_1d(ye)
        temp = np.atleast_1d(temp)
        rho = np.atleast_1d(rho)
        if is_scalar:
            return self.get_caller([key])(ye=ye, temp=temp, rho=rho)[0]
        return self.get_caller([key])(ye=ye, temp=temp, rho=rho)

    def __getitem__(self, key: str) -> "Array1D | Array3D":
        return self.get_key_with_units(key)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with name {self.name}"

    def _check_inputs(
        self,
        inputs: "dict[str, ND[np.float_]]",
    ) -> tuple["dict[str, ND[np.float_]]", tuple[int, ...]]:
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
        inputs: dict[str, "ND"],
    ) -> tuple[list["ND"], "Mask"]:
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
        offsets = np.array([self._offsets[name] for name in self.table_keys if name in arguments])
        inv_steps = np.array([self._inv_steps[name] for name in self.table_keys if name in arguments])
        return offsets, inv_steps

    def _slice_data(
        self,
        keys: list[str],
        data_slice: dict[str, int],
    ) -> "ND":

        data_slice = dict(sorted(data_slice.items(), key=lambda it: it[1]))

        def slice_data(data):
            for ii, tk in list(enumerate(self.table_keys))[::-1]:
                if tk in data_slice:
                    data = data.take(data_slice[tk], axis=ii)
            return data

        data = np.array([slice_data(self.get_key_with_units(kk)) for kk in keys])
        return data

    def _logspace_data(
        self,
        data: "ND",
    ) -> tuple["ND", "Mask"]:
        islog = np.array([np.all(dd > 0) for dd in data])
        data[islog] = np.log10(data[islog])
        return data, islog

    @staticmethod
    def _convert_args_to_kwargs(
        args: tuple["ND", ...],
        kwargs: dict[str, Any],
        arguments: list[str],
    ) -> dict[str, "ND"]:
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
    ) -> tuple[dict[str, "ND"], dict[str, Any]]:
        inputs = {key: kwargs.pop(key)
                  for key in arguments
                  if key in kwargs}

        return inputs, kwargs

    @staticmethod
    def _reshape_result(
        result: "ND",
        shape: tuple[int, ...],
        finite_mask: "Mask",
        islog: "Mask",
    ) -> list["ND"]:
        reshaped = []
        for log, res in zip(islog, result):
            tmp = np.zeros(shape)*np.nan
            if log:
                tmp[finite_mask] = 10**res
            else:
                tmp[finite_mask] = res
            reshaped.append(tmp)
        return reshaped


def batched_bisection(eval_f, a, b, *, xtol=1e-8, maxiter=100):
    """
    Vectorized bisection on many targets at once.

    Parameters
    ----------
    eval_f : callable
        f(t) -> array. Must be vectorized over t (broadcast OK).
        Should satisfy f(a)*f(b) <= 0 elementwise.
    a, b : float or array-like
        Lower/upper bounds; will be broadcast to the common shape.
    xtol : float
        Stop when max(b - a) < xtol.
    maxiter : int
        Safety cap on iterations.

    Returns
    -------
    t : ndarray
        Approximate roots with the broadcast shape of a and b.
    """
    lo = np.broadcast_to(np.asarray(a, dtype=float), np.broadcast(np.asarray(a), np.asarray(b)).shape).copy()
    hi = np.broadcast_to(np.asarray(b, dtype=float), lo.shape).copy()

    f_lo = eval_f(lo)
    f_hi = eval_f(hi)

    bad = f_lo * f_hi > 0
    if np.any(bad):
        raise ValueError("Bisection requires sign change on [a,b] per element.")

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        f_mid = eval_f(mid)

        left_interval = f_lo * f_mid <= 0  # keep [lo, mid] where sign changes
        hi = np.where(left_interval, mid, hi)
        f_hi = np.where(left_interval, f_mid, f_hi)

        right_interval = ~left_interval   # keep [mid, hi] otherwise
        lo = np.where(right_interval, mid, lo)
        f_lo = np.where(right_interval, f_mid, f_lo)

        if np.max(hi - lo) < xtol:
            break

    return 0.5 * (lo + hi)
