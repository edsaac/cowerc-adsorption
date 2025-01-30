from functools import partial
from dataclasses import dataclass, asdict
from collections.abc import Iterable, Mapping
from typing import Optional, Callable
from copy import deepcopy
from numbers import Number

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from matplotlib.patheffects import Stroke, Normal
from scipy.special import i0
from scipy.integrate import quad
from scipy.optimize import curve_fit, Bounds

line_shade = [
    Stroke(linewidth=4, foreground="grey"),
    Normal(),
]


def float_to_scilatex(value: Number, precision: int = 1) -> str:
    precision = int(max(precision, 0))
    coefficient, exponent = f"{value:.{precision}e}".split("e")
    coefficient = float(coefficient)
    exponent = int(exponent)
    return f"{coefficient:.{precision}f} \\times 10^{{{exponent}}}"


UNITS_LUT = dict(
    k_T="L/µg.h",
    q_m="µg/g",
    b="L/µg",
    C_0="µg/L",
    Z="cm",
    v="cm/h",
    rho_p="g/L",
    porosity="-",
)

LATEX_LUT = dict(
    k_T=r"$k_{T}$",
    q_m=r"$q_m$",
    b=r"$b$",
    C_0=r"$C_0$",
    Z=r"$Z$",
    v=r"$v$",
    rho_p=r"$\rho_p$",
    porosity=r"$n$",
)


def J_function(x: float, y: float | np.ndarray) -> float | np.ndarray:
    R"""Calculate the J function for the Thomas model"""

    def to_integrate(tau, yi):
        return np.exp(-yi - tau) * i0(2 * np.sqrt(yi * tau))

    if isinstance(y, Iterable):
        integration = [quad(to_integrate, 0, x, args=(yi,)) for yi in y]

    else:
        integration = [quad(to_integrate, 0, x, args=(y,))]

    integral_value = np.array([iarr[0] for iarr in integration])

    return 1 - integral_value


class DataclassMapping(Mapping):
    @property
    def _asdict(self):
        return asdict(self)

    def __iter__(self):
        return iter(self._asdict)

    def __len__(self):
        return len(self._asdict)

    def __getitem__(self, key):
        return self._asdict[key]

    def items(self):
        return self._asdict.items()

    def values(self):
        return self._asdict.values()

    def keys(self):
        return self._asdict.keys()

    def copy(self):
        return deepcopy(self)


class ModelParameters(DataclassMapping):
    @property
    def are_fitted(self):
        return all([p is not None for p in self.values()])

    @property
    def fixed_parameters(self):
        return {k: v for k, v in self.items() if v is not None}

    @property
    def fittable_parameters(self):
        return {k: v for k, v in self.items() if v is None}

    @property
    def latex_lut(self):
        raise NotImplementedError(
            "This should be implemented by child classes"
        )

    def print_values(self):
        raise NotImplementedError(
            "This should be implemented by child classes"
        )

    def report_fit(self):
        report = "Best-fit parameters:"

        for k, v in self.parameters.items():
            report += f"\n- ${LATEX_LUT[k]}$ = {v:.2e} {UNITS_LUT[k]}"

        t = self.btc.time
        y_obs = self.btc.conc / self.setup.C_0
        y_fit = self.callable(t, **self.parameters)
        res_sum = np.sum((y_obs - y_fit) ** 2)
        variance = np.sum((y_obs - np.mean(y_obs)) ** 2)
        R_squared = 1 - res_sum / variance

        report += f"\n- R² = {R_squared:.3f}"

        return report


@dataclass
class ThomasModelParameters(ModelParameters):
    k_T: Optional[float] = None  # L/µg.h
    q_m: Optional[float] = None  # µg/g
    b: Optional[float] = None  # L/µg

    @property
    def latex_lut(self):
        return dict(k_T=r"k_{T}", q_m=r"q_m", b=r"b")

    @property
    def units_lut(self):
        return dict(k_T="L/µg.h", q_m="µg/g", b="L/µg")

    def print_values(self):
        report = [
            f"${self.latex_lut[k]} = {float_to_scilatex(v, 2)}$ {self.units_lut[k]}"
            for k, v in self.items()
        ]
        return "\n".join(report)


@dataclass
class LogThomasModelParameters(ModelParameters):
    log_k_T: Optional[float] = None
    log_q_m: Optional[float] = None
    log_b: Optional[float] = None

    @property
    def latex_lut(self):
        return dict(log_k_T=r"k_{T}", log_q_m=r"q_m", log_b=r"b")

    @property
    def units_lut(self):
        return dict(log_k_T="L/µg.h", log_q_m="µg/g", log_b="L/µg")

    def print_values(self):
        report = [
            f"${self.latex_lut[k]} = {float_to_scilatex(10**v, 2)}$ {self.units_lut[k]}"
            for k, v in self.items()
        ]
        return "\n".join(report)


@dataclass
class ThomasExperimentalSetup(DataclassMapping):
    C_0: float
    length: float  # cm
    pore_velocity: float  # cm/h
    rho_p: float  # g/cm³
    porosity: float  # -
    particle_size: Optional[float] = None  # cm


@dataclass
class BreaktroughData(DataclassMapping):
    time: np.ndarray  # h
    conc: np.ndarray


def ThomasModel(
    t: np.ndarray,  # <- Independent variable
    k_T: float,  # Parameter
    q_m: float,  # Parameter
    b: float,  # Parameter
    C_0: float,  # Initial concentration
    length: float,  # Length of the column
    pore_velocity: float,  # Fluid velocity
    rho_p: float,  # Particle density
    porosity: float,  # Porosity
    **kwargs,
):
    """
    A model is a function compatible with the scipy.optimize.curve_fit function.

    model(time, **ModelParameters, **ExperimentalSetup)

    where **ExperimentalSetup are constant parameters that are not fitted, and
    **ModelParameters are the parameters that may or not be fitted.

    Parameters in ModelParameters are fitted if they are None.
    """
    Z = length
    v = pore_velocity

    r = 1 + (b * C_0)
    n = rho_p * q_m * k_T * Z * (1 - porosity) / (v * porosity)
    T = (
        porosity
        * (1 / b + C_0)
        * (v * t / Z - 1)
        / (rho_p * q_m * (1 - porosity))
    )

    J1 = J_function(n / r, n * T)
    J2 = J_function(n, n * T / r)

    return J1 / (J1 + (1 - J2) * np.exp((1 - 1 / r) * (n - n * T)))


def LogThomasModel(
    t: np.ndarray,  # <- Independent variable
    log_k_T: float,  # Parameter
    log_q_m: float,  # Parameter
    log_b: float,  # Parameter
    C_0: float,  # Initial concentration
    length: float,  # Length of the column
    pore_velocity: float,  # Fluid velocity
    rho_p: float,  # Particle density
    porosity: float,  # Porosity
    **kwargs,
):
    """
    A model is a function compatible with the scipy.optimize.curve_fit function.

    model(time, **ModelParameters, **ExperimentalSetup)

    where **ExperimentalSetup are constant parameters that are not fitted, and
    **ModelParameters are the parameters that may or not be fitted.

    Parameters in ModelParameters are fitted if they are None.
    """
    Z = length
    v = pore_velocity

    k_T = 10.0**log_k_T
    q_m = 10.0**log_q_m
    b = 10.0**log_b

    r = 1 + (b * C_0)
    n = rho_p * q_m * k_T * Z * (1 - porosity) / (v * porosity)
    T = (
        porosity
        * (1 / b + C_0)
        * (v * t / Z - 1)
        / (rho_p * q_m * (1 - porosity))
    )

    J1 = J_function(n / r, n * T)
    J2 = J_function(n, n * T / r)

    return J1 / (J1 + (1 - J2) * np.exp((1 - 1 / r) * (n - n * T)))


@dataclass
class Experiment:
    name: str
    contaminant: str
    setup: ThomasExperimentalSetup
    parameters: LogThomasModelParameters
    model: Callable = LogThomasModel
    btc: Optional[BreaktroughData] = None

    def plot_btc(self):
        if self.btc is None:
            raise ValueError("No breakthrough data provided")

        fig, ax = plt.subplots(figsize=(3, 3))

        ax.axhline(y=self.setup["C_0"], color="gray", ls=(0, (1, 1)), lw=1)
        ax.scatter(
            self.btc["time"],
            self.btc["conc"],
            label=f"{self.contaminant}\n"
            rf"$C_0 = {self.setup['C_0']} \text{{µg}}\,\text{{L}}^{{-1}}$",
            path_effects=line_shade,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_title(f"{self.name} - {self.contaminant}")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Concentration [µg/L]")
        ax.legend()

        return fig

    def plot_relative_btc(self, with_fit: bool = False, ax=None):
        if self.btc is None:
            raise ValueError("No breakthrough data provided")

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
            _return_fig = True
        else:
            _return_fig = False

        ax.axhline(y=1.0, color="gray", ls=(0, (1, 1)), lw=1)

        ax.scatter(
            self.btc["time"],
            self.btc["conc"] / self.setup["C_0"],
            label=f"{self.contaminant}\n"
            rf"$C_0 = {self.setup['C_0']} \text{{µg}}\,\text{{L}}^{{-1}}$",
            path_effects=line_shade,
        )

        if with_fit:
            if not self.parameters.are_fitted:
                raise ValueError("Parameters are not fitted")

            t = np.linspace(0.1, 1.1 * self.btc["time"].max(), 500)  # h
            c = self.callable(t, **self.parameters)
            label = self.parameters.print_values()
            ax.plot(t, c, label=label, path_effects=line_shade)

        if _return_fig:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_title(f"{self.name} - {self.contaminant}")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Rel. Conc. $C/C_0$ [-]")
            return fig

    def plot_relative_btc_in_porevols(self, with_fit: bool = False, ax=None):
        if self.btc is None:
            raise ValueError("No breakthrough data provided")

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
            _return_fig = True
        else:
            _return_fig = False

        ax.axhline(y=1.0, color="gray", ls=(0, (1, 1)), lw=1)

        ax.scatter(
            self.btc["time"] / self.pv,
            self.btc["conc"] / self.setup["C_0"],
            label=f"{self.contaminant}\n"
            rf"$C_0 = {self.setup['C_0']} \text{{µg}}\,\text{{L}}^{{-1}}$",
            path_effects=line_shade,
        )

        if with_fit:
            if not self.parameters.are_fitted:
                raise ValueError("Parameters are not fitted")

            t = np.linspace(0.1, 1.1 * self.btc["time"].max(), 500)  # h
            c = self.callable(t, **self.parameters)
            label = self.parameters.print_values()
            ax.plot(t / self.pv, c, label=label, path_effects=line_shade)

        if _return_fig:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_title(f"{self.name} - {self.contaminant}")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Rel. Conc. $C/C_0$ [-]")
            return fig

    def plot_only_fit(self, ax=None) -> Figure | None:
        """Make a plot of only the fitted model"""

        if not self.parameters.are_fitted:
            raise ValueError("Parameters are not fitted")

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
            _return_fig = True
        else:
            _return_fig = False

        t = np.linspace(0.1, 1.1 * self.btc["time"].max(), 500)  # h
        c = self.callable(t, **self.parameters)

        R2 = self.R2
        label = self.parameters.print_values() + f"\nR² = {R2:.3f}"

        ax.plot(t, c, label=label, path_effects=line_shade)

        if _return_fig:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_title(f"{self.name} - {self.contaminant}")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Rel. Conc. $C/C_0$ [-]")
            return fig

    @property
    def R2(self):
        if not self.parameters.are_fitted:
            raise ValueError("Parameters are not fitted")

        if self.btc is None:
            raise ValueError("No breakthrough data provided")

        res = np.sum(np.power(self.fit_result[2]["fvec"], 2))
        var = np.sum(np.power(self.btc.conc - np.mean(self.btc.conc), 2))

        return 1 - res / var

    @property
    def pv(self):
        "Returns a PV in time units"
        return self.setup.length / self.setup.pore_velocity

    @property
    def reynolds(self):
        if self.setup.particle_size is None:
            raise ValueError("Experiment has no defined particle size")

        kin_viscosity = 36  # cm²/h = 10⁻⁶ m²/s
        return (
            self.setup.particle_size * self.setup.pore_velocity / kin_viscosity
        )

    @property
    def callable(self):
        return partial(
            self.model,
            **self.parameters.fixed_parameters,
            **self.setup,
        )

    def fit(
        self,
        initial_guess: ThomasModelParameters | None = None,
        bounds: Bounds | None = None,
        loss: str = "soft_l1",
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
    ):
        if not initial_guess:
            raise ValueError("Initial guess not provided")

        if not bounds:
            raise ValueError("Bounds not provided")

        if self.parameters.are_fitted:
            return

        cfit = curve_fit(
            self.callable,
            self.btc.time,
            self.btc.conc / self.setup.C_0,
            p0=list(initial_guess.fixed_parameters.values()),
            bounds=bounds,
            method="trf",
            full_output=True,
            loss=loss,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        self.fit_result = cfit
        for k, v in zip(initial_guess, cfit[0]):
            setattr(self.parameters, k, v)
