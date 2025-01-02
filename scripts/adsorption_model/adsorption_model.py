from functools import partial
from dataclasses import dataclass, asdict
from collections.abc import Iterable, Mapping
from typing import Optional, Callable
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patheffects import Stroke, Normal
from scipy.special import i0
from scipy.integrate import quad
from scipy.optimize import curve_fit, Bounds

line_shade = [
    Stroke(linewidth=4, foreground="grey"),
    Normal(),
]


UNITS_LUT = dict(
    k_T="L/µg.h",
    q_m="µg/g",
    b="L/µg",
    C_0="µg/L",
    Z="cm",
    v="cm/h",
    rho_p="g/L",
    epsilon="-",
)

LATEX_LUT = dict(
    k_T=r"$k_{T}$",
    q_m=r"$q_m$",
    b=r"$b$",
    C_0=r"$C_0$",
    Z=r"$Z$",
    v=r"$v$",
    rho_p=r"$\rho_p$",
    epsilon=r"$\epsilon$",
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


@dataclass
class ThomasModelParameters(DataclassMapping):
    k_T: Optional[float] = None
    q_m: Optional[float] = None
    b: Optional[float] = None

    @property
    def are_fitted(self):
        return all([p is not None for p in self.values()])

    @property
    def fixed_parameters(self):
        return {k: v for k, v in self.items() if v is not None}

    @property
    def fittable_parameters(self):
        return {k: v for k, v in self.items() if v is None}


@dataclass
class ThomasExperimentalSetup(DataclassMapping):
    C_0: float
    length: float
    pore_velocity: float
    rho_p: float
    epsilon: float


@dataclass
class BreaktroughData(DataclassMapping):
    time: np.ndarray
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
    epsilon: float,  # Porosity
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
    n = rho_p * q_m * k_T * Z * (1 - epsilon) / (v * epsilon)
    T = epsilon * (1 / b + C_0) * (v * t / Z - 1) / (rho_p * q_m * (1 - epsilon))

    J1 = J_function(n / r, n * T)
    J2 = J_function(n, n * T / r)

    return J1 / (J1 + (1 - J2) * np.exp((1 - 1 / r) * (n - n * T)))


@dataclass
class Experiment:
    name: str
    contaminant: str
    setup: ThomasExperimentalSetup
    parameters: ThomasModelParameters
    model: Callable = ThomasModel
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

    def plot_relative_btc(self, with_fit: bool = False):
        if self.btc is None:
            raise ValueError("No breakthrough data provided")

        fig, ax = plt.subplots(figsize=(3, 3))
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

            t = np.arange(0.1, self.btc["time"].max(), 0.1)  # h
            c = self.callable(t, **self.parameters)
            ax.plot(t, c, label="Fit", path_effects=line_shade)

        ax.legend()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(f"{self.name} - {self.contaminant}")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Rel. Conc. $C/C_0$ [-]")

        return fig

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
        )

        for k, v in zip(initial_guess, cfit[0]):
            setattr(self.parameters, k, v)

    def report_fit(self):
        report = "Best-fit parameters:"

        for k, v in self.parameters.items():
            report += f"\n- ${LATEX_LUT[k]}$ = {v:.e} {UNITS_LUT[k]}"

        t = self.btc.time
        y_obs = self.btc.conc / self.setup.C_0
        y_fit = self.callable(t, **self.parameters)
        res_sum = np.sum((y_obs - y_fit) ** 2)
        variance = np.sum((y_obs - np.mean(y_obs)) ** 2)
        R_squared = 1 - res_sum / variance

        report += f"\n- R² = {R_squared:.3f}"

        return report
