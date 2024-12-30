from functools import partial
from dataclasses import dataclass
from collections.abc import Iterable
from typing import TypedDict, Callable, Optional

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


def thomas_model(
    t: np.ndarray,  # Time
    k_T: float,  # Parameter
    q_m: float,  # Parameter
    b: float,  # Parameter
    C_0: float,  # Initial concentration
    length: float,  # Length of the column
    pore_velocity: float,  # Fluid velocity
    rho_p: float,  # Particle density
    epsilon: float,  # Porosity
):
    Z = length
    v = pore_velocity

    r = 1 + (b * C_0)
    n = rho_p * q_m * k_T * Z * (1 - epsilon) / (v * epsilon)
    T = epsilon * (1 / b + C_0) * (v * t / Z - 1) / (rho_p * q_m * (1 - epsilon))

    J1 = J_function(n / r, n * T)
    J2 = J_function(n, n * T / r)

    return J1 / (J1 + (1 - J2) * np.exp((1 - 1 / r) * (n - n * T)))


class ThomasModelInputs(TypedDict):
    k_T: float  # Parameter
    q_m: float  # Parameter
    b: float  # Parameter
    C_0: float  # Initial concentration
    Z: float  # Length of the column
    v: float  # Fluid velocity
    rho_p: float  # Particle density
    epsilon: float  # Porosity


class ExperimentalSetup(TypedDict):
    length: float
    pore_velocity: float
    rho_p: float
    epsilon: float


class BreaktroughData(TypedDict):
    time: np.ndarray
    conc: np.ndarray
    C_0: float


class ParameterFittingConfig(TypedDict):
    initial_guess: float
    lower_bound: float
    upper_bound: float


class TargetParameters(TypedDict):
    k_T: float
    q_m: float
    b: float


@dataclass
class Experiment:
    name: str
    contaminant: str
    setup: ExperimentalSetup
    btc: BreaktroughData
    model: Callable = thomas_model
    parameters: TargetParameters | None = None
    initial_guess: Optional[list[float]] = None
    bounds: Optional[Bounds] = None

    def plot_btc(self):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.axhline(y=self.btc["C_0"], color="gray", ls=(0, (1, 1)), lw=1)
        ax.scatter(
            self.btc["time"],
            self.btc["conc"],
            label=f"{self.contaminant}\n"
            rf"$C_0 = {self.btc['C_0']} \text{{µg}}\,\text{{L}}^{{-1}}$",
            path_effects=line_shade,
        )
        ax.legend()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(f"{self.name} - {self.contaminant}")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Concentration [µg/L]")

        return fig

    def plot_relative_btc(self, with_fit: bool = False):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.axhline(y=1.0, color="gray", ls=(0, (1, 1)), lw=1)

        ax.scatter(
            self.btc["time"],
            self.btc["conc"] / self.btc["C_0"],
            label=f"{self.contaminant}\n"
            rf"$C_0 = {self.btc['C_0']} \text{{µg}}\,\text{{L}}^{{-1}}$",
            path_effects=line_shade,
        )

        if with_fit:
            if self.parameters is None:
                raise ValueError("Parameters not fitted")

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
        if self.parameters:
            fixed_parameters = {
                k: v for k, v in self.parameters.items() if v is not None
            }
        else:
            fixed_parameters = {}

        return partial(
            self.model, C_0=self.btc["C_0"], **self.setup, **fixed_parameters
        )

    def fit(
        self,
        initial_guess: TargetParameters,
        bounds: Bounds,
        loss: str = "soft_l1",
    ):
        if initial_guess is None:
            raise ValueError("Initial guess not provided")

        if bounds is None:
            raise ValueError("Bounds not provided")

        cfit = curve_fit(
            self.callable,
            self.btc["time"],
            self.btc["conc"] / self.btc["C_0"],
            p0=list(initial_guess.values()),
            bounds=bounds,
            method="trf",
            full_output=True,
            loss=loss,
        )

        optimal = {k: v for k, v in zip(initial_guess, cfit[0])}

        if self.parameters is None:
            self.parameters = optimal
        else:
            self.parameters.update(optimal)

    def report_fit(self):
        report = "Best-fit parameters:"

        for k, v in self.parameters.items():
            report += f"\n- ${LATEX_LUT[k]}$ = {v:.2e} {UNITS_LUT[k]}"

        t = self.btc["time"]
        y_obs = self.btc["conc"] / self.btc["C_0"]
        y_fit = self.callable(t, **self.parameters)
        res_sum = np.sum((y_obs - y_fit) ** 2)
        variance = np.sum((y_obs - np.mean(y_obs)) ** 2)
        R_squared = 1 - res_sum / variance

        report += f"\n- R² = {R_squared:.3f}"

        return report
