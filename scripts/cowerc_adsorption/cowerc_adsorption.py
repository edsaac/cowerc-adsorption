import functools
from time import perf_counter
from itertools import cycle
from typing import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray, ArrayLike
from numba import jit
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import patheffects


def _timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = perf_counter()
        value = func(*args, **kwargs)
        end_time = perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__}() in {run_time:.4f} secs")
        return value

    return wrapper_timer


_COLORS = ["darkgrey", "purple", "blue", "green", "orange", "red", "hotpink"]


@dataclass
class PhysicalParams:
    """Physical parameters for the adsorption model

    Parameters
    ----------
    L: float
        Length of the column [L]
    v: float
        Pore velocity through the column [L/T]
    n: float
        Porosity (L³/L³)
    sm: float
        Adsorption capacity of the adsorbate [M/L³]
    k_ads: ArrayLike
        Adsorption rate constant [1/T] for each contaminant
    k_des: ArrayLike
        Desorption rate constant [1/T] for each contaminant
    C_0: ArrayLike
        Initial concentration of each contaminant [M/L³]
    """

    L: float
    v: float
    n: float
    sm: float
    k_ads: ArrayLike
    k_des: ArrayLike
    C_0: ArrayLike

    def __post_init__(self) -> None:
        self.k_ads = np.array(self.k_ads)
        self.k_des = np.array(self.k_des)
        self.C_0 = np.array(self.C_0)

        if self.k_ads.shape != self.k_des.shape:
            raise ValueError("k_ads and k_des must have the same shape")

        if self.k_ads.shape != self.C_0.shape:
            raise ValueError("k_ads and C_0 must have the same shape")

    @property
    def N(self) -> int:
        """Infered number of contaminants

        Returns
        -------
        int
            Number of contaminants
        """
        return len(self.k_ads)

    @property
    def Dam_ads(self) -> NDArray:
        """Damköhler numbers for adsorption

        Returns
        -------
        NDArray
            Damköhler numbers for adsorption
        """
        return self.k_ads * self.L / self.v

    @property
    def Dam_des(self) -> NDArray:
        """Damköhler numbers for desorption

        Returns
        -------
        NDArray
            Damköhler numbers for desorption
        """
        return self.k_des * self.L / self.v

    @property
    def psi(self) -> NDArray:
        """Non-dimensional parameter psi

        Returns
        -------
        NDArray
            Non-dimensional parameter psi
        """
        return self.k_ads / self.k_des

    @property
    def kappa(self) -> NDArray:
        """Non-dimensional parameter kappa

        Returns
        -------
        NDArray
            Non-dimensional parameter psi
        """
        return self.n * self.C_0 / self.sm

    @property
    def nondim(self) -> dict[str, NDArray]:
        """
        Returns
        -------
        dict[str, NDArray]
            Convenience dict with non-dimensional parameters and boundary condition
        """
        return {
            "Dam_ads": self.Dam_ads,
            "Dam_des": self.Dam_des,
            "kappa": self.kappa,
            "bc": np.ones_like(self.C_0),
        }


@jit(cache=True)
def _advance_timestep(previous_step: tuple[NDArray, NDArray], next_step: tuple[NDArray, NDArray], *args):
    """Jiggery-pokery with slices and numba to go brr

    Previous step is (c, s) and next step is (c_new, s_new). Those are just references to the arrays,
    not actual copies so operations in this function are done in-place. That is why we do not return
    anything.

    Keeping the while loop outside the compiled function gives slightly better performance
    than keeping it inside. Compilation is lazy so it doesn't happen until the function is called -- which is why
    the first timestep will be slower. Keeping this function outside the class avoids recompilation every timestep.
    """
    c, s = previous_step
    c_new, s_new = next_step
    Dam_ads, Dam_des, kappa, bc, dt, dz = args

    c_new[:, 0] = bc[:]

    c_new[:, 1:] = (
        dt
        * (
            -Dam_ads[:, None] * c[:, 1:] * (1 - np.sum(s[:, 1:], axis=0))
            + (Dam_des[:, None] / kappa[:, None]) * s[:, 1:]
        )
        - dt / dz * (c[:, 1:] - c[:, :-1])
        + c[:, 1:]
    )

    s_new[:] = dt * (kappa[:, None] * Dam_ads[:, None] * c * (1 - np.sum(s, axis=0)) - Dam_des[:, None] * s) + s


@dataclass
class Simulation:
    Dam_ads: NDArray
    Dam_des: NDArray
    kappa: NDArray
    bc: NDArray

    def __post_init__(self):
        # Check shapes are consistent
        if any(
            [
                (self.Dam_ads.shape != self.Dam_des.shape),
                (self.Dam_ads.shape != self.kappa.shape),
                (self.Dam_ads.shape != self.bc.shape),
            ]
        ):
            raise ValueError("Dam_ads, Dam_des, psi and bc must have equal shape")

        self.N = len(self.bc)

        # Generate discretization
        self.dz = 0.01
        self.z = np.arange(0, 1 + self.dz, self.dz)

        ## Time stepping controls
        self.cfl = 0.2
        self.write_every = 10
        self.end_time = 50
        self.t = 0
        self.it = 0

        ## Create arrays for c and s
        c = np.zeros((self.N, len(self.z)))
        s = np.zeros_like(c)

        c[:, 0] = self.bc[:]

        # Initialize time index
        ## :TODO: Use xarray to store all the results
        self.times: dict[str, dict[str, np.ndarray]] = {}
        self.times[f"{self.t:.2f}"] = {
            "c": c.copy(),
            "s": s.copy(),
        }

    @property
    def dt(self) -> float:
        """Time step

        Returns
        -------
        float
            Time step
        """
        return self.cfl * self.dz

    @property
    def psi(self) -> NDArray:
        """Non-dimensional parameter psi

        Returns
        -------
        NDArray
            Non-dimensional parameter psi
        """
        return self.Dam_ads / self.Dam_des

    def _get_params_for_numba(self):
        return (self.Dam_ads, self.Dam_des, self.kappa, self.bc, self.dt, self.dz)

    @property
    def btc(self) -> tuple[list[float], NDArray]:
        """Breakthrough curve

        Returns
        -------
        tuple[list[float], NDArray]
            Time points and concentration at the effluent for all contaminants
        """
        tp = list(map(float, self.times.keys()))
        btc = np.vstack([x["c"][:, -1] for x in self.times.values()]).T
        return tp, btc

    def btc_df(self) -> pd.DataFrame:
        df = pd.DataFrame({}, index=self.btc[0])

        for i, ci in enumerate(self.btc[1]):
            df[f"conc_{i}"] = ci

        return df

    # @_timer
    def solve(self) -> None:
        """Numerical solution of the mass balance equations using an explicit
        finite difference method

        Returns
        -------
        None
        """
        c = self.times["0.00"]["c"].copy()
        s = self.times["0.00"]["s"].copy()

        c_new = np.zeros_like(c)
        s_new = np.zeros_like(s)

        while self.t < self.end_time:
            # Advance the timestep
            self.t += self.dt
            self.it += 1
            _advance_timestep((c, s), (c_new, s_new), *self._get_params_for_numba())

            # Swap arrays for next step
            c, c_new = c_new, c
            s, s_new = s_new, s

            # Write to dict every write_every steps
            if self.it % self.write_every == 0:
                self.times[f"{self.t:.2f}"] = {
                    "c": c.copy(),
                    "s": s.copy(),
                }

    def plot_over_z(self, times: Iterable[float | str]) -> plt.Figure:
        """Plot the concentration of each contaminant over the column for the
        specified times.

        Parameters
        ----------
        times : Iterable[float | str]
            Times to plot

        Returns
        -------
        plt.Figure
        """

        times = [f"{float(t):.2f}" for t in times]
        filtered_times = {k: v for k, v in self.times.items() if k in times}

        n_colors = len(filtered_times)
        plot_kwargs = dict(path_effects=[patheffects.withStroke(linewidth=4, foreground="grey")])

        fig, axes = plt.subplots(
            *(2, self.N),
            sharex=True,
            sharey=True,
            gridspec_kw={"wspace": 0.05, "hspace": 0.15},
            figsize=(8, 10),
            squeeze=False,
        )
        for ax in axes.flatten():
            ax.set_prop_cycle(plt.cycler("color", plt.cm.GnBu(np.linspace(0.10, 0.90, n_colors))))
            ax.spines.right.set_visible(False)
            ax.spines.bottom.set_visible(False)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            ax.set_facecolor("#FFFFFF80")
            ax.set_ylim(1.02, -0.02)
            ax.set_xlim(-0.04, 1.09)

        axc, axs = axes

        for k, data in filtered_times.items():
            for ax, s in zip(axs, data["s"]):
                ax.plot(s, self.z, label=k, **plot_kwargs)

            for ax, c in zip(axc, data["c"]):
                ax.plot(c, self.z, label=k, **plot_kwargs)

        for ax in axes.flatten():
            ax.legend(title=R"$\hat{t}$", fontsize=10)  # , bbox_to_anchor=(1, 0.5))

        axc[0].set_xlabel(R"$\hat{c}$", fontsize=14)
        axs[0].set_xlabel(R"$\hat{s}$", fontsize=14)
        for ax in axes[:, 0]:
            ax.set_ylabel(R"$\hat{z}$", fontsize=14)

        fig.set_facecolor("#FFFFFF80")

        plt.close()
        return fig

    def plot_heatmap(self) -> plt.Figure:
        """Plot the aqueous and the adsorbed concentrations over time and over
        depth as a heatmap.

        Returns
        -------
        plt.Figure
        """
        t = [float(ti) for ti in self.times.keys()]
        z = self.z
        c = np.dstack([ci["c"] for ci in self.times.values()])
        s = np.dstack([si["s"] for si in self.times.values()])

        fig, axes = plt.subplots(
            *(2, self.N),
            sharex="row",
            sharey=True,
            gridspec_kw={"wspace": 0.05, "hspace": 0.20},
            figsize=(12, self.N * 4),
            squeeze=False,
        )

        for ax in axes.flatten():
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.set_facecolor("#FFFFFF80")
            ax.set_ylim(1, 0)
            ax.set_xlim(0, self.end_time)

        axc, axs = axes
        pcm_kwargs = dict(cmap="bone_r", rasterized=True, vmin=0, vmax=1)
        for i, (ax, ci) in enumerate(zip(axc, c), start=1):
            im = ax.pcolormesh(t, z, ci, **pcm_kwargs)
            plt.colorbar(im, label=Rf"$\hat{{c}}_{i}$", aspect=10, shrink=0.8)
            ax.set_title(Rf"$\hat{{c}}_{i}$", fontsize=14)

        for i, (ax, si) in enumerate(zip(axs, s), start=1):
            im = ax.pcolormesh(t, z, si, **pcm_kwargs)
            plt.colorbar(im, label=Rf"$\hat{{s}}_{i}$")
            ax.set_title(Rf"$\hat{{s}}_{i}$", fontsize=14)

        fig.supxlabel(R"$\hat{t}$", fontsize=14, y=0.02)
        fig.set_facecolor("#FFFFFF80")

        plt.close(fig)
        return fig

    def plot_breakthrough(self) -> plt.Figure:
        """Plot breakthrough curves.

        Returns
        -------
        plt.Figure
        """
        t, btc = self.btc
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = cycle(_COLORS)

        max_btc = 1.0
        for i, (curve, color) in enumerate(zip(btc, colors), start=1):
            ax.plot(
                *(t, curve),
                path_effects=[patheffects.withStroke(linewidth=4, foreground="grey")],
                lw=2,
                c=color,
                label=Rf"$\hat{{c}}_{i}$",
            )
            max_btc = max(max_btc, max(curve))

        ax.set_facecolor("#FFFFFF80")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_ylabel(R"$\hat{c}$", fontsize=14)
        ax.set_xlabel(R"$\hat{t}$", fontsize=14)
        ax.set_ylim(-0.05, 1.1 * max_btc)
        ax.axhline(y=1, lw=1, ls="dotted", c="k")
        ax.legend(
            title=R"$\hat{c}_j\left( \hat{t} \right)$",
            fontsize=10,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )

        fig.set_facecolor("#FFFFFF80")

        plt.close()
        return fig


@dataclass
class ExperimentalBreakthroughData:
    """
    Parameters
    ----------
    time: NDArray
        Non-dimensional times for the breakthrough data (-)
    conc: NDArray
        Non-dimensional effluent concentrations (-)
    init_conc: NDArray
        Initial concentrations [mass/volume]
    """

    time: NDArray
    conc: NDArray
    c_0: ArrayLike

    def plot_breakthrough(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = cycle(_COLORS)

        max_btc = 1.0
        for i, (curve, color) in enumerate(zip(self.conc, colors), start=1):
            ax.scatter(
                *(self.time, curve),
                path_effects=[patheffects.withStroke(linewidth=4, foreground="grey")],
                s=10,
                c=color,
                label=Rf"$\hat{{c}}_{i}$",
            )
            max_btc = max(max_btc, max(curve))

        ax.set_facecolor("#FFFFFF80")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_ylabel(R"$\hat{c}$", fontsize=14)
        ax.set_xlabel(R"$\hat{t}$", fontsize=14)
        ax.set_ylim(-0.05, 1.1 * max_btc)
        ax.axhline(y=1, lw=1, ls="dotted", c="k")
        ax.legend(
            title=R"$\hat{c}_j\left( \hat{t} \right)$",
            fontsize=10,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )

        fig.set_facecolor("#FFFFFF80")

        plt.close()
        return fig

    def print_observations_OSTRICH(self) -> None:
        """Print observations for Ostrich"""
        for i, c in enumerate(self.conc.flatten(), 0):
            print(f"obs{i}\t{c:.4f}\t1.00\tresults.dat\tOST_NULL\t{i}\t1")


def plot_btc_and_data(simulation: Simulation, experimental_data: ExperimentalBreakthroughData):
    """Plot breakthrough curves from an simulation along with experimental data"""
    fig = simulation.plot_breakthrough()
    ax = fig.axes[0]
    colors = cycle(_COLORS)

    for btc, color in zip(experimental_data.conc, colors):
        ax.scatter(
            experimental_data.time,
            btc,
            c=color,
            path_effects=[patheffects.Stroke(linewidth=1, foreground="#000")],
        )

    ax.set_ylim(bottom=-0.09, top=1.4)
    plt.close(fig)
    return fig
