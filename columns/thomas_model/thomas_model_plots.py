import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import Stroke, Normal

from thomas_model_equations import thomas_model, ThomasModelInputs, UNITS_LUT, LATEX_LUT

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", ["#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"]
)

line_shade = [
    Stroke(linewidth=4, foreground="grey"),
    Normal(),
]


def make_scatter_plot(data: dict):
    """
    data = {
        experiment_name: {
            contaminant_name: {
                "time": time,
                "conc": concentration,
                "C_0": initial_concentration,
                "style": {line_style}
            }
        }
    }

    """

    fig_abs, axs = plt.subplots(1, len(data), figsize=(6, 3))
    for ax, (experiment_name, experiment_data) in zip(axs, data.items()):
        for contaminant_name, contaminant_data in experiment_data.items():
            if contaminant_name.startswith("_"):
                continue

            label = (
                f"{contaminant_name}\n"
                rf"$C_0 = {contaminant_data['C_0']} \text{{µg}}\,\text{{L}}^{{-1}}$"
            )
            ax.scatter(
                contaminant_data["time"],
                contaminant_data["conc"],
                label=label,
                **contaminant_data["style"],
            )

            ax.axhline(
                y=contaminant_data["C_0"],
                color=contaminant_data["style"]["color"],
                ls=(0, (1, 1)),
                lw=1,
            )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(experiment_name)
        ax.set_xlabel("Time [h]")

    fig_abs.supylabel("Concentration [µg/L]")

    ###

    fig_rel, axs = plt.subplots(1, len(data), figsize=(6, 3), sharey=True)
    for ax, (experiment_name, experiment_data) in zip(axs, data.items()):
        for contaminant_name, contaminant_data in experiment_data.items():
            if contaminant_name.startswith("_"):
                continue

            label = (
                f"{contaminant_name}\n"
                rf"$C_0 = {contaminant_data['C_0']} \text{{µg}}\,\text{{L}}^{{-1}}$"
            )
            ax.scatter(
                contaminant_data["time"],
                contaminant_data["conc"] / contaminant_data["C_0"],
                label=label,
                **contaminant_data["style"],
            )

        ax.axhline(
            y=1.0,
            color="k",
            ls=(0, (1, 1)),
            lw=1,
        )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(experiment_name)
        ax.set_xlabel("Time [h]")

    fig_rel.supylabel("Concentration [µg/L]")

    return (fig_abs, fig_rel)


def make_scatter_and_fit_plot(
    model_params: ThomasModelInputs,
    experiment: dict,
    contaminant: str = "",
):
    t_obs = experiment[contaminant]["time"]
    y_obs = experiment[contaminant]["conc"]

    t = np.arange(0.1, t_obs.max(), 0.1)  # h
    conc = thomas_model(t, **model_params)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(t, conc, path_effects=line_shade, label="Fit")
    ax.scatter(t_obs, y_obs, color="k", label="Data", marker="x")
    ax.legend()
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Rel. Concentration (-)")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(left=0)
    ax.set_title(contaminant)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig


def make_single_plot(thomas_model_params: ThomasModelInputs):
    t = np.arange(0.1, 40.5, 0.1)  # h
    conc = thomas_model(t, **thomas_model_params)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(t, conc, path_effects=line_shade, lw=3)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Rel. Concentration (-)")
    ax.set_ylim(0, 1.1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig


def variate_some_parameter(
    thomas_model_params: ThomasModelInputs, param: str, values: list[float]
):
    masked_params = thomas_model_params.copy()
    masked_params.pop(param)

    t = np.arange(0.1, 40.5, 0.1)  # h
    conc = {}

    for value in values:
        masked_params[param] = value
        conc[f"{value}"] = thomas_model(t, **masked_params)

    fig, ax = plt.subplots(figsize=(3, 3))
    for value, c in conc.items():
        ax.plot(t, c, label=value, path_effects=line_shade, lw=3)

    ax.set_ylim(0, 1.1)
    ax.set_xlim(left=0)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Rel. conc. [-]")

    unit = UNITS_LUT[param]
    symbol = LATEX_LUT[param]
    ax.legend(title=f"{symbol} [{unit}]", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig
