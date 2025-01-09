import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class Index:
    names: list[str]
    values: np.ndarray
    confidence: np.ndarray


@dataclass
class Results:
    ST: Index
    S1: Index
    S2: Index


def organize_results(Si: list[dict], problem: dict):
    names = problem.get("names")
    results = Results(
        ST=Index(
            names,
            np.array([s["ST"] for s in Si]).T,
            np.array([s["ST_conf"] for s in Si]).T,
        ),
        S1=Index(
            names,
            np.array([s["S1"] for s in Si]).T,
            np.array([s["S1_conf"] for s in Si]).T,
        ),
        S2=Index(
            [
                f"{names[0]}, {names[1]}",
                f"{names[0]}, {names[2]}",
                f"{names[1]}, {names[2]}",
            ],
            np.array(
                [
                    [s["S2"][0, 1] for s in Si],
                    [s["S2"][0, 2] for s in Si],
                    [s["S2"][1, 2] for s in Si],
                ]
            ),
            np.array(
                [
                    [s["S2_conf"][0, 1] for s in Si],
                    [s["S2_conf"][0, 2] for s in Si],
                    [s["S2_conf"][1, 2] for s in Si],
                ]
            ),
        ),
    )

    return results


def plot_at_point(results: Results, tidx: int):
    fig, axs = plt.subplots(
        1, 3, figsize=(10, 3), sharey=True, gridspec_kw={"wspace": 0.05}
    )

    for ax, (k, s) in zip(axs, asdict(results).items()):
        dummy_x = np.arange(len(s.names))
        ax.bar(dummy_x, s.values.T[tidx])
        ax.errorbar(
            dummy_x,
            s.values.T[tidx],
            yerr=s.confidence.T[tidx],
            fmt="x",
            color="k",
        )
        ax.set_xticks(dummy_x)
        ax.set_xticklabels(s.names)
        ax.set_title(k)

    for ax in axs:
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.set_ylim(top=1.1)

    plt.show()
