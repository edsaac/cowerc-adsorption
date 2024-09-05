from espuma import Boundary_Probe, Case_Directory
from sys import argv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler


def main():
    if len(argv) != 2:
        raise ValueError("Please provide the case directory as an argument")
    else:
        case_dir = Path(argv[1])

        if not case_dir.exists():
            raise ValueError(f"Case directory {case_dir} does not exist")

        if not case_dir.is_dir():
            raise ValueError(f"Case directory {case_dir} is not a directory")

        of_case = Case_Directory(case_dir)
        prb = Boundary_Probe(of_case, of_case.system.boundaryProbes)

        print(f"Probes for {' '.join(prb.field_names)} were found.")

        data = prb.array_data

        custom_cycler = cycler(color=["#444444", "#000000"]) + cycler(
            ls=["-", "dashed"]
        )

        with PdfPages("report.pdf") as pdf:
            for k, v in data.items():
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_prop_cycle(custom_cycler)
                v.plot.line(x="time", hue="probes", ax=ax, add_legend=True)
                ax.yaxis.get_label().set(size=14)
                ax.xaxis.get_label().set(size=14)
                ax.spines[["top", "right"]].set_visible(False)
                legend = ax.get_legend()
                legend.set(loc="lower center", bbox_to_anchor=(0.5, 1.1))
                for t in legend.get_texts():
                    t.set_fontsize(8)
                fig.tight_layout()
                pdf.savefig(fig)

        print("Report created in report.pdf")


if __name__ == "__main__":
    main()
