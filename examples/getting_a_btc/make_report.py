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

        report_path = f"report_{case_dir.name}.pdf"

        with PdfPages(report_path) as pdf:
            # Summary end figure
            fig_summary, ax_summary = plt.subplots(figsize=(5, 4))
            ax_summary.set_prop_cycle(plt.cycler("color", plt.cm.tab20c.colors))
            ax_summary.yaxis.get_label().set(size=12)
            ax_summary.xaxis.get_label().set(size=12)
            ax_summary.spines[["top", "right"]].set_visible(False)

            for k, v in data.items():
                # Individual plots
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))

                for ax in axs:
                    ax.set_prop_cycle(custom_cycler)
                    ax.yaxis.get_label().set(size=12)
                    ax.xaxis.get_label().set(size=12)
                    ax.spines[["top", "right"]].set_visible(False)

                # Absolute concentration
                ax = axs[0]
                v.plot.line(x="time", hue="probes", ax=axs[0], add_legend=True)

                if legend := ax.get_legend():
                    legend.set(loc="lower center", bbox_to_anchor=(0.5, 1.1))
                    for t in legend.get_texts():
                        t.set_fontsize(8)

                # Relative concentration
                ax = axs[1]
                rel_conc = v.isel(probes=0) / v.isel(probes=1)
                ax.plot(v.time, rel_conc, label=k)
                ax.set_ylim(0, 1.1)
                ax.axhline(1, color="gray", ls="dotted", lw=0.5)
                ax.set_ylabel("Relative concentration")
                ax.set_xlabel("Time (s)")

                fig.tight_layout()
                pdf.savefig(fig)

                # Summary plot
                ax_summary.plot(v.time, rel_conc, label=k)

            ax_summary.legend()
            ax_summary.set_ylim(-0.05, 1.1)
            ax_summary.axhline(1, color="gray", ls="dotted", lw=0.5)
            ax_summary.set_ylabel("Relative concentration")
            ax_summary.set_xlabel("Time (s)")
            fig_summary.tight_layout()
            pdf.savefig(fig_summary)

        print(f"Report created in {report_path}")


if __name__ == "__main__":
    main()
