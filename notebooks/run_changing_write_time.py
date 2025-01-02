from espuma import Case_Directory
from subprocess import run

of_template = Case_Directory("../templates/first_order_adsorption/rssct")
of_case = Case_Directory.clone_from_template(
    of_template, "_column_case", overwrite=False
)

of_case._blockMesh()

## Initial run to capture the first pore volumes
of_case.system.controlDict["endTime"] = 600
of_case.system.controlDict["writeInterval"] = 60
of_case.system.boundaryProbes["writeInterval"] = 0.07
of_case._runCase()


## Longer run to capture slower kinetics
latest_time = of_case._foamListTimes()[-1]

run(
    [
        "cp",
        str(of_case.path / "0/darcyFlux"),
        str(of_case.path / f"{latest_time}"),
    ]
)

of_case.system.controlDict["endTime"] = 162000
of_case.system.controlDict["writeInterval"] = 3600
of_case.system.boundaryProbes["writeInterval"] = 600

of_case._runCase()
