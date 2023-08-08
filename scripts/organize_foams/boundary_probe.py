from dataclasses import dataclass
from pathlib import Path
import numpy as np
import xarray as xr

@dataclass(slots=True, frozen=True)
class probePoint:
    x: float
    y: float
    z: float

@dataclass
class boundaryProbe:
    """
    Assumes that the contents of postProcessing probes have been parsed already
    with pointFiles.sh
    """

    path_data: str | Path
    path_time: str | Path
    path_xyz: str | Path

    def __post_init__(self):
        for path in [self.path_data, self.path_time, self.path_xyz]:
            if isinstance(path, str):
                path = Path(path)

        ## Get the number of probed fields
        self.fields_names = self.path_data.stem.replace("points_", "").split("_")
        self.n_fields = len(self.fields_names)

        ## Get the number of probes
        self.get_probe_points()
        self.n_probes = len(self.probes_points)

        ## Get the timesteps
        self.get_times()

        ## Determine if vector or scalar data
        self.set_dimensionality()

        ## Load data arrays
        self.parse_data()

    def infer_number_data_columns(self):
        """Should be n_probes * n_fields * (1 or 3)"""
        with open(self.path_data) as f:
            first_line = f.readline().split()
            self.n_cols = len(first_line)

    def set_dimensionality(self):
        self.infer_number_data_columns()

        if self.n_cols == self.n_probes * self.n_fields:
            self.dimensionality = "scalar"
        elif self.n_cols == 3 * self.n_probes * self.n_fields:
            self.dimensionality = "vector"
        else:
            raise RuntimeError("Field is neither vector nor scalar.")

    def get_probe_points(self):
        xyz = np.loadtxt(self.path_xyz)
        self.probes_points = [probePoint(*coord) for coord in xyz]

    def get_times(self):
        self.list_of_times = np.loadtxt(self.path_time)

    def parse_data(self):
        full_data = np.loadtxt(self.path_data).T

        data = dict()

        if self.dimensionality == "scalar":
            field_names_for_parsing = self.fields_names
            dimension_number = 1

        elif self.dimensionality == "vector":
            field_names_for_parsing = [
                f"{field}{dim}" for field in self.fields_names for dim in [*"xyz"]
            ]
            dimension_number = 3

        for i, field in enumerate(field_names_for_parsing):
            data[field] = xr.DataArray(
                full_data[i :: self.n_fields * dimension_number],
                dims=("probe", "time"),
                coords={"probe": self.probes_points, "time": self.list_of_times},
            )

        self.array_data = xr.Dataset(
            data, coords={"time": self.list_of_times, "probes": self.probes_points}
        )
