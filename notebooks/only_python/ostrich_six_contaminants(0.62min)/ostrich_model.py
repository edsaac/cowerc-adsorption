#!/usr/bin/env python
from cowerc_adsorption import PhysicalParams, Simulation
import numpy as np


if __name__ == "__main__":
    ## Experiment constants
    lenght = 3.00
    pore_velocity = 9.646
    porosity = 0.5
    times_to_query = [107.89, 335.14, 953.29, 1363.83, 1870.79, 2713.93, 3334.83, 4216.99, 5137.76, 5943.13, 6841.82]

    c_0 = [1.605, 0.706, 1.436, 0.971, 1.061, 1.142]

    ## Read model parameters
    with open("./parameters.dat", "r") as f:
        args = f.readlines()
        sm = 10 ** float(args[0])
        k_ads = [10 ** float(arg) for arg in args[1:7]]
        k_des = [10 ** float(arg) for arg in args[7:]]

    p = PhysicalParams(
        L=lenght,
        v=pore_velocity,
        n=porosity,
        sm=sm,
        k_ads=k_ads,
        k_des=k_des,
        C_0=c_0,
    )

    sim = Simulation(**p.nondim)
    sim.end_time = 7000
    sim.cfl = 0.80
    sim.solve_v1()

    ## Query the btc from t
    t, btc = sim.btc
    results = [np.interp(times_to_query, t, c) for c in btc]
    results = np.array(results)

    if any(np.isnan(results.flatten())):
        raise ValueError("a NaN in the results")

    with open("./log.dat", "w") as f:
        f.write(repr(p))
        f.write("\n")
        f.write(repr(sim))

    with open("./results.dat", "w") as f:
        f.write("\n".join(map(str, results.flatten())))
