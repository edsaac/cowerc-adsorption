#!/usr/bin/env python
from cowerc_adsorption import PhysicalParams, Simulation
import numpy as np


if __name__ == "__main__":
    ## Experiment constants
    lenght = 2.18
    pore_velocity = 9.645754126781533
    porosity = 0.5
    times_to_query = [73.35, 302.37, 489.34, 709.59, 876.01, 1152.56, 1480.31, 1751.52, 2185.45, 2687.19, 2958.39]

    c_0 = [2.455, 1.187, 1.032, 2.287, 1.125, 1.225]

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
    sim.end_time = 3000
    sim.cfl = 0.80
    sim.solve()

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
