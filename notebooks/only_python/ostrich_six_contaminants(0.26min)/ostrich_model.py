#!/usr/bin/env python
from cowerc_adsorption import PhysicalParams, Simulation
import numpy as np


if __name__ == "__main__":
    ## Experiment constants
    lenght = 1.28
    pore_velocity = 9.645754126781533
    porosity = 0.5
    times_to_query = [22.87, 68.62, 160.4, 239.15, 331.8, 464.02, 569.34, 714.1, 855.25, 952.2]

    c_0 = [0.7615399, 0.61912659, 1.26731412, 0.88821344, 1.04322627, 0.99472645]

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
    sim.end_time = 980
    sim.cfl = 0.90
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
