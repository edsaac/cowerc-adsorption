#!/usr/bin/env python
from cowerc_adsorption import PhysicalParams, Simulation
import numpy as np


if __name__ == "__main__":
    ## Experiment constants
    lenght = 0.85
    pore_velocity = 9.645754126781533
    porosity = 0.5
    times_to_query = [2.64, 8.09, 24.43, 68.33, 112.34, 204.95, 253.98, 299.91, 345.84, 437.7, 529.56, 601.92]

    c_0 = [166.5, 363.83, 168.33, 333.33, 178.5, 378.0]
    molecular_weight = np.array([214.04, 361.8, 314.05, 296.1, 414.1, 400.11])
    c_0 = c_0 / molecular_weight

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
    sim.end_time = 650
    sim.cfl = 0.85
    sim.solve()

    ## Query the btc from t
    t, btc = sim.btc
    results = [np.interp(times_to_query, t, c) for c in btc]
    results = np.array(results)

    with open("./log.dat", "w") as f:
        f.write(repr(p))
        f.write("\n")
        f.write(repr(sim))

    with open("./results.dat", "w") as f:
        f.write("\n".join(map(str, results.flatten())))
