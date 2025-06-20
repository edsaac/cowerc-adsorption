#!/usr/bin/env python
from cowerc_adsorption import PhysicalParams, Simulation
import numpy as np


if __name__ == "__main__":
    ## Experiment constants
    lenght = 2.54
    pore_velocity = 3.173
    porosity = 0.5

    times_to_query = [
        1.751000e01,
        1.986480e03,
        3.517240e03,
        7.074360e03,
        1.063148e04,
        1.544346e04,
        1.976560e04,
        2.348264e04,
        3.156864e04,
        3.606367e04,
        3.952138e04,
        4.441980e04,
        4.809362e04,
        5.284798e04,
        5.641374e04,
        6.116809e04,
        6.473386e04,
    ]

    c_0 = [2.66894928, 1.1912988, 1.82447442, 0.99863034, 1.09163878, 1.60000643]

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
    sim.end_time = 68_000
    sim.write_every = 2500
    sim.cfl = 0.85
    sim.solve()

    ## Query the btc from t
    t, btc = sim.btc
    results = [np.interp(times_to_query, t, c) for c in btc]
    results = np.array(results)

    if any(np.isnan(results.flatten())):
        raise ValueError("a NaN in the results")

    with open("log.dat", "w") as f:
        f.write(repr(p))
        f.write("\n")
        f.write(repr(sim))

    with open("results.dat", "w") as f:
        f.write("\n".join(map(lambda x: f"{x:.5E}", results.flatten())))
