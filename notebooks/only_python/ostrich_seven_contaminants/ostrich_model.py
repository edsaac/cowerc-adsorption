#!/usr/bin/env python
from cowerc_adsorption import PhysicalParams, Simulation
import numpy as np


if __name__ == "__main__":
    ## Experiment constants
    lenght = 0.85
    pore_velocity = 9.645754126781533
    porosity = 0.5
    times_to_query = [
        10.0,
        23.47,
        54.23,
        114.13,
        147.46,
        295.44,
        394.16,
        512.94,
        743.66,
        982.9,
        1123.9,
        1315.38,
        1491.17,
        1646.43,
        1750.79,
        1979.98,
        2267.49,
        2408.92,
        2686.74,
        2916.87,
        3101.92,
        3233.11,
        3488.82,
        3657.06,
        3860.73,
        3943.17,
        4044.96,
        4199.84,
        4357.82,
        4487.45,
        4654.18,
        4764.58,
        4898.95,
        5032.16,
        5313.25,
        5454.69,
        5707.36,
    ]

    c_0 = [23.99683152, 12.48820903, 15.74782413, 8.95782909, 11.52679673, 12.04393965, 8.09707393]

    ## Read model parameters
    with open("./parameters.dat", "r") as f:
        args = f.readlines()
        sm = 10 ** float(args[0])
        k_ads = [10 ** float(arg) for arg in args[1:8]]
        k_des = [10 ** float(arg) for arg in args[8:]]

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
    sim.write_every = 100
    sim.cfl = 0.92
    sim.end_time = 6000
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
