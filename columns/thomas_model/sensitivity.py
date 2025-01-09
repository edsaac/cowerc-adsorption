from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from multiprocessing import Pool


t = np.arange(0, 40.1, 0.1)


def linear_function(t: np.ndarray, a: float, b: float):
    return a * t + b


def wrapped(X):
    a, b = X
    return linear_function(t, a, b)


st.title("Sobol Sensitivity Analysis")


r"""
Consider the following model:

$$
\begin{equation}
        f(x) = a t + b
\end{equation}
$$

With parameters $a$ and $b$ and for $t \in (0, 40)$
"""

# Define the model inputs
names = ["a", "b"]

cols = st.columns(2)

with cols[0]:
    a_bounds = st.slider("$a$", -10.0, 10.0, step=0.1, value=(-1.0, 1.0))

with cols[1]:
    b_bounds = st.slider("$b$", -10.0, 10.0, step=0.1, value=(-10.0, 10.0))

problem = {
    "num_vars": len(names),
    "names": names,
    "bounds": [
        a_bounds,
        b_bounds,
    ],
}

# Generate samples
param_values = sample(problem, 516)

# Run model (example)
with Pool() as pool:
    Y = pool.map(wrapped, param_values)

# Perform analysis
ndY = np.array(Y).T
Si = [analyze(problem, y, print_to_console=False) for y in ndY]

# Plot
names = problem["names"]
first_order = np.array([s["S1"] for s in Si]).T
confidence = np.array([s["S1_conf"] for s in Si]).T

t_select = st.slider("$t$", t.min(), t.max(), t.mean(), step=0.1)

fig, ax = plt.subplots()
for name, si, conf in zip(names, first_order, confidence):
    ax.plot(t, si, label=name)
    ax.axvline(x=t_select)
    ax.fill_between(t, si - conf, si + conf, alpha=0.2)

ax.set_xlabel("$t$")
ax.set_ylabel("First-order sensitivity index $S_i$")
ax.legend()
st.pyplot(fig)

it = np.argmin(np.abs(t - t_select))

fig, axs = plt.subplots(
    1, 3, figsize=(10, 3), sharey=True, gridspec_kw={"wspace": 0.05}
)
for ax in axs:
    ax.set_ylim(0, 1.1)
Si[it].plot(ax=axs)
st.pyplot(fig)
