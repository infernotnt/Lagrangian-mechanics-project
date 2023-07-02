import numpy as np
import scipy as sc
from scipy import integrate
import matplotlib.pyplot as plt
from sys import getsizeof
import tqdm
from textwrap import wrap
import os

import yaml
import pickle


def ode_stabilisation(y, t, Omega, Omega_squared, a, a_squared, R, g):
    # Uses constraint stabilisation method
    # y[0] = r,             ret_params[0] = \dot r
    # y[1] = \dot r,        ret_params[1] = \ddot r
    # y[2] = \theta         ret_params[2] = \dot \theta
    # y[3] = \dot \theta    ret_params[3] = \ddot \theta
    ret = np.empty(4)
    ret[0] = y[1]
    ret[1] = -2 * a * y[1] + a_squared * (R - y[0])

    ret[2] = y[3]
    ret[3] = (
        -2 * y[1] * y[3]
        + g * np.sin(y[2])
        + Omega_squared * y[0] * np.cos(y[2]) * np.sin(y[2])
    )
    ret[3] /= y[0]

    return ret


def ode_simplest(y, t, Omega, Omega_squared, a, a_squared, R, g):
    # This ode was aquired by not assuming the additional degree of freedom for r, in other words r=R was taken from the get-go
    ret = np.empty(4)
    ret[0] = 0
    ret[1] = 0

    ret[2] = y[3]
    ret[3] = Omega_squared * np.sin(y[2]) * np.cos(y[2]) + g / R * np.sin(y[2])

    return ret


def get_stable_theta_from_analytic_solution(Omega):
    temp = np.sqrt(GRAVITY / RING_R)

    if isinstance(Omega, np.ndarray) is False:
        return (
            np.pi
            if Omega < temp
            else np.pi - np.arccos(GRAVITY / np.square(Omega) / RING_R)
        )

    ret = np.empty(Omega.shape[0])
    for i, Omega in enumerate(Omega):
        ret[i] = (
            np.pi
            if Omega < temp
            else np.pi - np.arccos(GRAVITY / np.square(Omega) / RING_R)
        )

    return ret


def get_sol(ode_func, Omega, theta_init, time_view, theta_dot_init=0):
    func_params = (
        Omega,
        Omega**2,
        parameters["a"],
        parameters["a"] ** 2,
        RING_R,
        GRAVITY,
    )
    inits = [
        RING_R,
        #RING_R * 1.02,
        0,
        theta_init,
        theta_dot_init,
    ]
    sol = integrate.odeint(ode_func, inits, time_view, args=func_params)

    return sol.view()


def simulate_stable():

    theta_stable_resolution = (
        parameters["theta_stable_start"] - parameters["theta_stable_end"]
    ) / parameters["nr_theta_stable_samples"]

    theta_samples = np.linspace(
        parameters["theta_stable_start"],
        parameters["theta_stable_end"],
        parameters["nr_theta_stable_samples"],
    )

    loading = tqdm.trange(len(Omega_arr), desc="Simulating for stable thetas: ")
    stable_arr = [[], []]
    for i, Omega_curr in enumerate(Omega_arr):

        loading.update()

        theta_stable = None
        for j, theta_curr in enumerate(theta_samples):
            sol = get_sol(parameters["ode_used"], Omega_curr, theta_curr, t.view())

            spread = sol[:, 2].max() - sol[:, 2].min()
            avg = np.average(sol[:, 2])

            spread_is_small = spread < theta_stable_resolution * 1.1
            avg_is_theta = avg - theta_curr < 0.1

            if spread_is_small and avg_is_theta:
                theta_stable = theta_curr

        if theta_stable is not None:
            stable_arr[0].append(Omega_curr)
            stable_arr[1].append(theta_stable)
        else:
            print("no stable theta found for Omega=", Omega_curr)

    return stable_arr


RING_R = 1  # m
BALL_M = 0.1  # kg
GRAVITY = 9.81  # m/s^2

parameters = {}

# ------- Parameters

parameters["Omega_stable_start"] = 0
parameters["Omega_stable_end"] = np.sqrt(GRAVITY / RING_R) * 7

parameters["nr_Omega_stable"], parameters["nr_theta_stable_samples"] = 100, 300
parameters["theta_stable_start"], parameters["theta_stable_end"] = np.pi, np.pi * 0.499
parameters["time_start"], parameters["time_end"] = 0, 20
parameters["nr_time"] = 10**3

parameters["ode_used"] = ode_stabilisation
parameters["theta_init"] = np.pi * 0.99

parameters["a"] = 100
# 'a' controlls the 'force' at which the ball will be constrained to the bar

# -----------------

t = np.linspace(parameters["time_start"], parameters["time_end"], parameters["nr_time"])

Omega_arr = np.linspace(
    parameters["Omega_stable_start"],
    parameters["Omega_stable_end"],
    parameters["nr_Omega_stable"],
)

stable_arr = [[], []]

STABLE_PARAMETERS_FILE_NAME = "parameters.pickle"
STABLE_POINTS_FILE_NAME = "points.pickle"

stable_file_valid = True

if (os.path.exists(STABLE_PARAMETERS_FILE_NAME) is False) or (
    os.path.exists(STABLE_POINTS_FILE_NAME) is False
):
    stable_file_valid = False
else:
    with open(STABLE_PARAMETERS_FILE_NAME, "rb") as file:
        loaded_parameters = pickle.load(file)
    with open(STABLE_POINTS_FILE_NAME, "rb") as file:
        loaded_stable_arr = pickle.load(file)

    if loaded_parameters != parameters:
        stable_file_valid = False
    else:
        stable_arr = loaded_stable_arr

if stable_file_valid:
    print(
        "Files for saving simulations of stable points found, skipping the simulation"
    )
else:
    print(
        "Files for saving simulations of stable points with the same parameters NOT found"
    )
    stable_arr = simulate_stable()

    with open(STABLE_PARAMETERS_FILE_NAME, "wb") as file:
        pickle.dump(parameters, file)
    with open(STABLE_POINTS_FILE_NAME, "wb") as file:
        pickle.dump(stable_arr, file)

nrCalc1 = (
    parameters["nr_Omega_stable"]
    * parameters["nr_theta_stable_samples"]
    * parameters["nr_time"]
)
nrCalc2 = parameters["nr_Omega_stable"] * parameters["nr_theta_stable_samples"]
print(
    f"theta was calculated {nrCalc1:,}, times, in {nrCalc2:,} distinct simulations for the stable theta information"
)


# %%


figs = [None]
axs = [None]

figs[0], axs[0] = plt.subplots(1, 1)

axs[0].set_title(
    r"$\theta_{stabilno}$ u zavisnosti od ugaone brzine rotiranja obruča $\Omega$"
)
axs[0].axvline(
    np.sqrt(GRAVITY / RING_R),
    label=r"$\sqrt{\frac{g}{R}}$",
    color="red",
    linestyle="--",
)
axs[0].set_xlabel(r"$\Omega$ [rad/s]")
axs[0].set_ylabel(r"$\theta_{stabilno}$ [rad]")
axs[0].axhline(np.pi, label="dno obruča", color="black", linestyle="--")
axs[0].axhline(np.pi / 2, label="desna strana obruča", color="green", linestyle="--")

axs[0].plot(stable_arr[0], stable_arr[1], label="numericko resenje")
axs[0].plot(
    Omega_arr,
    get_stable_theta_from_analytic_solution(Omega_arr),
    label="analiticko resenje", linestyle="--"
)


# %%


two_Omegas = [
    np.sqrt(GRAVITY / RING_R) * 0.5,
    np.sqrt(GRAVITY / RING_R) * 1.5,
]  # One Omega smaller than the critical Omega and one greater
two_theta_inits = [
    get_stable_theta_from_analytic_solution(Omega) * 1.05 for Omega in two_Omegas
]
two_labels = [
    "$\Omega="
    + str(Omega)[:4]
    + "$ rad/s, "
    + r"$\theta_{stabilno} = "
    + str(get_stable_theta_from_analytic_solution(Omega))[:4]
    + r"$ rad, $\theta_{init} = "
    + str(theta_init)[:4]
    + "$ rad"
    for theta_init, Omega in zip(two_theta_inits, two_Omegas)
]

two_sol = [
    get_sol(parameters["ode_used"], Omega, theta, t.view())
    for Omega, theta in zip(two_Omegas, two_theta_inits)
]

# Draw theta(t)
figs.append(None)
axs.append(None)
figs[-1], axs[-1] = plt.subplots(1, 1)
title = "\n".join(
    wrap(
        r"Ugao otklona kuglice $\Delta \theta$ (u odnosu na ispitivani ravnotežni položaj) u zavisnosti od vremena $t$",
        60,
    )
)
axs[-1].set_title(title)
axs[-1].set_xlabel(r"$t$ [s]")
axs[-1].set_ylabel(r"$\Delta \theta$ [rad]")
for i, j in enumerate([len(figs) - 1] * 2):
    axs[j].plot(
        t,
        two_sol[i][:, 2] - get_stable_theta_from_analytic_solution(two_Omegas[i]),
        label=two_labels[i],
    )


# Draw the trajectory
figs.append(None)
axs.append(None)
figs[-1], axs[-1] = plt.subplots(1, 1)
title = "\n".join(
    wrap(
        r"Trajektorija kuglice",
        60,
    )
)
axs[-1].set_title(title)
axs[-1].set_xlabel(r"$x$ [m]")
axs[-1].set_ylabel(r"$y$ [m]")
angles = np.linspace(
    0, np.pi * 2, 1000
)  # angle from the x axis like its usualy found in math
axs[-1].plot(
    np.cos(angles) * RING_R,
    (np.sin(angles) + 1) * RING_R,
    label="obruč",
    color="gray",
    linestyle="--",
)
for i, j in enumerate([len(figs) - 1] * 2):
    x = np.sin(two_sol[i][:, 2]) * RING_R
    y = (np.cos(two_sol[i][:, 2]) + 1) * RING_R
    axs[j].plot(x, y, label=two_labels[i])
    # axs[j].set_aspect("equal", adjustable="box")


# Draw the phase diagram
figs.append(None)
axs.append(None)
figs[-1], axs[-1] = plt.subplots(1, 1)
title = "\n".join(
    wrap(
        r"Zavisnost ugaone brzine $\dot \theta$ kuglice od ugla njenog otklona (u odnosu na ravnotežni položaj) $\Delta \theta$",
        60,
    )
)
axs[-1].set_title(title)
axs[-1].set_xlabel(r"$\Delta \theta$ [rad]")
axs[-1].set_ylabel(r"$\dot \theta$ [rad/s]")
for i, j in enumerate([len(figs) - 1] * 2):
    axs[j].plot(
        two_sol[i][:, 2] - get_stable_theta_from_analytic_solution(two_Omegas[i]),
        # two_sol[:, 2],
        two_sol[i][:, 3],
        label=two_labels[i],
    )


# Draw the force due to the R=r "constraint"
figs.append(None)
axs.append(None)
figs[-1], axs[-1] = plt.subplots(1, 1)
title = "\n".join(
    wrap(
        r"Sila ogranicenja $F_{const}$ u zavisnosti od vremena $t$",
        60,
    )
)
axs[-1].set_title(title)
axs[-1].set_xlabel(r"$t$ [s]")
axs[-1].set_ylabel(r"$F_{const}(t)$ [N]")
for i, j in enumerate([len(figs) - 1] * 2):
    f = (
        BALL_M * GRAVITY * np.cos(two_sol[i][:, 2])
        - BALL_M * RING_R * np.square(two_sol[i][:, 3])
        - BALL_M * RING_R * np.square(np.sin(two_sol[i][:, 2])) * two_Omegas[i]
    )
    r_dot_dot = -2 * parameters["a"] * two_sol[i][:, 1] + np.square(parameters["a"]) * (
        RING_R - two_sol[i][:, 0]
    )  # this is from the constraint stabilisation formula
    f = f + BALL_M * r_dot_dot
    axs[j].plot(t, f, label=two_labels[i])


# Draw the constraint function stabilizing over time
figs.append(None)
axs.append(None)
figs[-1], axs[-1] = plt.subplots(1, 1)
title = "\n".join(
    wrap(
        r"Funkcija ograničenja $f(r)=r-R$ u zavisnosti od vremena $t$",
        60,
    )
)
axs[-1].set_title(title)
axs[-1].set_xlabel(r"$t$ [s]")
axs[-1].set_ylabel(r"$f(t)$ [m]")
for i, j in enumerate([len(figs) - 1] * 2):
    axs[j].plot(t, two_sol[i][:, 0] - RING_R, label=two_labels[i])


# Draw and calcualte the movement of the non stable case
Omega_unstable = np.sqrt(GRAVITY / RING_R) * 20
theta_init_unstable = np.pi * 0.2
#theta_dot_init_unstable = Omega_unstable * 0.5886  # Weird case 1 where it goes in circle forever
#theta_dot_init_unstable = Omega_unstable * 0.58855  # Weird case 1 where it goes from top to top without ever going over
#theta_dot_init_unstable = Omega_unstable * 0.5805 # 3
theta_dot_init_unstable = Omega_unstable * 0.580038
t_unstable = np.linspace(0, 2, 10**5)
sol_unstable = get_sol(
    parameters["ode_used"],
    Omega_unstable,
    theta_init_unstable,
    t_unstable.view(),
    theta_dot_init_unstable,
)
figs.append(None)
axs.append(None)
figs[-1], axs[-1] = plt.subplots(1, 1)
title = "\n".join(
    wrap(
        r"Zavisnost $\theta$ i $\dot \theta$ od vremena $t$ u nestabilnom slucaju",
        60,
    )
)
axs[-1].set_xlabel(r"$t$ [s]")
axs[-1].set_ylabel(r"$\theta$[rad], $\dot \theta$ skalirano")
axs[-1].axhline(np.pi, label="dno obruča", color="black", linestyle="--")
axs[-1].axhline(np.pi / 2, label="desna strana obruča", color="green", linestyle="--")
text = str(
    r"Slučaj sa $\theta_{init}="
    + str(theta_init_unstable)[:4]
    + r"$ rad, $\dot \theta_{init} = "
    + str(theta_dot_init_unstable)[:4]
    + r" rad/s, i $\Omega="
    + str(Omega_unstable)[:4]
    + r"$ rad/s"
)
text = wrap(text, 60)
text = "\n".join(text)
figs[-1].text(0.5, -0.02, text, wrap=True, horizontalalignment="center", fontsize=12)
axs[-1].plot(t_unstable, sol_unstable[:, 2], label=r"$\theta$")
scale = (np.max(sol_unstable[:, 2]) - np.min(sol_unstable[:, 2])) / (
    np.max(sol_unstable[:, 3]) - np.min(sol_unstable[:, 3])
)

axs[-1].plot(t_unstable, sol_unstable[:, 3] * scale, label=r"$\dot \theta$")


for axes in axs:
    axes.grid(color="gray", linestyle="--")
    axes.legend()

for ax, fig in zip(axs, figs):
    # fig.tight_layout()
    fig.show()

parameters["a"] = input()
