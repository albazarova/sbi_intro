import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann
import pickle

mass_of_argon = 39.948  # amu

"""Simulation from https://pythoninchemistry.org/sim_and_scat/molecular_dynamics/build_an_md"""


def lj_force(r, epsilon, sigma):
    """
    Implementation of the Lennard-Jones potential
    to calculate the force of the interaction.

    Parameters
    ----------
    r: float
        Distance between two particles (Å)
    epsilon: float
        Potential energy at the equilibrium bond
        length (eV)
    sigma: float
        Distance at which the potential energy is
        zero (Å)

    Returns
    -------
    float
        Force of the van der Waals interaction (eV/Å)
    """
    return 48 * epsilon * np.power(sigma, 12) / np.power(
        r, 13
    ) - 24 * epsilon * np.power(sigma, 6) / np.power(r, 7)


def init_velocity(T, number_of_particles):
    """
    Initialise the velocities for a series of
    particles.

    Parameters
    ----------
    T: float
        Temperature of the system at
        initialisation (K)
    number_of_particles: int
        Number of particles in the system

    Returns
    -------
    ndarray of floats
        Initial velocities for a series of
        particles (eVs/Åamu)
    """
    R = np.random.rand(number_of_particles) - 0.5
    return R * np.sqrt(Boltzmann * T / (mass_of_argon * 1.602e-19))


def get_accelerations(positions, lj_params):
    """
    Calculate the acceleration on each particle
    as a  result of each other particle.
    N.B. We use the Python convention of
    numbering from 0.

    Parameters
    ----------
    positions: ndarray of floats
        The positions, in a single dimension,
        for all of the particles

    lj_params: tuple of 2 parameters
        The Lennard-Jhones potential parameters
        epsilon and sigma as a tuple (epsilon, sigma)
    Returns
    -------
    ndarray of floats
        The acceleration on each
        particle (eV/Åamu)
    """
    accel_x = np.zeros((positions.size, positions.size))
    for i in range(0, positions.size - 1):
        for j in range(i + 1, positions.size):
            r_x = positions[j] - positions[i]
            rmag = np.sqrt(r_x * r_x)
            force_scalar = lj_force(rmag, *lj_params)
            force_x = force_scalar * r_x / rmag
            accel_x[i, j] = force_x / mass_of_argon
            accel_x[j, i] = -force_x / mass_of_argon
    return np.sum(accel_x, axis=0)


def update_pos(x, v, a, dt, box_length):
    """
    Update the particle positions.

    Parameters
    ----------
    x: ndarray of floats
        The positions of the particles in a
        single dimension
    v: ndarray of floats
        The velocities of the particles in a
        single dimension
    a: ndarray of floats
        The accelerations of the particles in a
        single dimension
    dt: float
        The timestep length

    Returns
    -------
    ndarray of floats:
        New positions of the particles in a single
        dimension
    """
    out_of_bounds = np.abs(x) > box_length / 2.0
    v[out_of_bounds] *= -1
    x = x + v * dt + 0.5 * a * dt * dt
    return x, v


def update_velo(v, a, a1, dt):
    """
    Update the particle velocities.

    Parameters
    ----------
    v: ndarray of floats
        The velocities of the particles in a
        single dimension (eVs/Åamu)
    a: ndarray of floats
        The accelerations of the particles in a
        single dimension at the previous
        timestep (eV/Åamu)
    a1: ndarray of floats
        The accelerations of the particles in a
        single dimension at the current
        timestep (eV/Åamu)
    dt: float
        The timestep length

    Returns
    -------
    ndarray of floats:
        New velocities of the particles in a
        single dimension (eVs/Åamu)
    """
    return v + 0.5 * (a + a1) * dt


def run_md(
    dt, number_of_steps, x, initial_temp, epsilon, sigma, box_length=20, seed=None
):
    """
    Run a molecular dynamics (MD) simulation using the velocity Verlet algorithm.

    Parameters
    ----------
    dt : float
        The timestep length (s).
    number_of_steps : int
        Number of iterations in the simulation.
    x : ndarray of floats
        The initial positions of the particles in a single dimension (Å).
    initial_temp : float
        Temperature of the system at initialization (K).
    epsilon : float
        Depth of the potential well in the Lennard-Jones potential (energy units).
    sigma : float
        Finite distance at which the inter-particle potential is zero (Å).
    box_length : float, optional, default=20
        Length of the simulation box (Å).
    seed : int, optional, default=None
        Random seed for reproducibility.

    Returns
    -------
    positions : ndarray of floats
        The positions of all particles at each timestep of the simulation (Å).
        Shape: (number_of_steps, num_particles).
    """
    if seed:
        np.random.seed(seed)
    num_particles = x.shape[0]
    positions = np.zeros((number_of_steps, num_particles))
    v = init_velocity(initial_temp, num_particles)
    lj_params = (epsilon, sigma)
    a = get_accelerations(x, lj_params)
    for i in range(number_of_steps):
        x, v = update_pos(x, v, a, dt, box_length)
        a1 = get_accelerations(x, lj_params)
        v = update_velo(v, a, a1, dt)
        # v_cm = v.mean()  # they all weigh the same
        # v -= v_cm
        a = np.array(a1)
        positions[i, :] = x
    return positions


if __name__ == "__main__":
    box_length = 20
    x = np.linspace(-box_length / 2, box_length / 2, 6)[1:-1]
    lj_params = (0.0103, 3.4)
    temperature = 600
    t0 = 0.0
    dt = 0.2
    t_steps = 4000
    t = np.linspace(t0, t0 + (dt * t_steps), t_steps, endpoint=False)
    sim_pos = run_md(
        dt, t_steps, x, temperature, *lj_params, box_length=box_length, seed=None
    )

    # for i in range(sim_pos.shape[1]):
    #     plt.plot(t, sim_pos[:, i], ".", label="atom {}".format(i))
    # plt.xlabel(r"time (s)")
    # plt.ylabel(r"$x$-Position/Å")
    # plt.legend(frameon=False)
    # plt.ylim(-box_length / 2.0, box_length / 2.0)
    # plt.savefig("figures/example_free_trajectories.png")
    
    with open("../../data/observation_free.pkl", "wb") as pf:
        pickle.dump((t, sim_pos), pf)
