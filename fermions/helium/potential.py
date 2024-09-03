import numpy as np
import jax.numpy as jnp

def minimum_distance(x, sdim, L):
    """Computes distances between particles using minimum image convention"""
    n_particles = x.shape[0] // sdim
    x = x.reshape(-1, sdim)

    distances = (-x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :])[
        jnp.triu_indices(n_particles, 1)
    ]
    distances = jnp.remainder(distances + L / 2.0, L) - L / 2.0

    if sdim == 2:
        distancesl0 = np.array([L,0]).reshape((1,-1)) - distances
        distances0l = np.array([0,L]).reshape((1,-1)) - distances
        distancesll = np.array([L,L]).reshape((1,-1)) - distances
        distances = jnp.concatenate([distances, distancesl0, distances0l, distancesll])

    dis = jnp.linalg.norm(distances, axis=1)

    if sdim == 1:
        dis2 = L - dis
        dis = jnp.concatenate([dis, dis2])

    return dis

def potential(x, sdim, L):
    """Compute Aziz potential for single sample x"""
    dis = minimum_distance(x, sdim, L)
    eps = 7.846373
    A = 0.544850 * 10**6
    alpha = 13.353384
    c6 = 1.37332412
    c8 = 0.4253785
    c10 = 0.178100
    D = 1.241314

    return jnp.sum(
        eps
        * (
            A * jnp.exp(-alpha * dis)
            - (c6 / dis**6 + c8 / dis**8 + c10 / dis**10)
            * jnp.where(dis < D, jnp.exp(-((D / dis - 1) ** 2)), 1.0)
        )
    )