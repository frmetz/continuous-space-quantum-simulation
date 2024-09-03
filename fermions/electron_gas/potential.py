from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnums=(1,))
def smallest_vecs(basis, n):
    dim = basis.shape[-1]
    r = np.arange(-n, (n+1))
    vecs = jnp.array(np.meshgrid(*(dim*(r,)))).T.reshape(-1, dim)
    return vecs

def ewald_coulomb(x, L, sdim, alpha, kmax=20, cutoff=None):
    if cutoff is None:
        cutoff = 2.*L
    n_particles = x.shape[0] // sdim
    x = x.reshape(-1, n_particles, sdim)

    dists = x[..., :, jnp.newaxis, :] - x[..., jnp.newaxis, :, :]

    def real_sum(dists):
        mask = 1 - jnp.eye(n_particles)
        d = jnp.linalg.norm(dists, axis=-1)

        # apply erfc to distances, but keep zero on diagonal
        res1 = jnp.where(mask[jnp.newaxis, :], jax.scipy.special.erfc(alpha * d) / d, 0).reshape(-1)

        Rn = smallest_vecs(jnp.eye(sdim), 2) * L
        temp = jnp.linalg.norm(Rn, axis=-1)
        Rn = jnp.where(temp[:, jnp.newaxis] <= cutoff, Rn, 0)
        temp = jnp.linalg.norm(Rn, axis=-1)
        d = jnp.linalg.norm(dists[..., jnp.newaxis, :] + Rn, axis=-1)
        res2 = jnp.where(temp == 0, 0, jax.scipy.special.erfc(alpha * d) / d).reshape(-1)

        res = jnp.sum(res1) + jnp.sum(res2)
        return res

    def recip_sum(x):
        dim = x.shape[-1]
        V = L ** dim

        Gm = smallest_vecs(jnp.eye(dim), kmax) / L
        G = jnp.linalg.norm(Gm, axis=-1, keepdims=True)
        Gm = jnp.where(G <= kmax/L, Gm, 0)
        G = jnp.linalg.norm(Gm, axis=-1)
        G2 = (G ** 2).reshape(-1)

        if dim == 3:
            t1 = jnp.where(G2 == 0, 0, jnp.exp(-G2 * jnp.pi ** 2 / (alpha ** 2)) / G2)
            prefactor = 1 / (V * jnp.pi)
        elif dim == 2:
            t1 = jnp.where(G2 == 0, 0, jax.scipy.special.erfc(jnp.pi * G / alpha ) / G)
            prefactor = 1 / V

        # structure factor
        S = jnp.sum(jnp.where(G2 == 0, 0, jnp.exp(1j * 2 * jnp.pi * jnp.einsum('...ij,kj->...ik', x, Gm))), axis=-2).reshape(-1)
        t2 = jnp.abs(S) ** 2

        res = prefactor * jnp.sum(t1 * t2)
        return res

    def self_energy():
        return -n_particles * alpha / jnp.sqrt(jnp.pi)

    def constant():
        V = L ** sdim
        if sdim == 3:
            return -n_particles ** 2 * jnp.pi / (2 * alpha ** 2 * V)
        elif sdim == 2:
            return -n_particles ** 2 * jnp.sqrt(jnp.pi) / (alpha * V)

    ereal = real_sum(dists)
    erec = recip_sum(x)
    eself = self_energy()
    const = constant()

    return (0.5*ereal + 0.5*erec + eself + const)