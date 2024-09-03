from typing import Any, Callable, Sequence, Tuple
import scipy
import numpy as np
import jax.numpy as jnp
from jax.nn.initializers import lecun_normal
import flax.linen as nn
from netket.utils import HashableArray
from netket.utils.types import DType

def smallest_kvecs(basis, n, m):
    dim = basis.shape[-1]
    r = np.arange(-n, (n + 1))

    vecs = np.array(np.meshgrid(*(dim * (r,)))).T.reshape(-1, dim)
    vecs = vecs[np.argsort(np.linalg.norm(vecs, axis=1))]
    return vecs[:m, :]

class PlaneWaves(nn.Module):
    """ create a linear combination of plane waves """
    kvecs: HashableArray
    n_per_spin: int
    init: Callable = lecun_normal()
    dtype: DType = float

    @nn.compact
    def __call__(self, x):
        """ linear combination of Plane Waves

        Args:
            x (jnp.ndarray): (..., D)
        """
        kx = jnp.einsum('kd,...d->...k', self.kvecs, x)
        res = jnp.exp(1j*kx)
        params = self.param('params', self.init, (self.kvecs.shape[0], self.n_per_spin), self.dtype)
        res = jnp.einsum('ki,...k->...i', jnp.abs(params), res)
        return res

class LogSlaterDet(nn.Module):
    n_per_spin: Tuple[int]
    orbitals: Sequence[Callable[..., Any]]
    add_signs: bool = True

    @nn.compact
    def __call__(self, x):
        """x: (n_samples, n_particles, s_dim)"""

        if not self.orbitals:
            raise ValueError(f'Empty LogSlaterDet module {self.name}.')

        assert x.ndim == 3, f"got {x.shape}"
        n_particles = float(sum(self.n_per_spin))

        mat = []
        i = 0
        for key, value in self.orbitals.items():
            o = value(x)
            mat.append(o)

        mat = jnp.concatenate(mat, axis=-1)

        assert mat.shape[-2] == mat.shape[-1]

        """mask to make sure that off diagonal blocks are zero"""
        xx = jnp.tile(jnp.array(self.n_per_spin[0] * [1] + self.n_per_spin[1] * [0])[:, None], (1, self.n_per_spin[0]))
        y = jnp.tile(jnp.array(self.n_per_spin[0] * [0] + self.n_per_spin[1] * [1])[:, None], (1, self.n_per_spin[1]))
        mask = jnp.concatenate((xx, y), axis=-1)

        mat = mask * mat
        assert mat.shape[-1] == mat.shape[-2], f"got {mat.shape}"

        signs, logslaterdet = jnp.linalg.slogdet(mat)

        log_norm = 0.5*np.log(scipy.special.factorial(n_particles))
        logslaterdet = logslaterdet - log_norm
        if self.add_signs:
            logslaterdet = logslaterdet + jnp.log(signs.astype(complex))

        return logslaterdet