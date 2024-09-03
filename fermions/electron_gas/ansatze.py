import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)
from jax import numpy as jnp

from flax import linen as nn
from flax.linen.initializers import normal, ones

from netket.hilbert import ContinuousHilbert
from netket.utils.types import DType

import sys
sys.path.append('..')
from circuits import *
from slater_determinant import *

def minimum_distance(x, L):
    """Computes distances between particles using minimum image convention"""
    n_particles = x.shape[0]
    distances = (-x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :])[
        jnp.triu_indices(n_particles, 1)
    ]
    distances = jnp.remainder(distances + L / 2.0, L) - L / 2.0
    return distances

def j(x, L, c=2.):
    x = jnp.abs(x)
    return (x * (1 - c * (x / L) ** 3)) ** 2


class Ansatz(nn.Module):
    hilbert: ContinuousHilbert
    n_per_spin: tuple[int]
    sdim: int = 1
    jastrow: bool = True
    backflow: bool = True
    L: float = 1.
    layers: int = 1
    param_std_quantum: float = 0.1
    param_std_classical: float = 0.1
    jas_cutoff: float = 6
    param_dtype: DType = jnp.float64

    def setup(self):
        nks = 5
        okvec = 2*np.pi/self.L*smallest_kvecs(jnp.eye(self.sdim), 10, nks)

        pw_basis = {}
        if self.n_per_spin[0] > 0: pw_basis['orbtials_up'] =  PlaneWaves(okvec, self.n_per_spin[0])
        if self.n_per_spin[1] > 0: pw_basis['orbtials_down'] = PlaneWaves(okvec, self.n_per_spin[1])
        self.orb_basis = pw_basis
        self.slater = LogSlaterDet(n_per_spin=self.n_per_spin, orbitals=self.orb_basis)

    def __hash__(self): return id(self)

    @nn.compact
    def __call__(self, x):
        """ input: x = (..., N*d) """
        sha = x.shape
        N = sum(self.n_per_spin)
        x = x.reshape(-1, N, self.sdim)

        ###################################
        # Jastrow
        ###################################
        jastrow = 0.0
        if self.jastrow:
            j_param = self.param("jastrow", ones, (self.jas_cutoff,), self.param_dtype)

            L = jnp.array(self.hilbert.extent)
            sdim = L.size
            d = jax.vmap(minimum_distance, in_axes=(0, None))(x, L)

            jastrow = jnp.sum(jnp.array([j_param[n-1] * jnp.sum(j(d, L), axis=-1) ** (n/2) for n in range(1, self.jas_cutoff+1)]), axis=0)
            jastrow = jnp.sum(jastrow, axis=-1)

        ###################################
        # backflow
        ###################################
        if self.backflow:
            param_single_qubit = self.param("param_single_qubit", normal(stddev=self.param_std_quantum), (2, self.layers, 4, 3), self.param_dtype)
            param_two_qubit = self.param("param_two_qubit", normal(stddev=self.param_std_quantum), (2, self.layers, 4), self.param_dtype)
            param_encoding = self.param("param_encoding", ones, (2, self.layers, 2), self.param_dtype) * 0.5
            param_encoding_bias = self.param("param_encoding_bias", normal(stddev=self.param_std_quantum), (2, self.layers, 2), self.param_dtype)

            param_quantum = [[param_single_qubit[i], param_two_qubit[i], param_encoding[i], param_encoding_bias[i]] for i in range(2)]
            param_classical = self.param("param_classical", normal(stddev=self.param_std_classical), (2,2), self.param_dtype)

            pqc_out_real = jnp.array(pqc2d_vmap(param_quantum[0], x, N, self.L)).T.reshape(-1, sdim, N).T.reshape(-1, sdim, N).transpose(0,2,1)
            pqc_out_imag = jnp.array(pqc2d_vmap(param_quantum[1], x, N, self.L)).T.reshape(-1, sdim, N).T.reshape(-1, sdim, N).transpose(0,2,1)

            x = x + param_classical[0] * pqc_out_real + 1.j * param_classical[1] * pqc_out_imag # additive backflow

        ###############################
        # combine
        ###############################
        logpsi = self.slater(x) + jastrow
        return logpsi.reshape(sha[0])