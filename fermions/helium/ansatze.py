import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax.nn.initializers import ones

from flax import linen as nn
from flax.linen.initializers import normal

from netket.hilbert import ContinuousHilbert
from netket.utils.types import DType

import sys
sys.path.append('..')
from circuits import *
from slater_determinant import *

def periodic_distance(x, sdim, L):
    x = x.reshape(-1, sdim)
    n_particles = x.shape[0]
    dis = -x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :]
    dis = dis[jnp.triu_indices(n_particles, 1)]
    dis = L[jnp.newaxis, :] / 2.0 * jnp.sin(jnp.pi * dis / L[jnp.newaxis, :])
    return dis

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
    dtype = jnp.float64
    cusp_exponent: float = 5
    param_dtype: DType = jnp.float64

    def setup(self):
        nks = self.n_per_spin[0]
        if self.sdim == 1:
            nks = 3
        elif self.sdim == 2:
            nks = 5
        okvec = 2*np.pi/self.L*smallest_kvecs(jnp.eye(self.sdim), 5, nks)

        pw_basis = {}
        if self.n_per_spin[0] > 0: pw_basis['orbtials_up'] =  PlaneWaves(okvec, self.n_per_spin[0])
        if self.n_per_spin[1] > 0: pw_basis['orbtials_down'] = PlaneWaves(okvec, self.n_per_spin[1])
        self.orb_basis = pw_basis
        self.slater = LogSlaterDet(n_per_spin=self.n_per_spin, orbitals=self.orb_basis)

    @nn.compact
    def __call__(self, x):
        """ input: x = (..., N*d) """
        sha = x.shape
        N = sum(self.n_per_spin)
        x = x.reshape(-1, N, self.sdim)

        ###############################
        # Jastrow
        ###############################
        cusp = 0.0
        if self.jastrow:
            cusp_param = self.param("cusp", normal(stddev=1.0), (1,), self.param_dtype)
            L = jnp.array(self.hilbert.extent)
            sdim = L.size

            d = jax.vmap(periodic_distance, in_axes=(0, None, None))(x, sdim, L)
            dis = jnp.linalg.norm(d, axis=-1)

            if self.cusp_exponent is not None:
                cusp = -0.5 * jnp.sum(cusp_param / dis**self.cusp_exponent, axis=-1)

        ###############################
        # backflow
        ###############################
        if self.backflow:
            if sdim == 1:
                param_single_qubit = self.param("param_single_qubit", normal(stddev=self.param_std_quantum), (2, self.layers, 2, 3), self.param_dtype)
                param_two_qubit = self.param("param_two_qubit", normal(stddev=self.param_std_quantum), (2, self.layers, ), self.param_dtype)
                param_encoding = self.param("param_encoding", ones, (2, self.layers), self.param_dtype) * 0.5
                param_encoding_bias = self.param("param_encoding_bias", normal(stddev=self.param_std_quantum), (2, self.layers), self.param_dtype)
            elif sdim == 2:
                param_single_qubit = self.param("param_single_qubit", normal(stddev=self.param_std_quantum), (2, self.layers, 4, 3), self.param_dtype)
                param_two_qubit = self.param("param_two_qubit", normal(stddev=self.param_std_quantum), (2, self.layers, 4), self.param_dtype)
                param_encoding = self.param("param_encoding", ones, (2, self.layers, 2), self.param_dtype) * 0.5
                param_encoding_bias = self.param("param_encoding_bias", normal(stddev=self.param_std_quantum), (2, self.layers, 2), self.param_dtype)

            param_quantum = [[param_single_qubit[i], param_two_qubit[i], param_encoding[i], param_encoding_bias[i]] for i in range(2)]
            param_classical = self.param("param_classical", normal(stddev=self.param_std_classical), (2,2), self.param_dtype)
        
            if sdim == 1:
                pqc_out_real = jnp.array(pqc1d_vmap(param_quantum[0], x, N, self.L))
                pqc_out_imag = jnp.array(pqc1d_vmap(param_quantum[1], x, N, self.L))
            elif sdim == 2:
                pqc_out_real = jnp.array(pqc2d_vmap(param_quantum[0], x, N, self.L))
                pqc_out_imag = jnp.array(pqc2d_vmap(param_quantum[1], x, N, self.L))
            pqc_out_real = pqc_out_real.T.reshape(-1, sdim, N).transpose(0,2,1)
            pqc_out_imag = pqc_out_imag.T.reshape(-1, sdim, N).transpose(0,2,1)

            x = x + param_classical[0] * pqc_out_real + 1.j * param_classical[1] * pqc_out_imag # additive backflow

        ###############################
        # combine
        ###############################
        logpsi = self.slater(x) + cusp
        return logpsi.reshape(sha[0])