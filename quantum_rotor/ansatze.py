from jax import random
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import flax.linen as nn
from flax.linen.initializers import normal

from netket.utils.types import DType, Array, NNInitFunc

from circuits import *

def custom_init_ones(key, shape, dtype = jnp.float_):
    return jnp.ones(shape, dtype) + random.normal(key, shape, dtype) * 0.01

class PQC1(nn.Module):
    N: int
    layers: int
    gate_range: int
    param_dtype: DType = jnp.float64
    param_init_normal: NNInitFunc = normal(stddev=1.0)
    param_init_ones: NNInitFunc = custom_init_ones

    @nn.compact
    def __call__(self, x: Array):
        x = x.reshape((-1, self.N))

        param_single_qubit = self.param("param_single_qubit", self.param_init_normal, (self.layers, 2, self.N//2, 3), self.param_dtype) # divide by 2 because of reflection symmetry
        param_two_qubit = self.param("param_two_qubit", self.param_init_normal, (self.layers, (self.N//2)**2), self.param_dtype)
        param_quantum = [param_single_qubit, param_two_qubit]

        param_classical = self.param("param_classical", self.param_init_ones, (self.N//2,), self.param_dtype)
        classical_coefficients = jnp.concatenate((param_classical, param_classical[::-1])) # reflection symmetry

        pqc_out = jnp.array(pqc1_vmap(param_quantum, x, self.N, self.gate_range)).T

        logpsi = jnp.sum(classical_coefficients * pqc_out, axis=1)
        return logpsi

class PQC2(nn.Module):
    N: int
    layers: int
    gate_range: int
    param_dtype: DType = jnp.float64
    param_init_normal: NNInitFunc = normal(stddev=1.0)
    param_init_ones: NNInitFunc = custom_init_ones

    @nn.compact
    def __call__(self, x: Array):
        x = x.reshape((-1, self.N))

        param_single_qubit = self.param("param_single_qubit", self.param_init_normal, (self.layers, 2, self.N//2, 3), self.param_dtype) # divide by 2 because of reflection symmetry
        param_two_qubit = self.param("param_two_qubit", self.param_init_normal, (self.layers, (self.N//2)**2), self.param_dtype)
        param_encoding = self.param("param_encoding", self.param_init_ones, (self.layers, (self.N//2)**2), self.param_dtype)
        param_quantum = [param_single_qubit, param_two_qubit, param_encoding]

        param_classical = self.param("param_classical", self.param_init_ones, (self.N//2,), self.param_dtype)
        classical_coefficients = jnp.concatenate((param_classical, param_classical[::-1])) # reflection symmetry

        pqc_out = jnp.array(pqc2_vmap(param_quantum, x, self.N, self.gate_range)).T

        logpsi = jnp.sum(classical_coefficients * pqc_out, axis=1)
        return logpsi

class MeanField(nn.Module):
    N: int
    k_max: int = 1
    param_dtype: DType = jnp.float64
    param_init: NNInitFunc = normal(stddev=0.1)

    @nn.compact
    def __call__(self, x: Array):
        x = x.reshape((-1, self.N))

        param = self.param("param_mf", self.param_init, (self.k_max, 1, self.N//2), self.param_dtype)
        coeff = jnp.concatenate((param, param[:,:,::-1]), axis=2)

        logpsi = jnp.sum(coeff * jnp.array([jnp.cos(k*x) for k in range(1, self.k_max+1)]), axis=(0,2))
        return logpsi
    
class Jastrow(nn.Module):
    N: int
    k_max: int = 1
    n_max: int = 3
    param_dtype: DType = jnp.float64
    param_init: NNInitFunc = normal(stddev=0.1)
    params: dict = None

    def __hash__(self): return id(self)

    @nn.compact
    def __call__(self, x: Array):
        x = x.reshape((-1, self.N))

        if self.params is None:
            param = self.param("param_jas", self.param_init, (self.k_max, (self.N//2)**2), self.param_dtype)
        else:
            dummy_param = self.param("param", self.param_init, (1,), self.param_dtype)
            param = self.params['param_jas']

        j = 0
        y = []
        for n in range(1, self.n_max):
            for i, site in enumerate(range(self.N//2-(n+1)//2)):
                theta_diff = x[:,site]-x[:,site+n]
                res = jnp.sum(jnp.array([param[k-1,i+j] * jnp.cos(k*theta_diff) for k in range(1, self.k_max+1)]), axis=0)
                y.append(res)
            for i, site in enumerate(reversed(range(self.N//2+n//2,self.N))):
                theta_diff = x[:,site]-x[:,site-n]
                res = jnp.sum(jnp.array([param[k-1,i+j] * jnp.cos(k*theta_diff) for k in range(1, self.k_max+1)]), axis=0)
                y.append(res)
            j += i+1

        logpsi = jnp.sum(jnp.array(y), axis=0)

        return logpsi
    
class Hybrid(nn.Module):
    classical: Jastrow|MeanField = None
    quantum: PQC1|PQC2 = None

    def __hash__(self): return id(self)

    @nn.compact
    def __call__(self, x: Array):
        pqc = self.quantum(x)
        jastrow = self.classical(x)
        return pqc + jastrow