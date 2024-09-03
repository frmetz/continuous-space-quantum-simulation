import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import pennylane as qml

def periodic_distance(x, y, L):
    dis = x-y
    dis = L / 2.0 * jnp.sin(jnp.pi * dis / L)
    return dis ** 2

def pqc2d(params, inputs, N, L):
    dev = qml.device('default.qubit.jax', wires=2*N)
    @jax.jit
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        params_sq, params_ent, params_encoding, params_encoding_bias = params
        layers = params_ent.shape[0]
        inputs = inputs.reshape(N,2)

        for i in range(layers):

            for j in range(2):
                for n in range(N):
                    for m in range(n+1, N):
                        d = periodic_distance(inputs[n,j], inputs[m,j], L)
                        qml.IsingXX(params_encoding[i,j] * d + params_encoding_bias[i,j], wires = [n+N*j, m+N*j])

            for j in range(2):
                for n in range(N):
                    qml.Rot(*params_sq[i,j], wires = n+N*j)

            for j in range(2):
                for n in range(N):
                    for m in range(n+1, N):
                        qml.IsingYY(params_ent[i,j], wires = [n+N*j, m+N*j])

            for n in range(N):
                qml.IsingYY(params_ent[i,2], wires = [n, n+N])

            for n in range(N):
                for m in range(N):
                    if n != m:
                        qml.IsingYY(params_ent[i,3], wires = [n, m+N])

            for j in range(2):
                for n in range(N):
                    qml.Rot(*params_sq[i,j+2], wires = n+N*j)

        return [qml.expval(qml.PauliZ(i)) for i in range(2*N)]
    
    return qnode(params, inputs)

def pqc1d(params, inputs, N, L):
    dev = qml.device('default.qubit.jax', wires=N)
    @jax.jit
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        params_sq, params_ent, params_encoding, params_encoding_bias = params
        layers = params_ent.shape[0]
        inputs = inputs.reshape(N)

        for i in range(layers):

            for j in range(N):
                for k in range(j+1, N):
                    d = periodic_distance(inputs[j], inputs[k], L)
                    qml.IsingXX(params_encoding[i] * d + params_encoding_bias[i], wires = [j, k])

            for j in range(N):
                qml.Rot(*params_sq[i,0], wires = j)

            for j in range(N):
                for k in range(j+1, N):
                    qml.IsingYY(params_ent[i], wires = [j, k])

            for j in range(N):
                qml.Rot(*params_sq[i,1], wires = j)
        
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]
    
    return qnode(params, inputs)

pqc1d_vmap = jax.jit(jax.vmap(pqc1d, in_axes=(None, 0, None, None), out_axes=0), static_argnums=(2,3))
pqc2d_vmap = jax.jit(jax.vmap(pqc2d, in_axes=(None, 0, None, None), out_axes=0), static_argnums=(2,3))