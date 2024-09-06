import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import pennylane as qml

def encoding_layer(p, x, gate, gate_range):
    N = x.shape[0]
    j = 0
    for n in range(1,gate_range):
        for i, wire in enumerate(range(N//2-(n+1)//2)):
            gate(p[i+j] * jnp.cos(x[wire]-x[wire+n]), wires=[wire,wire+n])
        for i, wire in enumerate(reversed(range(N//2+n//2,N))):
            gate(p[i+j] * jnp.cos(x[wire]-x[wire-n]), wires=[wire,wire-n])
        j += i+1

def variational_entangling_layer(p, gate, N, gate_range):
    j = 0
    for n in range(1,gate_range):
        for i, wire in enumerate(range(N//2-(n+1)//2)):
            gate(p[i+j], wires=[wire,wire+n])
        for i, wire in enumerate(reversed(range(N//2+n//2,N))):
            gate(p[i+j], wires=[wire,wire-n])
        j += i+1

def pqc1(params, inputs, N, gate_range=2):
    dev = qml.device('default.qubit.jax', wires=N)
    @jax.jit
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        params_sq, params_ent = params
        layers, _ = params_ent.shape

        for i in range(layers):

            for j in range(N):
                qml.RX(inputs[j], wires = j)

            for j, wire in enumerate(range(N//2)):
                qml.Rot(*params_sq[i,0,j,:], wires = wire)
            for j, wire in enumerate(reversed(range(N//2,N))):
                qml.Rot(*params_sq[i,0,j,:], wires = wire)

            variational_entangling_layer(params_ent[i], qml.IsingYY, N, gate_range=gate_range)

            for j, wire in enumerate(range(N//2)):
                qml.Rot(*params_sq[i,1,j,:], wires = wire)
            for j, wire in enumerate(reversed(range(N//2,N))):
                qml.Rot(*params_sq[i,1,j,:], wires = wire)

        return [qml.expval(qml.PauliZ(i)) for i in range(N)]
    return qnode(params, inputs)

def pqc2(params, inputs, N, gate_range=2):
    dev = qml.device('default.qubit.jax', wires=N)
    @jax.jit
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs):
        params_sq, params_ent, params_enc = params
        layers, _ = params_ent.shape

        for i in range(layers):

            encoding_layer(params_enc[i], inputs, qml.IsingXX, gate_range=N)

            for j, wire in enumerate(range(N//2)):
                qml.Rot(*params_sq[i,0,j,:], wires = wire)
            for j, wire in enumerate(reversed(range(N//2,N))):
                qml.Rot(*params_sq[i,0,j,:], wires = wire)

            variational_entangling_layer(params_ent[i], qml.IsingYY, N, gate_range=gate_range)

            for j, wire in enumerate(range(N//2)):
                qml.Rot(*params_sq[i,1,j,:], wires = wire)
            for j, wire in enumerate(reversed(range(N//2,N))):
                qml.Rot(*params_sq[i,1,j,:], wires = wire)

        return [qml.expval(qml.PauliZ(i)) for i in range(N)]
    return qnode(params, inputs)

pqc1_vmap = jax.jit(jax.vmap(pqc1, in_axes=(None, 0, None, None), out_axes=0), static_argnums=(2,3))
pqc2_vmap = jax.jit(jax.vmap(pqc2, in_axes=(None, 0, None, None), out_axes=0), static_argnums=(2,3))
