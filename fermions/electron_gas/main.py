import netket as nk
nk.config.netket_experimental_fft_autocorrelation = True

import time
import numpy as np
from matplotlib import pyplot as plt

from ansatze import *
from potential import *

N = 3 # number of particles
sdim = 2 # spatial dimension
rs = 200 # wigner seitz radius
if sdim == 3:
    L = ((3 / (4*np.pi*N))**(-1./sdim))
elif sdim == 2:
    L = (np.pi*N)**(1./sdim)

hilb = nk.hilbert.Particle(N=N, L=(L,)*sdim, pbc=True)
sab = nk.sampler.MetropolisGaussian(hilb, sigma=0.1, n_chains=16, n_sweeps=32)

potential = lambda x: ewald_coulomb(x, L, sdim, alpha=2.0, kmax=20)
epot = nk.operator.PotentialEnergy(hilb, potential)
ekin = nk.operator.KineticEnergy(hilb, mass=rs)
ha = (ekin + epot)

model = Ansatz(hilbert=hilb, n_per_spin=(N,0), sdim=sdim, jastrow=True, backflow=True, L=L, layers=1, param_std_quantum=1.0, param_std_classical=0.1, jas_cutoff=6)

vs = nk.vqs.MCState(sab, model, n_samples=1024, n_discard_per_chain=10, seed=12, sampler_seed=34)

op = nk.optimizer.Sgd(0.1)
sr = nk.optimizer.SR(solver=nk.optimizer.solver.svd)

gs = nk.VMC(ha, op, sab, variational_state=vs, preconditioner=sr)

def mycb(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    return True
log=nk.logging.RuntimeLog()

start_time = time.time()
gs.run(n_iter=100, callback=mycb, out=log)
print("Time taken: ", time.time() - start_time)

# plt.plot(log.data["Energy"]["iters"], log.data["Energy"]["Mean"]["real"])
plt.plot(log.data["Energy"]["iters"], log.data["Energy"]["Mean"].real)
plt.ylabel('Energy')
plt.xlabel('Iterations')
plt.savefig('energy.png')
