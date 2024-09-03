import netket as nk
nk.config.netket_experimental_fft_autocorrelation = True

import time
from matplotlib import pyplot as plt

from ansatze import *
from potential import *

sdim = 1 # spatial dimension
N = 3 # number of particles
if sdim == 1:
    d = 0.3  # density
elif sdim == 2:
    d = 0.03
rm = 2.9673  # Angstrom
L = (N / d)**(1./sdim) / rm  # size of box

hilb = nk.hilbert.Particle(N=N, L=(L,)*sdim, pbc=True)
sab = nk.sampler.MetropolisGaussian(hilb, sigma=0.1, n_chains=16, n_sweeps=32)

ekin = nk.operator.KineticEnergy(hilb, mass=1.0)
pot = nk.operator.PotentialEnergy(hilb, lambda x: potential(x, sdim, L))
ha = ekin + pot

model = Ansatz(hilbert=hilb, n_per_spin=(N,0), sdim=sdim, jastrow=True, backflow=True, L=L, cusp_exponent=5, layers=1, param_std_quantum=1.0, param_std_classical=0.1)

vs = nk.vqs.MCState(sab, model, n_samples=1024, n_discard_per_chain=10, seed=12, sampler_seed=34)

op = nk.optimizer.Sgd(0.0001)
sr = nk.optimizer.SR(solver=nk.optimizer.solver.svd)

gs = nk.VMC(ha, op, sab, variational_state=vs, preconditioner=sr)

def mycb(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    return True
log=nk.logging.RuntimeLog()

start_time = time.time()
gs.run(n_iter=200, callback=mycb, out=log)
print("Time taken: ", time.time() - start_time)

# plt.plot(log.data["Energy"]["iters"], log.data["Energy"]["Mean"]["real"])
plt.plot(log.data["Energy"]["iters"], log.data["Energy"]["Mean"].real)
plt.ylabel('Energy')
plt.xlabel('Iterations')
plt.savefig('energy.png')