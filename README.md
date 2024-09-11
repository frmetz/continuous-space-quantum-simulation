# Simulating continuous-space systems with quantum-classical wave functions

This repository contains the code and data for the corresponding preprint article [arXiv:2409.06415](https://arxiv.org/abs/2409.06415).

Most non-relativistic interacting quantum many-body systems, such as atomic and molecular ensembles or materials, are naturally described in terms of continuous-space Hamiltonians. The simulation of their ground-state properties on digital quantum computers is challenging because current algorithms require discretization, which usually amounts to choosing a finite basis set, inevitably introducing errors. In this work, we propose an alternative, discretization-free approach that combines classical and quantum resources in a global variational ansatz, optimized using the framework of variational Monte Carlo. We introduce both purely quantum as well as hybrid quantum-classical ansatze and benchmark them on three paradigmatic continuous-space systems that are either very challenging or beyond the reach of current quantum approaches: the one-dimensional quantum rotor model, a system of Helium-3 particles in one and two dimensions, and the two-dimensional homogeneous electron gas. We embed relevant constraints such as the antisymmetry of fermionic wave functions directly into the ansatz. Many-body correlations are introduced via backflow transformations represented by parameterized quantum circuits. We demonstrate that the accuracy of the simulation can be systematically improved by increasing the number of circuit parameters and study the effects of shot noise. Furthermore, we show that the hybrid ansatz improves the ground-state energies obtained using the purely classical wave function.


## Content
Each folder ([quantum_rotor](quantum_rotor), [helium](fermions/helium), [electron_gas](fermions/electron_gas)) contains the code and data needed to reproduce the results for the respective systems shown in the paper. The notebooks `figures.ipynb` contain the code for generating the plots (according to their figure numbers in the article). The data is stored in the correspinding `data/` folder.

The code is structured in the following way:
* `main.py` and `main.ipynb` contain the code needed for running the VMC optimization. The jupyter notebook [main.ipynb](quantum_rotor/main.ipynb) explains the steps in detail. The `main.py` scripts of the fermionic systems follow the same logic.
* `ansatze.py` contain the wave function ansatze defined as `flax.linen` modules.
* `circuits.py` defines the PQCs used in the wave function ansatze.

## Requirements

The code requires
* Python (3.11.5)
* [NetKet](https://netket.readthedocs.io/en/latest/index.html) (3.9.2)
* [PennyLane](https://pennylane.ai/) (0.32.0)
* matplotlib (3.8.0)
* [Jupyter Notebook](https://jupyter.org/)

## Citation

If you use our code/models for your research, consider citing our paper.
```
@misc{metz2024,
      title={Simulating continuous-space systems with quantum-classical wave functions}, 
      author={Friederike Metz and Gabriel Pescia and Giuseppe Carleo},
      year={2024},
      eprint={2409.06415},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2409.06415}, 
}
```

