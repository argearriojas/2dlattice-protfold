# Protein folding on 2D Lattice model with MJ interaction potentials

This work is based on these previously published manuscripts:
- [(2012 Perdomo-Ortiz) Finding low-energy conformations of lattice protein models by quantum annealing](https://www.nature.com/articles/srep00571)
- [(2013  Babbush) Construction of Energy Functions for Lattice Heteropolymer Models: A Case Study in Constraint Satisfaction Programming and Adiabatic Quantum Optimization](https://arxiv.org/abs/1211.3422)

This project presents two implementations for solving the problem of protein folding on a 2D Lattice model:
- Simulated Annealing Monte Carlo simulation: i-th amino acid positions are simulated as random walks relative to previous residue
- QBit representation using Turn Encoding of Self-Avoiding Walks: The objective energy function has been constructed
    - Simulated annealing implementation on conventional CPU
    - Quantum annealing on QPU

## Results

- Protein folding simulated annealing
    - [6 amino acid](https://htmlpreview.github.io/?https://github.com/argearriojas/2dlattice-protfold/blob/master/2DLattice_MJ_SimulatedAnnealing/2DLattice_MJ_SimulatedAnnealing_L06.html)
    - [10 amino acid](https://htmlpreview.github.io/?https://github.com/argearriojas/2dlattice-protfold/blob/master/2DLattice_MJ_SimulatedAnnealing/2DLattice_MJ_SimulatedAnnealing_L10.html)
- Protein folding, turn ancilla quantum annealing model, simulated annealing
    - [6 amino acid](https://htmlpreview.github.io/?https://github.com/argearriojas/2dlattice-protfold/blob/master/2DLattice_MJ_SimulatedAncillaEncoding/2DLattice_MJ_SimulatedAncillaEncoding_L06.html)
- Protein folding, turn ancilla quantum annealing model, use QPU
    - [6 amino acid](https://htmlpreview.github.io/?https://github.com/argearriojas/2dlattice-protfold/blob/master/2DLattice_MJ_QPUAncillaEncoding/2DLattice_MJ_QPUAncillaEncoding_L06.html)
