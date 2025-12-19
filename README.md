
# PolymermcCPCMEM.jl

A standardized Julia framework for Polymer Monte Carlo (MC) simulations, designed for high-precision reconstruction of chromatin 3D structures. 

This repository implements both standard polymer simulations and **mediator-particle-mediated interactions**. It utilizes **Maximum Entropy Principle (MaxEnt)** optimization algorithms to accurately fit experimental Hi-C data.

## ✨ Key Features

*   **Hybrid Physical Models**: Supports both standard polymer MC simulations and explicit mediator-particle-based simulations (CPCMEM).
*   **MaxEnt Optimization**: 
    *   **Standard MaxEnt**: Optimization of interaction potentials for standard polymer models.
    *   **Mediator-based MaxEnt**: Optimization involving mediator particles to capture complex folding patterns.
*   **High-Precision Fitting**: Iterative optimization workflow to achieve precise agreement with experimental Hi-C contact maps.
*   **Parallel Computing**: Built on Julia's `Distributed` module for efficient multi-core sampling and ensemble averaging.

## 📂 Repository Structure

```text
PolymermcCPCMEM/
├── src/
│   └── PolymermcCPCMEM.jl    # Core module: Physics models, energy functions, and MC moves
├── scripts/
│   └── run-CPCMEM.jl         # Main script: Simulation setup and MaxEnt optimization loop
├── Project.toml              # Project dependencies
└── Manifest.toml             # Locked dependency versions
```

## 🚀 Quick Start

### 1. Prerequisites
Ensure **Julia** (v1.6 or later) is installed on your system.

### 2. Installation
Instantiate the environment to install required dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 3. Usage
Run the simulation and optimization pipeline. This can be executed locally or submitted to an HPC scheduler (e.g., Slurm).

```bash
# Run locally
julia --project=. scripts/run-CPCMEM.jl

# Or via Slurm (if headers are configured in the script)
sbatch scripts/run-CPCMEM.jl
```

## 📝 Citation
This code is associated with the manuscript: *[Insert Title Here]*.
