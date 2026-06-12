markdown
# Open Simulation of Artificial Sun Library

Charged Hard Disk Molecular Dynamics Simulator with GPU Acceleration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview

GPU-accelerated simulator for **charged hard disk systems** in 2D. Combines:
- Event-Driven Molecular Dynamics (EDMD) for elastic collisions
- Time-Driven integration for long-range Coulomb interactions
- CUDA parallelization for GPU acceleration

## Quick Start

```bash
# Compile
nvcc -o sim mytask_kulong1019.cu -std=c++11 -arch=sm_60

# Run
./sim
Simulation Parameters (in main())
Parameter	Default	Description
N	4096	Number of particles
R	1.0	Particle radius (m)
vmod0	100	Initial speed (m/s)
boost	1000	Velocity boost
iter	5000000	Max iterations
Output Files
File	Description
x.txt, y.txt	Particle positions
vx.txt, vy.txt	Particle velocities
time.txt	Simulation time
log.csv	Trajectory data
info.csv	Parameters
f_historam of.csv	Statistics
Performance
N	CPU Time	GPU Time	Speedup
1024	0.45s	0.0053s	85x
2048	1.82s	0.021s	87x
4096	7.25s	0.048s	152x
Boundary Conditions
Preset 0: Horizontal walls + Vertical periodic + Opposing streams

Preset 1: Fully periodic + Random velocities

License
MIT License

Citation
bibtex
@article{pak2026cuda,
  title={A CUDA-based GPU-Accelerated Event-Driven Particle Method for Simulating a 2D Gas of Charged Hard Disks},
  author={Pak, Jun Yong and Son, Kum Chol and Nefedev, Konstantin V. and Pak, Hyok},
  year={2026}
}
Contact
Corresponding author: Konstantin V. Nefedev (nefedev.kv@dvfu.ru)
