# LaughlinEdge
GPU-accelerated Monte-Carlo code to study edge excitations of Laughlin fractional quantum Hall droplets in weak anharmonic confinement.

Analyze the edge excitations of a Laughlin fractional quantum Hall droplet in a weak rotationally-symmetric anharmonic confinement by Monte-Carlo sampling on a single GPU.
The code allows the computation of the excitation spectrum, eigenvectors, edge dynamic structure factor, and the time evolution of an initially prepared Laughlin state under space-time dependent perturbations.


## Features
Main features:
1) Spectrum computation
        edgeMC::spectrum_compute
2) Edge dynamic structure factor
        edgeMC::dsf_compute
3) Edge spectral function
        (coming soon)
4) Matrix elements of an excitation
        edgeMC::excitationResponse_compute
5) Time evolution in response to an excitation
        timeEvolution::edgeDensityResponse_compute


Example usage snippets are provided in main.cu.


## Building
1) Define the system parameters.
   The parameters of the system (number of particles, filling fraction, confinement potential) are decided in
   system_parameter.cpp
   Compile and run this file to generate the header
   ./modules/sys_params.h
   which is used by the main code.

2) Select the computation by editing
   main.cu
   to choose the desired observable, type of external excitation, ...

3) Build the project
   mkdir build
   cd build
   cmake ..
   make

   and run it with
   ./LaughlinEdge

   Simulation results are written to: ./output/
