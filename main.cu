// !!! BEFORE THIS CODE IS COMPILED, you should generate a sys_config.h file containing a few of the system's parameters (such as number of particles and filling)

// Sketch of the idea
// Study the edge excitations of a Laughlin droplet (with ground state angular momentum L0)
// in a rotationally symmetric confinement potential V
// Computes preliminary data to extract the spectrum and eigenvectors
// H|dM,j> = E_{dM,j} |dM,j>
// Stuff are calculated by diagonalizing the Schrodinger equation: 
// The code implements power sum symmetric polynomials (PSSP) edge excitations |dM,A>. They are neither orthogonal for finite number of particles nor the correct eigenmodes
// The code returns all the confinement matrix elements <dM,A|V|dM,B> and the metric <dM,A|dM,B>
// performs exact-diagonalization
// and eventually rotates other observables (computed again via MC sampling of PSSP) onto the eigenvector basis

// The main module is 
// edge_montecarlo.h --> extracts energies, optionally (edge) dynamic structure factor, (edge) spectral function
// It can also compute the matrix elements of an excitation V and the edge densities so as to reconstruct the dynamics (H+V)\psi(t) = i \psi'(t): dynamics.h

// The matrix elements obtained from a given Monte-Carlo run can be saved by passing (true, #) to the _compute calls (default choice is false): 
// in this case the matrix elements of the iteration are saved as #.tsv. The files can be used to compute statistical errorbars
// the data from the given run are returned instead (without errorbars)

#include <iostream>
#include <stdio.h>
#include <vector>

#include "./modules/complex_numbers.h"
#include "./modules/tensors.h"
#include "./modules/integer_partitions.h"
#include "./modules/edge_montecarlo.h"
#include "./modules/dynamics.h"

#include "./modules/sys_params.h"

int main(){	
	
	// SPECTRUM CALCULATION SNIPPET 
	/*
	int angularMomentumSector = 3;

	edgeMC::monteCarloParameters simulationParameters; // Default parameters (number of samples etc)
	simulationParameters.setMCSamples(20000);

	std::vector<double> evals = edgeMC::spectrum_compute(simulationParameters, angularMomentumSector, true, 0);

	std::cout << "The spectrum within the L = " << angularMomentumSector << " sector: " << std::endl;
	for(auto e : evals){
		std::cout << "\tE = " << e << std::endl;
	}
	//*/
	
	// DSF CALCULATION SNIPPET
	/*
	int angularMomentumSector = 3;

	edgeMC::monteCarloParameters simulationParameters; // Default parameters (number of samples etc)
	simulationParameters.setMCSamples(200000);

	std::vector<std::pair<double,double>> e_dsf = edgeMC::dsf_compute(simulationParameters, angularMomentumSector, true, 0);
	std::cout << "The spectrum and edge dynamic structure factor within the L = " << angularMomentumSector << " sector: " << std::endl;
	for(auto elem : e_dsf){
		std::cout << "\tE = " << elem.first << "\t\tS = " << elem.second << "\n";
	}
	//*/

	// EXCITATION MATRIX ELEMENTS AND DYNAMICS SNIPPET
	/*
	// Define the excitation: V = f(t) U(r)
	// Notice this specific example carries two units on angular momentum
	auto spatial_profile = [] __device__ ( complex<float>* position ){
		double U = 0;

		for(int i=0; i<sys_params::particlesNumber; i++){
			U += pow( re( position[i] ) / sys_params::RCl, 2 );
		}

		return U;
	};

	// Define the relevant angular-momenta subspace whete the dynamics occurs
	vector<int> angular_momenta = {0, 2, 4, 6};

	// Initialize the parameters
	edgeMC::monteCarloParameters simulationParameters; // Default parameters (number of samples etc)
	simulationParameters.setMCSamples(50000);
	
	// Call the sampler
	edgeMC::excitationResponse_compute( simulationParameters, angular_momenta, spatial_profile, 0 );

	// Define the temporal profile of the excitation
	double TMax = 1000; // Final integration time
	double save_step = 0.5; // Time-steps at which data are saved (not the actual integration step)

	double excitation_strength = 0.1; // A simple Gaussian pulse
	double t0 = 25.;
	double tau = 5.;

	auto temporal_profile = [&] ( double t ){
		double z = (t-t0)/tau;
		return excitation_strength * std::exp( - z*z );
	};

	timeEvolution::edgeDensityResponse_compute( sys_params::particlesNumber, sys_params::inverseFilling, angular_momenta, temporal_profile, TMax, save_step ); // By default parameter imports data from 0.tsv

	//*/

	return 0;
}

