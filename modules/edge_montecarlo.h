// Here all the relevant matrix elements are sampled
#ifndef EDGEMC_H
#define EDGEMC_H

#include <vector>

#include "./complex_numbers.h"
#include "./tensors.h"
#include "./integer_partitions.h"
#include "./sys_params.h"

namespace edgeMC{	
	using namespace iparts;

	// A structure for the results
	struct results{
		float* acceptance; // Stores the measured acceptance in each block
		
		// SPECTRUM
		complex<double>* fHg_ff; // Stores Σ (Vconf)g/f, computed with Markov chain Monte Carlo by generating configurations distributed as |f|²/<f|f>
		complex<double>* fg_ff; // Σ g/f, sampled according to |f|²/<f|f>
		double* gg_ff; // Σ |g/f|², sampled according to |f|²/<f|f>
		
		// DYNAMIC STRUCTURE FACTOR
		complex<double>* fVCg_ff; // Σ Cos(θ)g/f, sampled according to |f|²/<f|f>
		complex<double>* fVSg_ff; // Σ Sin(θ) g/f, sampled according to |f|²/<f|f>
		
		// SPECTRAL FUNCTION
		
		// EXCITATION POTENTIAL for the dynamics
		complex<double>* fUg_ff; // Σ (Uexc)g/f, sampled according to |f|²/<f|f>
		
		// DENSITIES (as Fourier transform, to compress information)
		complex<double>* r_ct; // Angular Fourier transform of the 2D edge-density operator  <f|ρ(r, θ)|g>/<f|f>
		complex<double>* r_st; 
	};

	// Check for CUDA errors
	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	__host__ __device__ inline void gpuAssert(const cudaError_t &code, const char *file, const int &line){
		if(code != cudaSuccess){
			printf("\nGPUassert - CUDA returned error code: %s %s %d (error code %d)\n", cudaGetErrorString(code), file, line, code);
		}
	}	

	// Parameters for the MCMC
	class monteCarloParameters{
		public:
			int maxDegree; // the largest k appearing in the polynomials \sum_i z_i^k
			int extendedMaxDegree; // = maxDegree + 1
			int subspaceDimension; // the dimension of the edge Hilbert space

			int injectedAngularMomentum; // The injected angular momentum (if only one)

			int *injectedAngularMomenta; // An array of angular momenta
			int numberOfSectors; // The length of the array

			int mesh_points_number; // The number of mesh points
			float dr; // The r spacing

			// Monte Carlo step parameters
			float randomStep; // The step of the Markov-Chain Monte Carlo			
			uint burnInMoves; // The number of burn-in moves to thermalize the state	
			uint MCSamples; // The number of Monte-Carlo global moves (per single thread)
			
			uint threadsPerBlock; // the number of threads per block
			uint nBlocks; // the number of blocks
			uint gridSize; // the size of the grid
			size_t totalSamplesNumber; // the total number of MC global moves

			// Acceptance bisection parameters
			uint targetAcceptanceMoves; // Number of samples (per thread) to bisect the empirical acceptance to the target value
			float targetAcceptance; // The ideal value for the acceptance
			float acceptance_tol; // The tolerance
						
			// Constructor for initializing parameters
			__host__ monteCarloParameters();

			// MC parameters
			__host__ void setEdgeSpaceInfo(int maxPolyDegree, int subDim);

			 // Manually adjust parameters
			 __host__ void setMesh(int new_mesh_points_number);
			 __host__ void setBurnIn(uint newBurnInMoves);
			 __host__ void setMCSamples(uint newMCSamples);
			 __host__ void setGPU_threads_blocks(uint newThreadsPerBlock, uint newNBlocks);
			 __host__ void setAcceptanceBisectionParams(uint newTargetAcceptanceMoves, float newTargetAcceptance, float newAcceptance_tol);
			 
			 __host__ __device__ void setRandomStep(float newRandomStep);
	};

	// Markov-Chain Monte Carlo
	class montecarlo{
		public:
			// Initialization and "destructor"
			__device__ montecarlo(	curandState* state, // the states for random number generation
									int thread_id, // the id of the given thread
									monteCarloParameters &parameters, // the MC parameters
									integer_partitions::partition* dev_partitions, // all the partitions
									int samplingEdgeState, // the partition that is used for sampling
									bool mc_step_test = false // defaulted to false. If true, only a small thermalization is done: used to bisect the good Markov-Chain Monte-Carlo coordinate step
								 );
								 
			__device__ void cleanup(); // Cleans up dynamically allocated memory
			__device__ curandState stateRelease(); // Returns the cuRAND local state

			__device__ float burnin_acceptance_get() const; // Returns the acceptance after burn-in

			//////////////
			// "Samplers"
			__device__ void hamiltonian_sampler( results *dev_results, int samplingEdgeState ); // Compute the matrix elements of H (and the metric) only
			__device__ void edge_dsf_sampler( results *dev_results, int samplingEdgeState ); // Additionally samples the edge dynamic structure factor

			template<typename Func>
			__device__ void excitationResponse_sampler( results *dev_results, int samplingEdgeState, Func excitation_potential ); // Additionally samples an external potential (to study nonlinear dynamics)

		private:
			////////////////////////
			// Private variables
			
			curandState localState; // Local state for the random numbers
			int tid; // Thread identifier
			
			monteCarloParameters params; // Monte-Carlo parameters

			complex<float> position[sys_params::particlesNumber]; // the particle positions, as z=x+iy			
			complex<float> scaled_position[sys_params::particlesNumber]; // the same positions, but scaled down
			complex<double> position_powers[sys_params::particlesNumber]; // the same positions, but stores powers. Used as a temporary array
			float position_norm[sys_params::particlesNumber]; // the norms of the positions (squared distance from the origin)
			float angle[sys_params::particlesNumber]; // the angle associated to the position
			
			complex<double> current_ek[sys_params::ansatzMaxPartitionSize]; // the current values of e_k = \sum_k z_i^k -- only the k needed for the sampling partition (they get updated more frequently)
			complex<double> proposed_ek[sys_params::ansatzMaxPartitionSize]; // the proposed values of e_k (after a single particle update)
			
			complex<double> ek_list[sys_params::ansatzExtendedMaxDegree]; // the current values of e_k. All the possible k.
			complex<double> ekr_list[sys_params::ansatzExtendedMaxDegree*sys_params::ansatzExtendedMaxDegree]; // the powers e_k^r. All the possible r. The matrix is flattened. ekr_list[k + dim(k) * r] = e_k^r
			
			int partition_size; // The dimension of the sampling partition. e.g. {2,2,2,1,1,1,1} -> [2x3,1x4] has dimension 2
			integer_partitions::partition_element* sampling_partition; // A pointer to the sampling partition, in compressed form
			integer_partitions::partition* device_partitions; // a pointer to all the other partitions
			
			float burnin_acceptance;
						
			///////////////////////
			// Private functions
			
			// Progress bar printing
			__device__ void progress_bar(float percentage);
		
			// Atomic addition of two complex numbers
			template <typename T1, typename T2>
			__device__ void atomicAddComplex_block(complex<T1> &z0, const complex<T2> &z1){
				// Atomically adds z1 to z0
				atomicAdd_block(z0.pointer_to_real(), re(z1));
				atomicAdd_block(z0.pointer_to_imag(), im(z1));
			}

			// Initialization of the MC run
			__device__ void particle_positions_initialize(); // Initializing the positions

			// Hamiltonian
			__device__ double confinement(); // The confinement Hamiltonian
			__device__ double cosine_excitation(); // A (self-adjoint) cos(dM theta) excitation.
			__device__ double sine_excitation(); // A (self-adjoint) sin(dM theta) excitation. Together they allow to reconstruct the edge dynamic structure factor e^{i dM theta}
			
			// Power sum symmetric polynomials
			__device__ void ek_construct(); // Building blocks construction: e_k = \sum_i z_i^k
			__device__ void ekr_construct(); // All the powers e_k^r
			__device__ complex<double> symPol_construct(integer_partitions::partition_element* p, int p_size); // Computes all the needed power sum symmetric polynomials from the e_k^r table
			
			__device__ void sampling_ek_construct(); // Initializes the sums of powers
			__device__ void sampling_ek_update( const complex<float> new_pos, const complex<float> old_pos); // Updates the sums of powers after a move
			__device__ complex<double> sampling_symPol_construct( const complex<double> *ek ); // Computes a power sum symmetric polynomial from a given array e_k (containing only the relevant ones)

			// Burn-in
			__device__ void burnin( uint moves_number );

	};

	// Seed for the random numbers
	__global__ void initializeRandom(curandState *state, uint seed);
	
	// Acceptance bisection (determine the optimal random MC step)
	__global__ void mcmc_acceptanceBisection(	curandState *state, 
												integer_partitions::partition* dev_partitions, int samplingEdgeState,
												float* dev_acceptances, 
												monteCarloParameters params );
		
	void optimal_step(	monteCarloParameters &params,
						curandState* devState, 
						integer_partitions::partition* dev_partitions, int samplingEdgeState );

	
	// Compute the spectrum
	__global__ void mcmc_hamiltonian(	curandState *state, 
										integer_partitions::partition* dev_partitions, int samplingEdgeState,
										monteCarloParameters params,
										results *averaged_results );

	std::vector<double> spectrum_compute(	monteCarloParameters &params,
											int angular_momentum_sector,
											bool saveOutput = false, int fileNumber = 0 ); // By default it does not save


	// Compute the (edge) dynamic structure factor
	__global__ void mcmc_edge_dsf(	curandState *state, 
									integer_partitions::partition* dev_partitions, int samplingEdgeState,
									monteCarloParameters params,
									results *dev_results );

	std::vector<std::pair<double,double>> dsf_compute(	monteCarloParameters &params,
														int angular_momentum_sector,
														bool saveOutput = false, int fileNumber = 0 ); // By default it does not save

	
	// (Nonlinear) response to an excitation
	template<typename Func>
	__global__ void mcmc_excitationResponse(	curandState *state, 
												integer_partitions::partition* dev_partitions, int samplingEdgeState,
												monteCarloParameters params,
												results *dev_results,
												Func excitation_potential );

	template<typename Func>
	void excitationResponse_compute(	monteCarloParameters &params, 
										vector<int> &angular_momentum_sectors, 
										Func excitation_potential,
										int fileNumber );

}

// Include templated functions implemetation
#include "./em_edgeResponse.tpp"

#endif
