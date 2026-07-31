#ifndef GFMC_H
#define GFMC_H

#include <vector>

#include "./complex_numbers.h"
#include "./tensors.h"
#include "./integer_partitions.h"
#include "./sys_params.h"

namespace greenFunctionMC{	
	using namespace iparts;

	// A structure for the results
	struct results{
		float* acceptance; // Stores the measured acceptance in each block
		
		// SPECTRAL FUNCTION

		// OCCUPATION NUMBERS 
		complex<double> *occupationNumbers;
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

			int injectedAngularMomentum; // The injected angular momentum

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
									int samplingLeftEdgeState, int samplingRightEdgeState, // the partitions that are used for sampling
									bool mc_step_test = false // defaulted to false. If true, only a small thermalization is done: used to bisect the good Markov-Chain Monte-Carlo coordinate step
								 );
								 
			__device__ void cleanup(); // Cleans up dynamically allocated memory
			__device__ curandState stateRelease(); // Returns the cuRAND local state

			__device__ float burnin_acceptance_get() const; // Returns the acceptance after burn-in

			//////////////
			// "Samplers"
			//__device__ void spectral_function_sampler( results *dev_results, int samplingLeftEdgeState, int samplingRightEdgeState ); // Compute the spectral function weights only 
			__device__ void occupation_numbers_sampler( results *dev_results, int samplingLeftEdgeState, int samplingRightEdgeState ); // Compute the occupation numbers weights only

		private:
			////////////////////////
			// Private variables
			
			curandState localState; // Local state for the random numbers
			int tid; // Thread identifier
			
			monteCarloParameters params; // Monte-Carlo parameters

			complex<float> position[sys_params::particlesNumber+1]; // the particle positions, as z=x+iy			
			complex<float> scaled_position[sys_params::particlesNumber+1]; // the same positions, but scaled down
			complex<double> position_powers[sys_params::particlesNumber+1]; // the same positions, but stores powers. Used as a temporary array
			float position_norm[sys_params::particlesNumber+1]; // the norms of the positions (squared distance from the origin)

			complex<double> current_right_ek[sys_params::ansatzMaxPartitionSize], current_left_ek[sys_params::ansatzMaxPartitionSize]; // the current values of e_k = \sum_k z_i^k -- only the k needed for the sampling partition (they get updated more frequently)
			complex<double> proposed_right_ek[sys_params::ansatzMaxPartitionSize], proposed_left_ek[sys_params::ansatzMaxPartitionSize]; // the proposed values of e_k (after a single particle update)
			
			int left_partition_size, right_partition_size; // The dimension of the sampling partition. e.g. {2,2,2,1,1,1,1} -> [2x3,1x4] has dimension 2
			integer_partitions::partition_element *sampling_left_partition, *sampling_right_partition; // A pointer to the sampling partitions, in compressed form
			integer_partitions::partition *device_partitions; // a pointer to all the other partitions
			
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

			// Power sum symmetric polynomials
			__device__ void sampling_ek_construct(); // Initializes the sums of powers
			__device__ void sampling_left_ek_update( const complex<float> new_pos, const complex<float> old_pos); // Updates the sums of powers after a move
			__device__ void sampling_right_ek_update( const complex<float> new_pos, const complex<float> old_pos); // Updates the sums of powers after a move
			__device__ complex<double> sampling_left_symPol_construct( const complex<double> *ek ); // Computes a power sum symmetric polynomial from a given array e_k (containing only the relevant ones)
			__device__ complex<double> sampling_right_symPol_construct( const complex<double> *ek ); // Computes a power sum symmetric polynomial from a given array e_k (containing only the relevant ones)

			// Burn-in
			__device__ void burnin( uint moves_number );

	};

	// Seed for the random numbers
	__global__ void initializeRandom(curandState *state, uint seed);
	
	// Acceptance bisection (determine the optimal random MC step)
	__global__ void mcmc_acceptanceBisection(	curandState *state, 
												integer_partitions::partition* dev_partitions, 
												int samplingLeftEdgeState, int samplingRightEdgeState,
												float* dev_acceptances, 
												monteCarloParameters params );
		
	void optimal_step(	monteCarloParameters &params,
						curandState* devState, 
						integer_partitions::partition* dev_partitions, 
						int samplingLeftEdgeState, int samplingRightEdgeState );



	// Measuring occupation numbers
	__global__ void mcmc_occupation_numbers(	curandState *state, 
												integer_partitions::partition* dev_partitions, 
												int left_samplingEdgeState, int right_samplingEdgeState,
												monteCarloParameters params,
												results *dev_results );

	void on_compute(	monteCarloParameters &params,
						int angular_momentum_sector,
						cmatrix<double> *metric,
						int fileNumber = 0); // The output is saved to a file named fileNumber_l=#.tsv, where # is the angular momentum of the number operator a^\dagger_l a_l


	void occupation_numbers_compute(	monteCarloParameters &params,
										int angular_momentum_sector,
										int fileNumber = 0); // Returns un-normalized occupation numbers (they should be normalized with the metric). This function, and its overloaded counterpart below, call on_compute


	void occupation_numbers_compute(	monteCarloParameters &params,
										int angular_momentum_sector,
										cmatrix<double> &metric,
										int fileNumber = 0); // Overloaded with the addition of the metric: occupation numbers are this time correctly normalized						

}

#endif
