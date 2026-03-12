#include "./edge_montecarlo.h"

namespace edgeMC {

	////////////////////////////////////////////////////////////////////
	// Members of the monteCarloParameters class

		// Initialization
		__host__ monteCarloParameters::monteCarloParameters(){
			// Mesh
			mesh_points_number = 200;
			dr = 1.7 * sys_params::DCl / mesh_points_number;

			// Initialize subspaceDimension to -1: used to check whether it has been assigned later
			subspaceDimension = -1;
			
			// Monte Carlo parameters (single thread)	
			randomStep = 0.8f; // Initial Monte-Carlo step. It gets updated authomatically by running some warm-up Markov Chains			
			burnInMoves = 1000; // Number of burn-in moves per thread
			MCSamples = 50000; // Number of global MC moves per thread
			
			targetAcceptanceMoves = 100; // Number of burn-in moves to do when the goal is to measure the acceptance
			targetAcceptance = 0.50f; // The ideal value for the acceptance
			acceptance_tol = 0.01f; // The tolerance	
			
			threadsPerBlock = 128; // the number of threads per block
			nBlocks = 32; // the number of blocks
			gridSize = nBlocks * threadsPerBlock; // the size of the grid
			totalSamplesNumber = MCSamples * gridSize; // the total number of MC global moves
			
			return;		 
		}

		// Parameters that NEED to be set
		__host__ void monteCarloParameters::setEdgeSpaceInfo(int maxPolyDegree, int subDim){
			maxDegree = maxPolyDegree;
			extendedMaxDegree = maxDegree+1;
				 
			subspaceDimension = subDim;
		}

		// Optional adjustment of some parameters
		 __host__ void monteCarloParameters::setMesh(int new_mesh_points_number){
			mesh_points_number = new_mesh_points_number;
			dr = 1.7 * sys_params::DCl / mesh_points_number;
			return;
		}

		__host__ void monteCarloParameters::setBurnIn(uint newBurnInMoves){
			burnInMoves = newBurnInMoves;
			return;
		}

		__host__ void monteCarloParameters::setMCSamples(uint newMCSamples){
			MCSamples = newMCSamples;
			totalSamplesNumber = MCSamples * gridSize;
			return;
		}

		__host__ void monteCarloParameters::setGPU_threads_blocks(uint newThreadsPerBlock, uint newNBlocks){
			threadsPerBlock = newThreadsPerBlock;
			nBlocks = newNBlocks;
			
			gridSize = nBlocks * threadsPerBlock;
			totalSamplesNumber = MCSamples * gridSize;
			return;
		}

		__host__ void monteCarloParameters::setAcceptanceBisectionParams(uint newTargetAcceptanceMoves, float newTargetAcceptance, float newAcceptance_tol){
			targetAcceptanceMoves = newTargetAcceptanceMoves;
			targetAcceptance = newTargetAcceptance;
			acceptance_tol = newAcceptance_tol;
			return;			
		}

		__host__ __device__ void monteCarloParameters::setRandomStep(float newRandomStep){
			randomStep = newRandomStep;
			return;
		}

}
