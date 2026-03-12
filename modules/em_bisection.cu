#include "./edge_montecarlo.h"

namespace edgeMC {
	////////////////////////////////////////////////////////////////////
	// Acceptance bisection

		__global__ void mcmc_acceptanceBisection(	curandState *state, 
													integer_partitions::partition* dev_partitions, int samplingEdgeState,
													float* dev_acceptances,
													monteCarloParameters params ){
			
			// Thread id	
			const int local_thread = threadIdx.x;
			const int block_id = blockIdx.x;
			const int block_size = blockDim.x;
			
			int tid = local_thread + block_id * block_size;

			// Set to zero the measured acceptances
			if(local_thread == 0){
				dev_acceptances[block_id] = 0;
			}
			__syncthreads();
					
			// Run the MC
			montecarlo markovChain = montecarlo( state, tid,
												 params, 
												 dev_partitions, samplingEdgeState, 
												 true
											   );
											   
			float local_acceptance = markovChain.burnin_acceptance_get();
			
			markovChain.cleanup();
			state[tid] = markovChain.stateRelease();
			
			// Average the acceptances over the blocks
			atomicAdd_block(&dev_acceptances[block_id], local_acceptance);
			__syncthreads();
			
			if(local_thread == 0){
				dev_acceptances[block_id] /= params.threadsPerBlock;
			}
			
							
			return;
		}

		void optimal_step(	monteCarloParameters &params, // Pass by reference the simulation parameters; the MC step will be updated at the end
							curandState* devState, 
							integer_partitions::partition* dev_partitions, int samplingEdgeState ){						

			// returns the expected acceptance given a Monte Carlo step newStep
			auto measureAcceptance = [&]( float newStep ){
				
				params.setRandomStep( newStep );
				
				float *dev_acceptances;
				gpuErrchk( cudaMalloc(&dev_acceptances, params.nBlocks * sizeof(float)) );
				float *host_acceptances = new float[params.nBlocks];
				
				mcmc_acceptanceBisection<<<params.nBlocks, params.threadsPerBlock>>>(devState, dev_partitions, samplingEdgeState, dev_acceptances, params);
				gpuErrchk( cudaPeekAtLastError() );
				gpuErrchk( cudaDeviceSynchronize() );

				gpuErrchk( cudaMemcpy(host_acceptances, dev_acceptances, params.nBlocks * sizeof(float), cudaMemcpyDeviceToHost) );
					
				double averaged_acceptance = 0;
				for(int i=0; i<params.nBlocks; i++){
					averaged_acceptance += host_acceptances[i] / params.nBlocks;
				}

				// Free the memory up	
				gpuErrchk( cudaFree(dev_acceptances) );
				delete[] host_acceptances;
				
				return averaged_acceptance;
			};
			
			// Set-up the bisection
			float step1, step2 = params.randomStep;
			float acceptance1, acceptance2 = measureAcceptance( step2 );

			// Check whether acceptance(step) is already close enough to the target
			if( abs(acceptance2 - params.targetAcceptance) < params.acceptance_tol ){	
				params.setRandomStep(step2);
				return;
			}
			
			// Preliminary cycle: setting step1
			// If the acceptance is larger than the target one, increase the random step (so that the acceptance drops) until the target is surpassed, and viceversa if its larger than the target
			if( acceptance2 > params.targetAcceptance ){
				step1 = step2;
				do{			
					step1 *= 2.0f;
					acceptance1 = measureAcceptance( step1 );
				}while( (acceptance2-params.targetAcceptance)*(acceptance1-params.targetAcceptance) > 0 );
			}
			else{
				step1 = step2;
				do{	
					step1 *= 0.5f;
					acceptance1 = measureAcceptance( step1 );
				}while( (acceptance2-params.targetAcceptance)*(acceptance1-params.targetAcceptance) > 0 );
			}
			
			// Check whether acceptance(step) is already close enough to the target
			if( abs(acceptance1 - params.targetAcceptance) < params.acceptance_tol){
				params.setRandomStep(step1);
				return;
			}
			
			// Swap the two steps if they are in reverse order
			auto swap = [](float &x, float &y){
				float tmp = x;
				x = y;
				y = tmp;
				
				return;
			};
			
			if(step2 < step1){
				swap(step1, step2);
				swap(acceptance1, acceptance2);
			}
			
			// Now we can truly bisect the interval
			while(true){
				
				float stepGuess = ( (params.targetAcceptance - acceptance1) * step2 + (acceptance2 - params.targetAcceptance) * step1 ) / (acceptance2 - acceptance1); // linear interpolation
				float acceptanceGuess = measureAcceptance( stepGuess );

				if( abs(acceptanceGuess - params.targetAcceptance) < params.acceptance_tol){ // if the interpolation is close, end
					params.setRandomStep(stepGuess);
					return;
				}
				
				if(  (acceptanceGuess-params.targetAcceptance)*(acceptance1-params.targetAcceptance) < 0 ){ // solution falls in the [step1, guess] range
					acceptance2 = acceptanceGuess;
					step2 = stepGuess;
				}
				else if( (acceptanceGuess-params.targetAcceptance)*(acceptance2-params.targetAcceptance) < 0 ){ // solution falls in the [guess, step2] range
					acceptance1 = acceptanceGuess;
					step1 = stepGuess;			
				}
				else{ // Something went wrong
					printf("\nSomething went wrong with the bisection search for the optimal Markov-Chain step\nCheck the code\n");
					exit(1);
				}
			}
			
			return;
		}

}
