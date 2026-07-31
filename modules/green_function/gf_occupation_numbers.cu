#include "../green_function.h"
#include "../factorial_logs.h"

#include <filesystem>
#include <string>
#include <fstream>
#include <fmt/core.h>

namespace greenFunctionMC {

	////////////////////////////////////////////////////////////////////
	// Compute the occupation numbers matrix elements <\psi_1| a^\dagger_l a_l \psi_2> in the power-sum sum symmetric polynomials basis
	// A single angular momentum number is required: that of the edge sector


		__device__ void montecarlo::occupation_numbers_sampler( results *dev_results, int samplingLeftEdgeState, int samplingRightEdgeState ){
			// Get the block id
			const int block_id = blockIdx.x;

			// Runs a mcmc - to sample the confinement matrix elements and metric
			// The data can be used to compute the spectrum

			// Observable-related temporary variables
			complex<double> occupationNumbers[sys_params::angularMomentumCutoff];

			for(int l=0; l<sys_params::angularMomentumCutoff; l++){			
				occupationNumbers[l] = 0;
			}

			// Temporary variables for the Markov Chain Monte Carlo
			uint local_acceptance = 0; // Acceptance associated to the given thread
			float AcceptanceRejection; // Probability of accepting a move in the Metropolis Hastings loop. If AR > random(0,1), the move is accepted

			complex<float> current_position; // Current position of the i-th particle
			float current_norm;	// Current norm of the i-th particle
			   
			complex<double> laughlin_ratio; // Ratio of the Jastrow part of the Laughlin wavefunction, at different configurations
			complex<double> jastrow_correction_factor; // (z'_i - z_0)/(z_i - z_0) * (z'_i - z_N)/(z_i - z_N)
			

			complex<double> left_polynomial_current = sampling_left_symPol_construct( current_left_ek ); // The current value of the sampling symmetric polynomial
			complex<double> right_polynomial_current = sampling_right_symPol_construct( current_right_ek ); // The current calue of the sampling symmetric polynomial

			complex<double> left_polynomial_proposed, right_polynomial_proposed; // Updated after the move

			// Temporary variables for the observables
			complex<double> singleParticle_wf_product_log;
			complex<double> jastrow;
			double phase;

			// Run the MCMC
			for(int iter=0; iter<params.MCSamples; iter++){
				// Move particle 0
				{
					// Update the position of the 0-th particle, saving its current state. If the move is rejected, these values are "restored"
					// In practice, position[], norm[] store the proposed configuration, while current_position, current_norm store the values before the move
					
					current_position = position[0];
					current_norm = position_norm[0];
					
					position[0] = position[0] + params.randomStep * complex<float>(curand_normal(&localState), curand_normal(&localState));
					position_norm[0] = norm(position[0]);
					
					// Compute the wavefunction ratio
					laughlin_ratio = 1;
					
					for(int j=1; j<sys_params::particlesNumber; j++){
						laughlin_ratio *= ( ( position[0] - position[j] ) / ( current_position - position[j] ) );
					}			

					sampling_left_ek_update( position[0], current_position );
					left_polynomial_proposed = sampling_left_symPol_construct( proposed_left_ek );

					AcceptanceRejection = cabs( icpow(laughlin_ratio, sys_params::inverseFilling) // The Laughlin Jastrow (for particle 0)
												* (left_polynomial_proposed/left_polynomial_current) // The ratio of polynomials
												* exp( 0.25f * ( current_norm - position_norm[0] ) ) ); // The Gaussians
					
					// If pr >= rand(0,1) the move is accepted. If pr>=1, there is no need to compute the random number in the first place
					if( AcceptanceRejection >= 1 or 
						AcceptanceRejection >= curand_uniform(&localState) ){
						// Increasing number of accepted moves
						local_acceptance ++;
						
						// Update the configuration
						left_polynomial_current = left_polynomial_proposed;

						for(int p=0; p<left_partition_size; p++){
							current_left_ek[p] = proposed_left_ek[p];
						}
					}
					else{
						// Restore the configuration
						position[0] = current_position;
						position_norm[0] = current_norm;
					}
				}

				// Move particles 1 to N-1
				for(int i=1; i<sys_params::particlesNumber; i++){
					// Update the position of the i-th particle, saving its current state. If the move is rejected, these values are "restored"
					// In practice, position[], norm[] store the proposed configuration, while current_position, current_norm store the values before the move
					
					current_position = position[i];
					current_norm = position_norm[i];
					
					position[i] = position[i] + params.randomStep * complex<float>(curand_normal(&localState), curand_normal(&localState));
					position_norm[i] = norm(position[i]);
					
					// Compute the wavefunction ratio
					laughlin_ratio = 1;
					
					for(int j=1; j<i; j++){
						laughlin_ratio *= ( ( position[i] - position[j] ) / ( current_position - position[j] ) );
					}
					for(int j=i+1; j<sys_params::particlesNumber; j++){
						laughlin_ratio *= ( ( position[i] - position[j] ) / ( current_position - position[j] ) );
					}			

					jastrow_correction_factor = ( (position[i]-position[0]) / (current_position-position[0]) ) *
												( (position[i]-position[sys_params::particlesNumber]) / (current_position-position[sys_params::particlesNumber]) );

					sampling_left_ek_update( position[i], current_position );
					sampling_right_ek_update( position[i], current_position );
					
					left_polynomial_proposed = sampling_left_symPol_construct( proposed_left_ek );
					right_polynomial_proposed = sampling_right_symPol_construct( proposed_right_ek );

					AcceptanceRejection = pow(cabs( jastrow_correction_factor ), sys_params::inverseFilling)  * // Taking into account particle 0 and particle N for the Laughlin Jastrow factor
										  cabs( (left_polynomial_proposed/left_polynomial_current) * (right_polynomial_proposed/right_polynomial_current) ) * // The ratios of polynomials
										  norm( icpow(laughlin_ratio, sys_params::inverseFilling) // The Laughlin Jastrow
												* exp( 0.25f * ( current_norm - position_norm[i] ) ) ); // The Gaussians


					// If pr >= rand(0,1) the move is accepted. If pr>=1, there is no need to compute the random number in the first place
					if( AcceptanceRejection >= 1 or 
						AcceptanceRejection >= curand_uniform(&localState) ){
						// Increasing number of accepted moves
						local_acceptance ++;
						
						// Update the configuration
						left_polynomial_current = left_polynomial_proposed;
						right_polynomial_current = right_polynomial_proposed;

						for(int p=0; p<left_partition_size; p++){
							current_left_ek[p] = proposed_left_ek[p];
						}
						for(int p=0; p<right_partition_size; p++){
							current_right_ek[p] = proposed_right_ek[p];
						}
					}
					else{
						// Restore the configuration
						position[i] = current_position;
						position_norm[i] = current_norm;
					}		
				}
			
				// Move particle N
				{
					// Update the position of the N-th particle, saving its current state. If the move is rejected, these values are "restored"
					// In practice, position[], norm[] store the proposed configuration, while current_position, current_norm store the values before the move
					
					current_position = position[sys_params::particlesNumber];
					current_norm = position_norm[sys_params::particlesNumber];
					
					position[sys_params::particlesNumber] = position[sys_params::particlesNumber] + params.randomStep * complex<float>(curand_normal(&localState), curand_normal(&localState));
					position_norm[sys_params::particlesNumber] = norm(position[sys_params::particlesNumber]);
					
					// Compute the wavefunction ratio
					laughlin_ratio = 1;
					
					for(int j=1; j<sys_params::particlesNumber; j++){
						laughlin_ratio *= ( ( position[sys_params::particlesNumber] - position[j] ) / ( current_position - position[j] ) );
					}			
					
					sampling_right_ek_update( position[sys_params::particlesNumber], current_position );
					right_polynomial_proposed = sampling_right_symPol_construct( proposed_right_ek );

					AcceptanceRejection = cabs( icpow(laughlin_ratio, sys_params::inverseFilling) // The Laughlin Jastrow
												* (right_polynomial_proposed/right_polynomial_current) // The ratio of polynomials
												* exp( 0.25f * ( current_norm - position_norm[sys_params::particlesNumber] )) ); // The Gaussians
					
					// If pr >= rand(0,1) the move is accepted. If pr>=1, there is no need to compute the random number in the first place
					if( AcceptanceRejection >= 1 or 
						AcceptanceRejection >= curand_uniform(&localState) ){
						// Increasing number of accepted moves
						local_acceptance ++;
						
						// Update the configuration
						right_polynomial_current = right_polynomial_proposed;

						for(int p=0; p<right_partition_size; p++){
							current_right_ek[p] = proposed_right_ek[p];
						}
					}
					else{
						// Restore the configuration
						position[sys_params::particlesNumber] = current_position;
						position_norm[sys_params::particlesNumber] = current_norm;
					}	
				}
			
			
				// The move is made: we now compute the observables
				// We first need to compute Psi_right / Psi_left = ( P_right / P_left ) * (prod_{j!=0,N} (z_j-z_N)/(z_0-z_j) )^m * exp(- (|z_N|^2-|z_0|^2)/4 )
				jastrow = 1.;
				for(int i=1; i<sys_params::particlesNumber; i++){
					jastrow *= ( position[i] - position[sys_params::particlesNumber] ) / ( position[0] - position[i] );
				}

				//wf_ratio = (right_polynomial_current / left_polynomial_current) * icpow(jastrow, sys_params::inverseFilling) * exp( -0.25f * ( position_norm[sys_params::particlesNumber] - position_norm[0] ) );
				phase = arg( right_polynomial_current / left_polynomial_current ) + sys_params::inverseFilling * arg( jastrow );

				// Then we compute the un-normalized occupation numbers
				for(int l=0; l<sys_params::angularMomentumCutoff; l++){
					singleParticle_wf_product_log = - ln2pi_fact(l) + l * clog(0.5f * position[0] * conj(position[sys_params::particlesNumber])) - 0.25f * (position_norm[0] + position_norm[sys_params::particlesNumber]);
						
					occupationNumbers[l] += cexp( singleParticle_wf_product_log + complex<double>(0, phase));
				}

				// Show progress	
				if(tid == 0 and (iter+1) % int(params.MCSamples/100.0f + 0.1f) == 0){
					progress_bar(100.0f*(iter+1)/params.MCSamples + 0.1f);
				}			
    			
			}
			
			// Update the results
			atomicAdd_block(&dev_results->acceptance[block_id], (double)local_acceptance / ( params.MCSamples * sys_params::particlesNumber));

			double scale_factor = 1. / params.MCSamples;
			for(int l=0; l<sys_params::angularMomentumCutoff; l++){	
				uint index = l + block_id*sys_params::angularMomentumCutoff;

				atomicAddComplex_block(dev_results->occupationNumbers[index], occupationNumbers[l] * scale_factor);
			}

			return;
		}

		__global__ void mcmc_occupation_numbers(	curandState *state, 
													integer_partitions::partition* dev_partitions, 
													int left_samplingEdgeState, int right_samplingEdgeState,
													monteCarloParameters params,
													results *dev_results ){

			// Thread id	
			const int local_thread = threadIdx.x;
			const int block_id = blockIdx.x;
			const int block_size = blockDim.x;
			
			int tid = local_thread + block_id * block_size;

			// Set to zero the observables
			if(local_thread == 0){
				dev_results->acceptance[block_id] = 0;
				
				for(int l=0; l<sys_params::angularMomentumCutoff; l++){	
					uint index = l + block_id*sys_params::angularMomentumCutoff;
					dev_results->occupationNumbers[index] = 0;
				}		
			}
			__syncthreads();
					
			// Run the MC
			__shared__ float block_acceptance; // Shared between the threads of each block
			if(tid == 0){
				printf("\tThermalization... ");
				block_acceptance = 0;
			}
			__syncthreads();

			montecarlo markovChain = montecarlo( state, tid,
												 params, 
												 dev_partitions, left_samplingEdgeState, right_samplingEdgeState
											   );
											   
			float local_acceptance = markovChain.burnin_acceptance_get();
			atomicAdd_block(&block_acceptance, local_acceptance); // Atomic addition
			__syncthreads(); // Sync the threads
	
			if(tid == 0){
				printf("Acceptance = %.2lf\n\tMetropolis loop\n", block_acceptance / params.threadsPerBlock);
			}			

			markovChain.occupation_numbers_sampler( dev_results, left_samplingEdgeState, right_samplingEdgeState );

			markovChain.cleanup();
			state[tid] = markovChain.stateRelease();
			__syncthreads();

			// Average each block over the threads			
			if(local_thread == 0){
				double scale_factor = 1. / params.threadsPerBlock;

				dev_results->acceptance[block_id] *= scale_factor;
				
				for(int l=0; l<sys_params::angularMomentumCutoff; l++){	
					uint index = l + block_id*sys_params::angularMomentumCutoff;

					dev_results->occupationNumbers[index] *= scale_factor;	
				}
			}
			
			return;
		}
	

		void on_compute( 	monteCarloParameters &params, // The parameters for the simulation
							int angular_momentum_sector, // The angular momentum sector
							cmatrix<double> *metric, // The metric
							int fileNumber
						){
			
			// First things first: set the injected angular momentum.
			// The highest possibly occupied state is at angular momentum m(N-1)+injectedAngularMomentum
			// We explore momenta around m(N-1)
			if( angular_momentum_sector > sys_params::ansatzMaxAngularMomentum ){
				std::cout << "The defined parameter ansatzMaxAngularMomentum is smaller than dL. Modify and re-run system_parameters.cpp" << std::endl;
				exit(1);
			}
			params.injectedAngularMomentum = angular_momentum_sector;

			// Set the gpu
			gpuErrchk( cudaDeviceSetLimit(cudaLimitMallocHeapSize,  128 *	1024 * 1024) );
			gpuErrchk( cudaDeviceSetLimit(cudaLimitStackSize,  32 *	1024) );
			gpuErrchk( cudaDeviceSynchronize() );

			// Generate the random seed for the random number generation
			srand(time(NULL));
			uint seed = rand();
			
			//  Random number generation stuff is here initialized
			curandState *devState;  
			gpuErrchk( cudaMalloc(&devState, params.gridSize*sizeof(curandState)) );
				
			initializeRandom<<<params.nBlocks, params.threadsPerBlock>>>(devState, seed);
			gpuErrchk( cudaDeviceSynchronize() );
						
			// Generate the partitions
			vector<int> angular_momenta = { angular_momentum_sector };
			integer_partitions P(sys_params::particlesNumber, angular_momenta);

			integer_partitions::partition* dev_partitions = P.getDevPartitions();

			params.setEdgeSpaceInfo(P.maximalDegree, P.subspaceDimension);
			if( params.extendedMaxDegree > sys_params::ansatzExtendedMaxDegree ){
				std::cout << "\nWith this angular momentum choice,\nthe maximal singe-coordinate degree is " << params.maxDegree << std::endl;
				std::cout << "The upper bound is set to " << sys_params::ansatzExtendedMaxDegree-1 << " in sys_params.h. It should be at least as large" << std::endl;
				std::cout << "Consider re-generating sys_params.h by running an updated system_parameters.cpp" << std::endl;
				exit(1);
			}
			if( P.longestPartitionLenght > sys_params::ansatzMaxPartitionSize ){
				std::cout << "\nWith this angular momentum choice,\nthe length of the longest (compressed) partition is " << P.longestPartitionLenght << std::endl;
				std::cout << "The upper bound is set to " << sys_params::ansatzMaxPartitionSize << " in sys_params.h. It should be at least as large" << std::endl;
				std::cout << "Consider re-generating sys_params.h by running an updated system_parameters.cpp" << std::endl;
				exit(1);
			}
			if( P.subspaceDimension > sys_params::ansatzSubspaceDimension){
				std::cout << "\nWith this angular momentum choice,\nthe edge Hilbert space size is " << P.subspaceDimension << std::endl;
				std::cout << "The upper bound is set to " << sys_params::ansatzSubspaceDimension << " in sys_params.h. It should be at least as large" << std::endl;
				std::cout << "Consider re-generating sys_params.h by running an updated system_parameters.cpp" << std::endl;
				exit(1);				
			}

			// Lambda to compute and return the results
			float acceptance;
			auto measure = [&] ( int left_edgeMode, int right_edgeMode, rank3tensor<complex<double>> &occupation_matrix_elements ){
				size_t sz = params.nBlocks * sys_params::angularMomentumCutoff;

				// Allocate device memory to store the results
				// 1) tmp_results as a pointer on the host, pointing to device memory
				results *tmp_results = new results;

				gpuErrchk( cudaMalloc(&tmp_results->acceptance, params.nBlocks * sizeof(float)) );
				gpuErrchk( cudaMalloc(&tmp_results->occupationNumbers, sz * sizeof(complex<double>)) );

				// 2) device_results as a pointer on the device, then copy the stuff pointed by tmp_results inside it
				results *device_results;								
				gpuErrchk( cudaMalloc(&device_results, sizeof(results)) );
				gpuErrchk( cudaMemcpy(device_results, tmp_results, sizeof(results), cudaMemcpyHostToDevice) );
				
				// 3) Launch the Markov-Chain -- measuring the elapsed time
				float time;
				cudaEvent_t start, stop;
				
				gpuErrchk( cudaEventCreate(&start) );
				gpuErrchk( cudaEventCreate(&stop) );
				gpuErrchk( cudaEventRecord(start, 0) );
				
				mcmc_occupation_numbers<<<params.nBlocks, params.threadsPerBlock>>>(devState, dev_partitions, left_edgeMode, right_edgeMode, params, device_results);
				gpuErrchk( cudaPeekAtLastError() );
				gpuErrchk( cudaDeviceSynchronize() );

				gpuErrchk( cudaEventRecord(stop, 0) );
				gpuErrchk( cudaEventSynchronize(stop) );
				gpuErrchk( cudaEventElapsedTime(&time, start, stop) );

				// 4) Allocate host memory to retrieve the results
				results *host_results   = new results;
				
				host_results->acceptance = new float[params.nBlocks];
				host_results->occupationNumbers = new complex<double>[sz];
				
				gpuErrchk( cudaMemcpy(host_results->acceptance, tmp_results->acceptance, params.nBlocks * sizeof(float), cudaMemcpyDeviceToHost) );
				gpuErrchk( cudaMemcpy(host_results->occupationNumbers, tmp_results->occupationNumbers, sz * sizeof(complex<double>), cudaMemcpyDeviceToHost) );

				// 5) Allocate space for averaging the results
				results *averaged_results = new results;
				averaged_results->acceptance = new float;
				averaged_results->occupationNumbers = new complex<double>[sys_params::angularMomentumCutoff];

				// 6) Initialize them to zero
				*(averaged_results->acceptance) = 0.;
				for(int l=0; l<sys_params::angularMomentumCutoff; l++){
					averaged_results->occupationNumbers[l] = 0.;
				}
			
				// 7) Average
				for(int block=0; block<params.nBlocks; block++){
					*(averaged_results->acceptance)  += host_results->acceptance[block] / params.nBlocks;
					
					for(int l=0; l<sys_params::angularMomentumCutoff; l++){
						uint index = l + block*sys_params::angularMomentumCutoff;
						averaged_results->occupationNumbers[l] += host_results->occupationNumbers[index] / params.nBlocks;				
					}
				}
			
				// 8) Normalize and save
				acceptance = *(averaged_results->acceptance);

				complex<double> sum = 0;
				for(int l=0; l<sys_params::angularMomentumCutoff; l++){
					sum += averaged_results->occupationNumbers[l];
				}

				if( metric ){ // If it is not the nullptr
					for(int l=0; l<sys_params::angularMomentumCutoff; l++){
						occupation_matrix_elements(left_edgeMode, right_edgeMode, l) = averaged_results->occupationNumbers[l] / sum * sys_params::particlesNumber * (*metric)(left_edgeMode, right_edgeMode);
					}					 
				}
				else{
					for(int l=0; l<sys_params::angularMomentumCutoff; l++){
						occupation_matrix_elements(left_edgeMode, right_edgeMode, l) = averaged_results->occupationNumbers[l] / sum * sys_params::particlesNumber;
					}						
				}

				// 9) Free the device and the temporary variables on the host			
				gpuErrchk( cudaFree(tmp_results->acceptance) );
				gpuErrchk( cudaFree(tmp_results->occupationNumbers) );
				delete tmp_results;

				gpuErrchk( cudaFree(device_results) );

				delete[] host_results->acceptance;
				delete[] host_results->occupationNumbers;
				delete host_results;

				delete averaged_results->acceptance;
				delete[] averaged_results->occupationNumbers;
				delete averaged_results;				
						
				return time/1000.; // Returns the elapsed time (in seconds)
			};

			// Observables to be filled-in
			rank3tensor<complex<double>> occupation_matrix_elements(params.subspaceDimension, params.subspaceDimension, sys_params::angularMomentumCutoff, 0);

			// Now loop over all the edge modes...
			for(int left_edgeMode=0; left_edgeMode<P.subspaceDimension; left_edgeMode++){
				for(int right_edgeMode=0; right_edgeMode<P.subspaceDimension; right_edgeMode++){
					// Print some info
					std::cout << "Sampling P1[" << left_edgeMode << "]: "; 
					P.printEdgePartition(left_edgeMode);
					std::cout << " and P2[" << right_edgeMode << "]: "; 
					P.printEdgePartition(right_edgeMode);

					// Optimal Monte Carlo step
					std::cout << "\n\tDetermining the optimal Markov-Chain random step to achieve " << params.targetAcceptance << " acceptance rate.";
					optimal_step(params, devState, dev_partitions, left_edgeMode, right_edgeMode);
					std::cout << " Optimal step: " << params.randomStep << "\n\n";

					// Measure
					float time = measure( left_edgeMode, right_edgeMode, occupation_matrix_elements );
					printf("\n\tFinshed. Total number of samples: %lu. Elapsed time: %.1fs\n", params.totalSamplesNumber, time);
					printf("\tAverage accepance\t%.2lf\n\n",  acceptance );
				}
			}

			// Symmetrize the results
			auto make_self_adjoint = [&params](rank3tensor<complex<double>> &mat){

				for(int l=0; l<sys_params::angularMomentumCutoff; l++){
					cmatrix<double> tmp(params.subspaceDimension, params.subspaceDimension, 0);

					for(int e1=0; e1<params.subspaceDimension; e1++){
						for(int e2=0; e2<params.subspaceDimension; e2++){
							tmp(e1, e2) = ( mat(e1, e2, l) + conj(mat(e2, e1, l)) ) / 2.;
						}
					}

					for(int e1=0; e1<params.subspaceDimension; e1++){
						for(int e2=0; e2<params.subspaceDimension; e2++){
							mat(e1, e2, l) = tmp(e1, e2);
						}
					}
				}

				return;
			};

			make_self_adjoint( occupation_matrix_elements );

			// Save if asked
			std::cout << "Saving the matrices (you need post-processing to rotate onto the eigenvector basis and eventually to compute statistical errorbars)\n\n";
			// Create directory structure
			std::string dir_path;
			if( metric ){ // If it is not the nullptr
				dir_path = fmt::format("../output/N={}_m={}/dL={}/Statistics/OccupationNumbers_normalized", sys_params::particlesNumber, sys_params::inverseFilling, angular_momentum_sector );
			}
			else{
				dir_path = fmt::format("../output/N={}_m={}/dL={}/Statistics/OccupationNumbers_not_normalized", sys_params::particlesNumber, sys_params::inverseFilling, angular_momentum_sector );
			}
			std::filesystem::create_directories(dir_path);

			// Create the output file
			std::string file_name = fmt::format("{}.tsv", fileNumber);
			std::ofstream out( dir_path + "/" + file_name );

			// Save
			for( int state1 = 0; state1 < params.subspaceDimension; state1++ ){
				for( int state2 = 0; state2 < params.subspaceDimension; state2++ ){
					for(int l=0; l<sys_params::angularMomentumCutoff; l++){
						out << fmt::format("{}\t{}\t{}\t{:.6f}\t{:.6f}\n", state1, state2, l, re(occupation_matrix_elements(state1,state2,l)), im(occupation_matrix_elements(state1,state2,l)));
					}
					out << "\n";
				}
			}
			
			// Free what's left
			gpuErrchk( cudaFree(devState) );

			return;			
		}



		void occupation_numbers_compute(	monteCarloParameters &params,
											int angular_momentum_sector,
											cmatrix<double> &metric,
											int fileNumber){

			on_compute(params, angular_momentum_sector, &metric, fileNumber);

		}											

		void occupation_numbers_compute(	monteCarloParameters &params,
											int angular_momentum_sector,
											int fileNumber){

			on_compute(params, angular_momentum_sector, nullptr, fileNumber);
		}



}
