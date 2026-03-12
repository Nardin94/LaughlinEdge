#ifndef EDGEMC_TPP
#define EDGEMC_TPP

#ifndef EDGEMC_H
#error __FILE__ should only be included from edge_montecarlo.h
#endif

#include <filesystem>
#include <string>
#include <fstream>
#include <fmt/core.h>

#include <type_traits>

namespace edgeMC {

    // We want to solve the time-dependent Schrodinger equation, starting from a Laughlin state
    // H\psi = i d/dt \psi
    // with a particular form of the Hamiltonian: H = H0 + V, V = f(t) U(r)
    // due to the separable form, we can sample U
    // and obtain an effective Schrodinger equation by decomposing \psi onto the edge modes
    // ( H_{ij} + f(t) U_{ij} ) C_j = i C'_j
    // We here sample U

	template<typename Func>
	__device__ void montecarlo::excitationResponse_sampler( results *dev_results, int target_state, Func excitation_potential ){
		// Get the block id
		const int block_id = blockIdx.x;

		// Runs a mcmc - to sample the confinement matrix elements and metric
		// The data can be used to compute the spectrum

		// Observable-related temporary variables
		float h0; // Hamiltonian
		float u; // Excitation potential
		complex<double> g_f; // Ratio of the two wavefunctions (at the same configuration)
		complex<double>	fHg[sys_params::ansatzSubspaceDimension]; // Building block for the confinement (f|Hg)
		complex<double>	fUg[sys_params::ansatzSubspaceDimension]; // Building block for the confinement (f|Ug)					
		complex<double>	fg[sys_params::ansatzSubspaceDimension]; // Building block for the scalar product (f|g)
		double gg[sys_params::ansatzSubspaceDimension]; // Normalization

        complex<double>	rc[sys_params::ansatzSubspaceDimension*sys_params::ansatzMultipletSize];
        complex<double> rs[sys_params::ansatzSubspaceDimension*sys_params::ansatzMultipletSize];
        double sumCos[sys_params::ansatzMultipletSize];
        double sumSin[sys_params::ansatzMultipletSize];

		for(int state=0; state<params.subspaceDimension; state++){			
			fHg[state] = 0;
			fUg[state] = 0;
			fg[state] = 0;
			gg[state] = 0;

			for(int J=0; J<params.numberOfSectors; J++){
                rc[J + state*params.numberOfSectors] = 0;
                rs[J + state*params.numberOfSectors] = 0;
			}
		}

		// Temporary variables for the Markov Chain Monte Carlo
		float local_acceptance = 0; // Acceptance associated to the given thread
		float AcceptanceRejection; // Probability of accepting a move in the Metropolis Hastings loop. If AR > random(0,1), the move is accepted

		complex<float> current_position; // Current position of the i-th particle
		float current_norm;	// Current norm of the i-th particle
			   
		complex<double> laughlin_ratio; // Ratio of the Jastrow part of the Laughlin wavefunction, at different configurations
				
		complex<double> polynomial_current = sampling_symPol_construct( current_ek ); // The current calue of the sampling symmetric polynomial
		complex<double> polynomial_proposed; // Updated after the move

		// Initialize the angular variables
		for(int i=0; i<sys_params::particlesNumber; i++){
			angle[i] = arg(position[i]);
		}	

		// Run the MCMC
		for(int iter=0; iter<params.MCSamples; iter++){
			// Move every particle once
			for(int i=0; i<sys_params::particlesNumber; i++){
				// Update the position of the i-th particle, saving its current state. If the move is rejected, these values are "restored"
				// In practice, position[], norm[] store the proposed configuration, while current_position, current_norm store the values before the move
					
				current_position = position[i];
				current_norm = position_norm[i];
					
				position[i] = position[i] + params.randomStep * complex<float>(curand_normal(&localState), curand_normal(&localState));
				position_norm[i] = norm(position[i]);
					
				// Compute the wavefunction ratio
				laughlin_ratio = 1;
					
				for(int j=0; j<i; j++){
					laughlin_ratio *= ( ( position[i] - position[j] ) / ( current_position - position[j] ) );
				}
				for(int j=i+1; j<sys_params::particlesNumber; j++){
					laughlin_ratio *= ( ( position[i] - position[j] ) / ( current_position - position[j] ) );
				}			

				sampling_ek_update( position[i], current_position );
				polynomial_proposed = sampling_symPol_construct( proposed_ek );
										
				AcceptanceRejection = norm( ( polynomial_proposed / polynomial_current ) // The ratio of polynomials
												* icpow(laughlin_ratio, sys_params::inverseFilling) // The Laughlin Jastrow
												* exp( 0.25f * ( current_norm - position_norm[i] ) ) ); // The Gaussians
					
				// If pr >= rand(0,1) the move is accepted. If pr>=1, there is no need to compute the random number in the first place
				if( AcceptanceRejection >= 1 or 
					AcceptanceRejection >= curand_uniform(&localState) ){
					// Increasing number of accepted moves
					local_acceptance ++;
						
					// Update the configuration
					polynomial_current = polynomial_proposed;

					for(int p=0; p<partition_size; p++){
						current_ek[p] = proposed_ek[p];
					}

					// Update the angle
					angle[i] = arg(position[i]);
                }
				else{
					// Restore the configuration
					position[i] = current_position;
					position_norm[i] = current_norm;
				}		
			}
			
			// Compute the observables	
			ek_construct(); // Compute the e_k polynomials
			ekr_construct(); // Compute their powers e_k^r
				
			h0 = confinement();
			u = excitation_potential( position );

            for(int J=0; J<params.numberOfSectors; J++){
                sumCos[J] = 0;
                sumSin[J] = 0;
                            
                for(int i=0; i<sys_params::particlesNumber; i++){	
                    sumCos[J] += cos(angle[i] * params.injectedAngularMomenta[J]);
                    sumSin[J] += sin(angle[i] * params.injectedAngularMomenta[J]);
                }
            }

			for(int state=0; state<params.subspaceDimension; state++){
				if( state != target_state ){	
					g_f = symPol_construct(device_partitions[state].partitionArray, device_partitions[state].size) / polynomial_current;
				}
				else{
					g_f = 1;
				}

				fHg[state] += h0 * g_f;
				fUg[state] += u * g_f;				
				fg[state] += g_f;
				gg[state] += norm(g_f);

                // Edge density fourier components
                for(int J=0; J<params.numberOfSectors; J++){				
                    rc[J + state*params.numberOfSectors] += sumCos[J] * g_f;
                    rs[J + state*params.numberOfSectors] += sumSin[J] * g_f;
                }                
			}

			// Show progress	
			if(tid == 0 and (iter+1) % int(params.MCSamples/100.0f + 0.1f) == 0){
				progress_bar(100.0f*(iter+1)/params.MCSamples + 0.1f);
			}
		}
			
		// Update the results
		atomicAdd_block(&dev_results->acceptance[block_id], local_acceptance / ( params.MCSamples * sys_params::particlesNumber));
		
        double scale_factor = 1. / params.MCSamples;

		for(int state=0; state<params.subspaceDimension; state++){		
			uint index = state + block_id*params.subspaceDimension;

            atomicAddComplex_block(dev_results->fHg_ff[index], fHg[state] * scale_factor);
			atomicAddComplex_block(dev_results->fUg_ff[index], fUg[state] * scale_factor);			
			atomicAddComplex_block(dev_results->fg_ff[index], fg[state] * scale_factor);
			atomicAdd_block(&dev_results->gg_ff[index], gg[state] * scale_factor);

            uint base1 = params.numberOfSectors*index;
            uint base2 = state*params.numberOfSectors;
            for(int J=0; J<params.numberOfSectors; J++){		
                atomicAddComplex_block(dev_results->r_ct[J + base1], rc[J + base2] * scale_factor);
                atomicAddComplex_block(dev_results->r_st[J + base1], rs[J + base2] * scale_factor);
            }
		}

        return;
    }


	template<typename Func>
	__global__ void mcmc_excitationResponse(	curandState *state, 
												integer_partitions::partition* dev_partitions, int samplingEdgeState,
												monteCarloParameters params,
												results *dev_results,
                                                Func excitation_potential ){

		// Thread id	
		const int local_thread = threadIdx.x;
		const int block_id = blockIdx.x;
		const int block_size = blockDim.x;
			
		int tid = local_thread + block_id * block_size;

		// Set to zero the observables
		if(local_thread == 0){
			dev_results->acceptance[block_id] = 0;
				
			for(int state=0; state<params.subspaceDimension; state++){
				uint index = state + block_id*params.subspaceDimension;

				dev_results->fHg_ff[index] = 0;
                dev_results->fUg_ff[index] = 0;
				dev_results->fg_ff[index] = 0;
				dev_results->gg_ff[index] = 0;

                for(int J=0; J<params.numberOfSectors; J++){
                    dev_results->r_ct[J + params.numberOfSectors*index] = 0;
                    dev_results->r_st[J + params.numberOfSectors*index] = 0;
                }
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
											 dev_partitions, samplingEdgeState,
                                             excitation_potential
										   );
											   
		float local_acceptance = markovChain.burnin_acceptance_get();
		atomicAdd_block(&block_acceptance, local_acceptance); // Atomic addition
		__syncthreads(); // Sync the threads
	
		if(tid == 0){
			printf("Acceptance = %.2lf\n\tMetropolis loop\n", block_acceptance / params.threadsPerBlock);
		}			

		markovChain.excitationResponse_sampler( dev_results, samplingEdgeState, excitation_potential );

		markovChain.cleanup();
		state[tid] = markovChain.stateRelease();
		__syncthreads();

		// Average each block over the threads			
		if(local_thread == 0){
            double scale_factor = 1. / params.threadsPerBlock;
            
			dev_results->acceptance[block_id] *= scale_factor;
				
			for(int state=0; state<params.subspaceDimension; state++){
				uint index = state + block_id*params.subspaceDimension;

				dev_results->fHg_ff[index] *= scale_factor;
                dev_results->fUg_ff[index] *= scale_factor;
				dev_results->fg_ff[index] *= scale_factor;
				dev_results->gg_ff[index] *= scale_factor;		

                uint base = params.numberOfSectors*index;
                for(int J=0; J<params.numberOfSectors; J++){
                    dev_results->r_ct[J + base] *= scale_factor;
                    dev_results->r_st[J + base] *= scale_factor;
                }                
			}
		}

        return;
    }

	template<typename Func>
	void excitationResponse_compute(	monteCarloParameters &params, 
                                        vector<int> &angular_momentum_sectors, 
                                        Func excitation_potential,
                                        int fileNumber ){
        // Check whether the provided function is legit
        static_assert(
            std::is_invocable_r_v<double, Func, complex<float>*>,
            "The excitation potential provided should be a __device__ function taking std::complex<float>* and returning double"
        );
        
        // Set gpu
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
		integer_partitions P(sys_params::particlesNumber, angular_momentum_sectors);

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

        vector<int> angular_momenta_list = P.getAngularMomenta();

        // Copy the angular momentum list
        int* dev_angular_momentum_sectors;
        cudaMalloc(&dev_angular_momentum_sectors, angular_momentum_sectors.size() * sizeof(int));
        cudaMemcpy(dev_angular_momentum_sectors, angular_momentum_sectors.vec_pointer(), angular_momentum_sectors.size() * sizeof(int), cudaMemcpyHostToDevice);

		params.injectedAngularMomenta = dev_angular_momentum_sectors; // An array of angular momenta
		params.numberOfSectors = angular_momentum_sectors.size(); // The length of the array        

        if( params.numberOfSectors > sys_params::ansatzMultipletSize ){
			std::cout << "\nYou chose " << params.numberOfSectors << " different angular momenta." << std::endl;
			std::cout << "The upper bound is set to " << sys_params::ansatzMultipletSize << " in sys_params.h. It should be at least as large" << std::endl;
			std::cout << "Consider re-generating sys_params.h by running an updated system_parameters.cpp" << std::endl;
			exit(1);
        }

		// Lambda to compute and return the results
		float acceptance;
		auto measure = [&] ( int edgeMode, cmatrix<double> &H, cmatrix<double> &M, cmatrix<double> &U, rank3tensor<complex<double>> &RC, rank3tensor<complex<double>> &RS ){
			size_t sz = params.nBlocks * params.subspaceDimension;

			// Allocate device memory to store the results
			// 1) tmp_results as a pointer on the host, pointing to device memory
			results *tmp_results = new results;

			gpuErrchk( cudaMalloc(&tmp_results->acceptance, params.nBlocks * sizeof(float)) );
			gpuErrchk( cudaMalloc(&tmp_results->fHg_ff, sz * sizeof(complex<double>)) );
            gpuErrchk( cudaMalloc(&tmp_results->fUg_ff, sz * sizeof(complex<double>)) );
			gpuErrchk( cudaMalloc(&tmp_results->fg_ff, sz * sizeof(complex<double>)) );
			gpuErrchk( cudaMalloc(&tmp_results->gg_ff, sz * sizeof(double)) );
            gpuErrchk( cudaMalloc(&tmp_results->r_ct, sz * params.numberOfSectors * sizeof(complex<double>)) );
            gpuErrchk( cudaMalloc(&tmp_results->r_st, sz * params.numberOfSectors * sizeof(complex<double>)) );


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
				
			mcmc_excitationResponse<<<params.nBlocks, params.threadsPerBlock>>>(devState, dev_partitions, edgeMode, params, device_results, excitation_potential);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );

			gpuErrchk( cudaEventRecord(stop, 0) );
			gpuErrchk( cudaEventSynchronize(stop) );
			gpuErrchk( cudaEventElapsedTime(&time, start, stop) );

			// 4) Allocate host memory to retrieve the results
			results *host_results = new results;
				
			host_results->acceptance = new float[params.nBlocks];
			host_results->fHg_ff = new complex<double>[sz];
            host_results->fUg_ff = new complex<double>[sz];
			host_results->fg_ff = new complex<double>[sz];
			host_results->gg_ff = new  double[sz];	
            host_results->r_ct = new complex<double>[sz * params.numberOfSectors];
            host_results->r_st = new complex<double>[sz * params.numberOfSectors];
				
			gpuErrchk( cudaMemcpy(host_results->acceptance, tmp_results->acceptance, params.nBlocks * sizeof(float), cudaMemcpyDeviceToHost) );
			gpuErrchk( cudaMemcpy(host_results->fHg_ff, tmp_results->fHg_ff, sz * sizeof(complex<double>), cudaMemcpyDeviceToHost) );
            gpuErrchk( cudaMemcpy(host_results->fUg_ff, tmp_results->fUg_ff, sz * sizeof(complex<double>), cudaMemcpyDeviceToHost) );
			gpuErrchk( cudaMemcpy(host_results->fg_ff, tmp_results->fg_ff, sz * sizeof(complex<double>), cudaMemcpyDeviceToHost) );
			gpuErrchk( cudaMemcpy(host_results->gg_ff, tmp_results->gg_ff, sz * sizeof(double), cudaMemcpyDeviceToHost) );
            gpuErrchk( cudaMemcpy(host_results->r_ct, tmp_results->r_ct, sz * params.numberOfSectors * sizeof(complex<double>), cudaMemcpyDeviceToHost) );
            gpuErrchk( cudaMemcpy(host_results->r_st, tmp_results->r_st, sz * params.numberOfSectors * sizeof(complex<double>), cudaMemcpyDeviceToHost) );

			// 5) Allocate space for averaging the results
			results *averaged_results = new results;
			averaged_results->acceptance = new float;
			averaged_results->fHg_ff = new complex<double>[P.subspaceDimension];
            averaged_results->fUg_ff = new complex<double>[P.subspaceDimension];
			averaged_results->fg_ff = new complex<double>[P.subspaceDimension];
			averaged_results->gg_ff = new double[P.subspaceDimension];
	        averaged_results->r_ct = new complex<double>[P.subspaceDimension * params.numberOfSectors];
            averaged_results->r_st = new complex<double>[P.subspaceDimension * params.numberOfSectors];


			// 6) Initialize them to zero
			*(averaged_results->acceptance) = 0.;
			for(int state=0; state<params.subspaceDimension; state++){
				averaged_results->fHg_ff[state] = 0.;
                averaged_results->fUg_ff[state] = 0.;
				averaged_results->fg_ff[state] = 0.;
				averaged_results->gg_ff[state] = 0.;

                for(int J=0; J<params.numberOfSectors; J++){
                    averaged_results->r_ct[J + params.numberOfSectors*state] = 0.;
                    averaged_results->r_st[J + params.numberOfSectors*state] = 0.;
                }
			}
			
			// 7) Average
			for(int block=0; block<params.nBlocks; block++){
		    	*(averaged_results->acceptance)  += host_results->acceptance[block] / params.nBlocks;
					
				for(int state=0; state<params.subspaceDimension; state++){
                    uint index = state + block*params.subspaceDimension;
					averaged_results->fHg_ff[state] += host_results->fHg_ff[index] / params.nBlocks;
                    averaged_results->fUg_ff[state] += host_results->fUg_ff[index] / params.nBlocks;
					averaged_results->fg_ff[state] += host_results->fg_ff[index] / params.nBlocks;
					averaged_results->gg_ff[state] += host_results->gg_ff[index] / params.nBlocks;
              
                    for(int J=0; J<params.numberOfSectors; J++){
                        averaged_results->r_ct[J + params.numberOfSectors*state] += host_results->r_ct[J + params.numberOfSectors*index] / params.nBlocks;
                        averaged_results->r_st[J + params.numberOfSectors*state] += host_results->r_st[J + params.numberOfSectors*index] / params.nBlocks;
                    }
				}
			}
		
			// 8) Output
			acceptance = *(averaged_results->acceptance);

			for(int state=0; state<params.subspaceDimension; state++){
                if( angular_momenta_list[state] == angular_momenta_list[edgeMode] ){ // Enforce angular momentum conservation
                    H(edgeMode, state) = averaged_results->fHg_ff[state] / sqrt(averaged_results->gg_ff[state]);
                    M(edgeMode, state) = averaged_results->fg_ff[state] / sqrt(averaged_results->gg_ff[state]);
                }

                U(edgeMode, state) = averaged_results->fUg_ff[state] / sqrt(averaged_results->gg_ff[state]); // In principle obeys selection rules as well - but can't know them in advance

                for(int J=0; J<params.numberOfSectors; J++){
                    if( std::abs( angular_momenta_list[state] - angular_momenta_list[edgeMode]) == angular_momentum_sectors[J] ){ // Enforce angular momentum conservation
                        RC(edgeMode, state, J) = averaged_results->r_ct[J + params.numberOfSectors*state] / sqrt(averaged_results->gg_ff[state]);
                        RS(edgeMode, state, J) = averaged_results->r_st[J + params.numberOfSectors*state] / sqrt(averaged_results->gg_ff[state]);
                    }
                }
			}

			// 9) Free the device and the temporary variables on the host			
			gpuErrchk( cudaFree(tmp_results->acceptance) );
			gpuErrchk( cudaFree(tmp_results->fHg_ff) );
            gpuErrchk( cudaFree(tmp_results->fUg_ff) );
			gpuErrchk( cudaFree(tmp_results->fg_ff) );
			gpuErrchk( cudaFree(tmp_results->gg_ff) );
            gpuErrchk( cudaFree(tmp_results->r_ct) );
            gpuErrchk( cudaFree(tmp_results->r_st) );
			delete tmp_results;

			gpuErrchk( cudaFree(device_results) );

			delete[] host_results->acceptance;
			delete[] host_results->fHg_ff;
            delete[] host_results->fUg_ff;
			delete[] host_results->fg_ff;
			delete[] host_results->gg_ff;
            delete[] host_results->r_ct;
            delete[] host_results->r_st;
			delete host_results;

			delete averaged_results->acceptance;
			delete[] averaged_results->fHg_ff;
            delete[] averaged_results->fUg_ff;
			delete[] averaged_results->fg_ff;
			delete[] averaged_results->gg_ff;
            delete[] averaged_results->r_ct;
            delete[] averaged_results->r_st;
			delete averaged_results;				
						
			return time/1000.; // Returns the elapsed time (in seconds)
		};

		// Observables to be filled-in
		cmatrix<double> H(params.subspaceDimension, params.subspaceDimension, 0);
		cmatrix<double> M(params.subspaceDimension, params.subspaceDimension, 0);	
        cmatrix<double> U(params.subspaceDimension, params.subspaceDimension, 0);

        rank3tensor<complex<double>> RC(params.subspaceDimension, params.subspaceDimension, params.numberOfSectors, 0);
        rank3tensor<complex<double>> RS(params.subspaceDimension, params.subspaceDimension, params.numberOfSectors, 0);

		// Now loop over all the edge modes...
		for(int edgeMode=0; edgeMode<P.subspaceDimension; edgeMode++){
			// Print some info
			std::cout << "Sampling P[" << edgeMode << "]: "; 
			P.printEdgePartition(edgeMode);

			// Optimal Monte Carlo step
			std::cout << "\n\tDetermining the optimal Markov-Chain random step to achieve " << params.targetAcceptance << " acceptance rate.";
			optimal_step(params, devState, dev_partitions, edgeMode);
			std::cout << " Optimal step: " << params.randomStep << "\n\n";

			// Measure
			float time = measure( edgeMode, H, M, U, RC, RS );
			printf("\n\tFinshed. Total number of samples: %lu. Elapsed time: %.1fs\n", params.totalSamplesNumber, time);
			printf("\tAverage accepance\t%.2lf\n\n",  acceptance );
		}

		// Symmetrize the results
		auto make_mat_self_adjoint = [](cmatrix<double> &mat){
			cmatrix<double> tmp = (mat + mat.adjoint()) / 2.;
			mat = tmp;
			return;
		};

		auto make_tens_self_adjoint = [](rank3tensor<complex<double>> &t){
            rank3tensor<complex<double>> tmp( t.s1(), t.s2(), t.s3() );

            for( int J=0; J<t.s3(); J++ ){

                for(int state1=0; state1<t.s1(); state1++){
                    for(int state2=0; state2<t.s2(); state2++){
                        tmp(state1, state2, J) = (t(state1, state2, J) + conj(t(state2, state1, J))) / 2.;
                    }
                }
            }

			return tmp;
		};

		make_mat_self_adjoint(H);
		make_mat_self_adjoint(M);
        make_mat_self_adjoint(U);

        make_tens_self_adjoint(RC);
        make_tens_self_adjoint(RS);

		// Save 
		std::cout << "Saving the matrices (you can post-process to compute statistical errorbars)\n";

		std::string dir_path_H = fmt::format("../output/N={}_m={}/ExternalExcitation/Statistics/Hamiltonian", sys_params::particlesNumber, sys_params::inverseFilling );
		std::filesystem::create_directories(dir_path_H);

		std::string dir_path_M = fmt::format("../output/N={}_m={}/ExternalExcitation/Statistics/Metric", sys_params::particlesNumber, sys_params::inverseFilling );
		std::filesystem::create_directories(dir_path_M);

		std::string dir_path_U = fmt::format("../output/N={}_m={}/ExternalExcitation/Statistics/ExcitationPotential", sys_params::particlesNumber, sys_params::inverseFilling );
		std::filesystem::create_directories(dir_path_U);  

		std::string file_name = fmt::format("{}.tsv", fileNumber);
		std::ofstream outH( dir_path_H + "/" + file_name );
		std::ofstream outM( dir_path_M + "/" + file_name );
        std::ofstream outU( dir_path_U + "/" + file_name );

		for( int state1 = 0; state1 < params.subspaceDimension; state1++ ){
			for( int state2 = 0; state2 < params.subspaceDimension; state2++ ){
				outH << fmt::format("{}\t{}\t{:.6f}\t{:.6f}\n", state1, state2, re(H(state1,state2)), im(H(state1,state2)));
				outM << fmt::format("{}\t{}\t{:.6f}\t{:.6f}\n", state1, state2, re(M(state1,state2)), im(M(state1,state2)));
                outU << fmt::format("{}\t{}\t{:.6f}\t{:.6f}\n", state1, state2, re(U(state1,state2)), im(U(state1,state2)));
			}
		}
        
        for( int J=0; J<params.numberOfSectors; J++){
            std::string dir_path_RCS = fmt::format("../output/N={}_m={}/ExternalExcitation/Statistics/EdgeDenstiy_FourierTransform/dL={}", sys_params::particlesNumber, sys_params::inverseFilling, angular_momentum_sectors[J] );
            std::filesystem::create_directories(dir_path_RCS);           
            std::ofstream outRCS( dir_path_RCS + "/" + file_name );

            for( int state1 = 0; state1 < params.subspaceDimension; state1++ ){
                for( int state2 = 0; state2 < params.subspaceDimension; state2++ ){
                    outRCS << fmt::format("{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n", state1, state2, re(RC(state1,state2,J)), im(RC(state1,state2,J)), re(RS(state1,state2,J)), im(RS(state1,state2,J)));
                }
            }
        }

        // Free what's left
        gpuErrchk( cudaFree(devState) ); 
        gpuErrchk( cudaFree(dev_angular_momentum_sectors) );

        return;
    }



}


#endif