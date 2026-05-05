#include "../green_function.h"

namespace greenFunctionMC {
	
	////////////////////////////////////////////////////////////////////
	// Members of the montecarlo class
	// specialized for spectral function and occupation numbers (need one extra particle and different sampling strategy)
		
		/////////////////////////////////////
		// Initialization and "destructor"
		__device__ montecarlo::montecarlo(	curandState* state, 
											int thread_id, 
											monteCarloParameters &simulation_parameters, 
											integer_partitions::partition* dev_partitions, 
											int samplingLeftEdgeState, int samplingRightEdgeState,
											bool mc_step_test
										 ) : tid(thread_id), params(simulation_parameters) {
			localState = state[tid];
			
			device_partitions = dev_partitions; // copy a pointer to the partitions (on global mem)
			sampling_left_partition = device_partitions[samplingLeftEdgeState].partitionArray; // pointer to the bra-partition used for the sampling
			left_partition_size = device_partitions[samplingLeftEdgeState].size; // its size

			sampling_right_partition = device_partitions[samplingRightEdgeState].partitionArray; // pointer to the ket-partition used for the sampling
			right_partition_size = device_partitions[samplingRightEdgeState].size; // its size			
			
			// Check whether maxDegree, extendedMaxDegree and subspaceDimension have been set
			if( params.subspaceDimension == -1 ){
				printf("\nYou forgot to .setEdgeSpaceInfo to set the subspace dimension and the maximal polynomial degree");
				return;
			}	
			
			// Initialize the positions of the particles, and their norms
			particle_positions_initialize();
			for(int i=0; i<=sys_params::particlesNumber; i++){
				position_norm[i] = norm(position[i]);
			}
			
			// Initialize the relevant e_k used in the mcmc (not all of them, used only for observable calculation)
			sampling_ek_construct();
			
			// Burn-in
			if( mc_step_test ){
				// This is the fast burn-in
				burnin( params.targetAcceptanceMoves );
			}
			else{
				// More lengthy burn-in
				burnin( params.burnInMoves );
			}
						
			return;
		}
		
		__device__ void montecarlo::cleanup(){
			// Release the dynamically allocated memory, if any

			return;
		}

		__device__ curandState montecarlo::stateRelease(){
			return localState;
		}

		//////////////////////
		// PUBLIC FUNCTIONS
		__device__ float montecarlo::burnin_acceptance_get() const{ // Returns the acceptance after burn-in
			return burnin_acceptance;
		}

		//////////////////////
		// PRIVATE FUNCTIONS
		// Progress bar
		__device__ void montecarlo::progress_bar(float percentage){
			int barWidth = 50;

			printf("\t[");
			int pos = int(barWidth * percentage / 100. + 0.1);
			for (int i = 0; i < barWidth; ++i) {
				if (i <= pos) printf("=");
				else printf(" ");
			}
			printf("]  ≈ %.0lf %%\r", percentage);
			
			return;
		}

		// Power sum symmetric polynomials
		__device__ void montecarlo::sampling_ek_construct(){
			for(int j=0; j<=sys_params::particlesNumber; j++){
				scaled_position[j]	= position[j] * sys_params::R0_reciprocal;
			}
			
			for(int i=0; i<left_partition_size; i++){
				current_left_ek[i] = 0;
				
				for(int j=0; j<sys_params::particlesNumber; j++){
					current_left_ek[i] += icpow(scaled_position[j], sampling_left_partition[i].number);
				}		
			}

			for(int i=0; i<right_partition_size; i++){
				current_right_ek[i] = 0;
				
				for(int j=1; j<=sys_params::particlesNumber; j++){
					current_right_ek[i] += icpow(scaled_position[j], sampling_right_partition[i].number);
				}		
			}			
			
			return;	
		}

		__device__ void montecarlo::sampling_left_ek_update( const complex<float> new_pos, const complex<float> old_pos){
			for(int i=0; i<left_partition_size; i++){
				proposed_left_ek[i] = current_left_ek[i] - icpow(old_pos * sys_params::R0_reciprocal, sampling_left_partition[i].number) + icpow(new_pos * sys_params::R0_reciprocal, sampling_left_partition[i].number);	
			}
			return;
		}

		__device__ void montecarlo::sampling_right_ek_update( const complex<float> new_pos, const complex<float> old_pos){
			for(int i=0; i<right_partition_size; i++){
				proposed_right_ek[i] = current_right_ek[i] - icpow(old_pos * sys_params::R0_reciprocal, sampling_right_partition[i].number) + icpow(new_pos * sys_params::R0_reciprocal, sampling_right_partition[i].number);	
			}
			return;
		}		

		__device__ complex<double> montecarlo::sampling_left_symPol_construct( const complex<double> *ek ){
			// Returns a single symmetric polynomial in N variables of degree dM
			// 		P_α = prod_k(e_(k_α)^repetitions(α)) 
			//			where α labels one of the partitions of dM (bounded to N variables) and repetitions(α) is the number of time the same number is repeated in the partition of dM
			
			complex<double> poly = icpow(ek[0], sampling_left_partition[0].repetitions);
			
			for(int i=1; i<left_partition_size; i++){
				poly *= icpow(ek[i], sampling_left_partition[i].repetitions);
			}
			
			return poly;
		}

		__device__ complex<double> montecarlo::sampling_right_symPol_construct( const complex<double> *ek ){
			// Returns a single symmetric polynomial in N variables of degree dM
			// 		P_α = prod_k(e_(k_α)^repetitions(α)) 
			//			where α labels one of the partitions of dM (bounded to N variables) and repetitions(α) is the number of time the same number is repeated in the partition of dM
			
			complex<double> poly = icpow(ek[0], sampling_right_partition[0].repetitions);
			
			for(int i=1; i<right_partition_size; i++){
				poly *= icpow(ek[i], sampling_right_partition[i].repetitions);
			}
			
			return poly;
		}

		// Initializing the positions of the particles
		__device__ void montecarlo::particle_positions_initialize(){		
			// The particles are randomly distributed in a square of size A = Rcl x Rcl
			//		Each occupies an area ~ A/N -> average distance between particles should be sqrt(N/A) = sqrt(N) / Rcl = 1 / sqrt(2m)
			//		Each time a configuration is drawn, check wether the distance with the other particles is large enough 
			//		This way, their positions are correlated and it should be easier to thermalize the configuration
			for(int i=0; i<=sys_params::particlesNumber; i++){
				if(i!=0){
					while(true){
						float x = - sys_params::RCl + curand_uniform(&localState) * sys_params::DCl;
						float y = - sys_params::RCl + curand_uniform(&localState) * sys_params::DCl;
						 
						complex<float> z = complex<float>(x, y);
						
						// Check that the generated positions are not the positions of another particle          
						bool well_spaced = true;
						for(int j=0; j<i; j++){
							// If two particles are closer than 1/sqrt(2m) the position is rejected
							if( cabs(z - position[j]) < 1./sqrt(2.*sys_params::inverseFilling) ){
								well_spaced = false;
								break;
							}
						}
						
						// If all coordinates are different, save them and move on to next particle
						if(well_spaced == true){
							position[i] = z;
							
							break;
						}
					}
				}
				else{
					float x = - sys_params::RCl + curand_uniform(&localState) * sys_params::DCl;
					float y = - sys_params::RCl + curand_uniform(&localState) * sys_params::DCl;         
					
					position[i] = complex<float>(x, y);
				}
			}
			
			return;
		}

		// Burn-in moves
		__device__ void montecarlo::burnin( uint moves_number ){
			// Runs an empty mcmc, returning the number of accepted single-particle moves
			// Used to thermalize / fix the Markov-chain step

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
			  
			for(int iter=0; iter<moves_number; iter++){
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
			}
			
			burnin_acceptance = (float)local_acceptance / ( moves_number * sys_params::particlesNumber);
			
			return;
		}


}
