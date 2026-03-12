#include "./edge_montecarlo.h"

namespace edgeMC {
	
	////////////////////////////////////////////////////////////////////
	// Members of the montecarlo class
		
		/////////////////////////////////////
		// Initialization and "destructor"
		__device__ montecarlo::montecarlo(	curandState* state, 
											int thread_id, 
											monteCarloParameters &simulation_parameters, 
											integer_partitions::partition* dev_partitions, 
											int samplingEdgeState,
											bool mc_step_test
										 ) : tid(thread_id), params(simulation_parameters) {
			localState = state[tid];
			
			device_partitions = dev_partitions; // copy a pointer to the partitions (on global mem)
			sampling_partition = device_partitions[samplingEdgeState].partitionArray; // pointer to the partition used for the sampling
			partition_size = device_partitions[samplingEdgeState].size; // its size
			
			// Check whether maxDegree, extendedMaxDegree and subspaceDimension have been set
			if( params.subspaceDimension == -1 ){
				printf("\nYou forgot to .setEdgeSpaceInfo to set the subspace dimension and the maximal polynomial degree");
				return;
			}	
			
			// Initialize the positions of the particles, and their norms
			particle_positions_initialize();
			for(int i=0; i<sys_params::particlesNumber; i++){
				position_norm[i] = norm(position[i]);
			}
			
			// Initialize the trivial parts of e_k = \sum z_i^k and e_k^r
			ek_list[0] = sys_params::particlesNumber;
			ekr_list[0] = 1;
			for(int r=1; r<params.extendedMaxDegree; r++){
				ekr_list[params.extendedMaxDegree * r] = ekr_list[params.extendedMaxDegree * (r-1)] * ek_list[0];
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

		// Hamiltonian stuff. In particular, confinement, dynamic structure factor	
		__device__ double montecarlo::confinement(){	
			double sum = 0;
			for(int i=0; i<sys_params::particlesNumber; i++){
				sum += pow(sqrt(position_norm[i]) * sys_params::RCl_reciprocal, sys_params::confinementExponent);
			}

			return sum * sys_params::confinementStrength;
		}

		__device__ double montecarlo::cosine_excitation(){
			double sum = 0;
			for(int i=0; i<sys_params::particlesNumber; i++){
				sum += cos(params.injectedAngularMomentum * angle[i]);
			}
			
			return sum;
		}

		__device__ double montecarlo::sine_excitation(){
			double sum = 0;
			for(int i=0; i<sys_params::particlesNumber; i++){
				sum += sin(params.injectedAngularMomentum * angle[i]);
			}
			
			return sum;
		}

		// Power sum symmetric polynomials
		__device__ void montecarlo::ek_construct(){
			// Returns an array of all the e_k = sum_i z_i^k
			// of order 0 <= k <= kmax
			// These aren't really elementary symmetric polynomials. They are the "elementary" building blocks 
			
			// Constructed iteratively
			// p[i] = z[i]
			// e_1 = sum( p[i] )
			// 1) p[i] *= z[i]
			// 2) e_2 = sum( p[i] )
			// repeat 1-2
			
			
			//ek_list[0] = sys_params::particlesNumber; // Already initialized
			
			for(int j=0; j<sys_params::particlesNumber; j++){
				scaled_position[j] = position[j] * sys_params::R0_reciprocal;
				position_powers[j] = scaled_position[j];
			}
				
			for(int k=1; k<params.maxDegree; k++){
				ek_list[k] = 0;
				for(int j=0; j<sys_params::particlesNumber; j++){
					ek_list[k] += position_powers[j];
					position_powers[j] *= scaled_position[j];
				}
			}

			ek_list[params.maxDegree] = 0;
			for(int j=0; j<sys_params::particlesNumber; j++){
				ek_list[params.maxDegree] += position_powers[j];
			}
			
			return;
		}

		__device__ void montecarlo::ekr_construct(){
			// Computes all the powers e_k^r
			// e_k^0 = 1 is trivial (and actually never used) yet convenient for indexing
			// Indices are flattened as e_k^r ---> ekPowers[k + maxDegree * r]
			// Therefore accessing ekPowers[k + extendedMaxDegree * r] gives (\sum_i z_i^k)^r 
			
			for(int k=1; k<params.extendedMaxDegree; k++){
				ekr_list[k] = 1; // e_k^0 = e_k.
				
				for(int r=1; r<params.extendedMaxDegree; r++){
					// Recursively fill e_k^r = e_k^(r-1) * e_k
					ekr_list[k + params.extendedMaxDegree * r] = ekr_list[k + params.extendedMaxDegree * (r-1)] * ek_list[k];
				}
			}
				
			return;
		}

		__device__ complex<double> montecarlo::symPol_construct(integer_partitions::partition_element* p, int p_size){
			// Returns all the symmetric polynomials in N variables of degree dM, using the elementary polynomials
			
			complex<double> poly = ekr_list[p[0].number + params.extendedMaxDegree * p[0].repetitions];
			for(int i=1; i<p_size; i++){
				poly *= ekr_list[p[i].number + params.extendedMaxDegree * p[i].repetitions];
			}
			
			return poly;
		}

		__device__ void montecarlo::sampling_ek_construct(){
			for(int j=0; j<sys_params::particlesNumber; j++){
				scaled_position[j]	= position[j] * sys_params::R0_reciprocal;
			}
			
			for(int i=0; i<partition_size; i++){
				current_ek[i] = 0;
				
				for(int j=0; j<sys_params::particlesNumber; j++){
					current_ek[i] += icpow(scaled_position[j], sampling_partition[i].number);
				}		
			}
			
			return;	
		}

		__device__ void montecarlo::sampling_ek_update( const complex<float> new_pos, const complex<float> old_pos){	
			for(int i=0; i<partition_size; i++){
				proposed_ek[i] = current_ek[i] - icpow(old_pos * sys_params::R0_reciprocal, sampling_partition[i].number) + icpow(new_pos * sys_params::R0_reciprocal, sampling_partition[i].number);	
			}
			return;
		}

		__device__ complex<double> montecarlo::sampling_symPol_construct( const complex<double> *ek ){
			// Returns a single symmetric polynomial in N variables of degree dM
			// 		P_α = prod_k(e_(k_α)^repetitions(α)) 
			//			where α labels one of the partitions of dM (bounded to N variables) and repetitions(α) is the number of time the same number is repeated in the partition of dM
			
			complex<double> poly = icpow(ek[0], sampling_partition[0].repetitions);
			
			for(int i=1; i<partition_size; i++){
				poly *= icpow(ek[i], sampling_partition[i].repetitions);
			}
			
			return poly;
		}

		// Initializing the positions of the particles
		__device__ void montecarlo::particle_positions_initialize(){		
			// The particles are randomly distributed in a square of size A = Rcl x Rcl
			//		Each occupies an area ~ A/N -> average distance between particles should be sqrt(N/A) = sqrt(N) / Rcl = 1 / sqrt(2m)
			//		Each time a configuration is drawn, check wether the distance with the other particles is large enough 
			//		This way, their positions are correlated and it should be easier to thermalize the configuration
			for(int i=0; i<sys_params::particlesNumber; i++){
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
				
			complex<double> polynomial_current = sampling_symPol_construct( current_ek ); // The current calue of the sampling symmetric polynomial
			complex<double> polynomial_proposed; // Updated after the move
			  
			for(int iter=0; iter<moves_number; iter++){
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
					}
					else{
						// Restore the configuration
						position[i] = current_position;
						position_norm[i] = current_norm;
					}		
				}
			}
			
			burnin_acceptance = (float)local_acceptance / ( moves_number * sys_params::particlesNumber);
			
			return;
		}


}
