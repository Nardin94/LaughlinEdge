#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <Eigen/Dense>

namespace timeEvolution
{

	class schrodinger{
		private:
			size_t dim; // Dimension

            Eigen::MatrixXcd effective_hamiltonian; // M^{-1} H: the inverse metric applied to the Hamiltonian 
            Eigen::MatrixXcd effective_excitation; // M^{-1} V: the inverse metric applied to the excitation        
						
			std::function<double(double)> F; // Time dependence of the excitation

		public:
			schrodinger(size_t hilberSpaceDimension);

			//Import the stuff needed
			void set_operators( Eigen::MatrixXcd &M, Eigen::MatrixXcd &H, Eigen::MatrixXcd &V );
	
			// Set the time function
			void set_timeDependence(std::function<double(double)> tF);
			
			//Compute the derivative for the stepper
			void operator() (const std::vector<std::complex<double>> &C, std::vector<std::complex<double>> &dCdt, const double t);
	};

	class observer{
		public:
            // Vector to store the outputs
            struct results{
                double time;
                Eigen::VectorXcd edgeDensitity_fts;
            };


			observer(uint hilberSpaceDimension, uint angularMomentumSectors);

			//Import the stuff needed
			void set_metric(Eigen::MatrixXcd &M);
            void set_edgeDensityMatrices(std::vector<Eigen::MatrixXcd> &edm);
			void set_maxTime(double T_end);
			void memory_reserve( uint ansatz_size );
			
			// Retrieve the results
			std::vector<results> getResults() const;
				
			//Print evaluation time; compute and push back results
			void operator()(const std::vector<std::complex<double>> &C, const double t);

		private:
			uint dim; // The edge Hilbert space dimension
            uint numberOfSectors; // The number of angular momentum sectors
			
			double T_max;
			
            // Print the progress
			void progress_bar(double percentage, double norm);
			
            // To check the norm of the time-evolved state, the M-norm of C is needed: <\psi|\psi> = C_\beta^* M_{\beta\alpha} C_\alpha
            Eigen::MatrixXcd metric;
			double norm_check(const std::vector<std::complex<double>> &C);

            // Reconstruct the density
            std::vector<Eigen::MatrixXcd> rho_l; // The matrix elements of the edge-density operator at angular momentum l
            Eigen::VectorXcd edgeDensity_reconstruct( const std::vector<std::complex<double>> &C );

            std::vector<results> output_results;
	};    


    void edgeDensityResponse_compute( 	uint particlesNumber, uint inverseFilling, 
										std::vector<int> angular_momenta, 
										std::function<double(double)> temporal_excitation_profile, 
										double end_time, double save_step, 
										int fileNumber = 0);
        
}


#endif