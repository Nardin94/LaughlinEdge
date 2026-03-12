#include <iostream>
#include <filesystem>
#include <fstream>
#include <fmt/core.h>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <boost/numeric/odeint.hpp>

#include "dynamics.h"


namespace timeEvolution{

    using namespace std::complex_literals;

//////////////////////
// Schrodinger class

	schrodinger::schrodinger(size_t hilberSpaceDimension) : dim(hilberSpaceDimension){
	}

	//Import the stuff needed			
	void schrodinger::set_operators( Eigen::MatrixXcd &M, Eigen::MatrixXcd &H, Eigen::MatrixXcd &V ){
        Eigen::MatrixXcd inverse_metric = M.inverse();
        effective_hamiltonian = inverse_metric * H;
        effective_excitation = inverse_metric * V;

	    return;
	}
		
	// Set the time function
	void schrodinger::set_timeDependence(std::function<double(double)> tF){
		F = tF;
				
	    return;
	}
		
	// Fill in the vector of derivatives for the stepper
	void schrodinger::operator()(const std::vector<std::complex<double>> &C, std::vector<std::complex<double>> &dCdt, const double t){															
			
        for(uint row=0; row<dim; row++){
            dCdt[row] = 0;

            for(uint col=0; col<dim; col++){
                dCdt[row] += ( effective_hamiltonian(row, col) + F(t) * effective_excitation(row, col) ) * C[col];
            }
        
            dCdt[row] *= -1i;
		}

		return;
    }


//////////////////////
// Observer class 

		void observer::progress_bar(double percentage, double norm){  
			int barWidth = 50;

			printf("\t\t[");
			int pos = int(barWidth * percentage / 100. + 0.1);
			for (int i = 0; i < barWidth; ++i) {
				if (i <= pos) printf("=");
				else printf(" ");
			}
			printf("]  ≈ %.0lf %%\tTime-evolved-state norm = %.6lf\r", percentage, norm);
			
			return;
		}

		double observer::norm_check(const std::vector<std::complex<double>> &C){
			//Unitarity check 
			std::complex<double> S = 0.;
			for(uint row=0; row<dim; row++){
                for(uint col=0; col<dim; col++){
				    S += std::conj(C[row]) * metric(row, col) * C[col];	
                }
			}
				
			return std::real( S );
		}
			
		observer::observer(uint hilberSpaceDimension, uint angularMomentumSectors) : dim(hilberSpaceDimension), numberOfSectors(angularMomentumSectors){
            
        }
			
		// Import the stuff needed		
		void observer::set_metric(Eigen::MatrixXcd  &M){
			metric = M;
			return;
		}

        void observer::set_edgeDensityMatrices(std::vector<Eigen::MatrixXcd> &edm){
            rho_l = edm;
            return;
        }

		void observer::set_maxTime(double T_end){
			T_max = T_end;
			return;
		}

		void observer::memory_reserve( uint ansatz_size ){
			output_results.reserve( ansatz_size );
		}

        // Reconstruct the edge density
        Eigen::VectorXcd observer::edgeDensity_reconstruct( const std::vector<std::complex<double>> &C ){
            Eigen::VectorXcd edge_densities( numberOfSectors );

            for(uint l=0; l<numberOfSectors; l++){

                edge_densities[l] = 0.;

                for(uint row=0; row<dim; row++){
                    for(uint col=0; col<dim; col++){
                        edge_densities[l] += std::conj(C[row]) * rho_l[l](row, col) * C[col];	
                    }
                }
            }

            return edge_densities;
        }

		// Retrieve the results
		std::vector<observer::results> observer::getResults() const{
			return output_results;
		}

		//Print evaluation time and push back to results
		void observer::operator()(const std::vector<std::complex<double>> &C, const double t){
			progress_bar( 100. * t / T_max, norm_check(C) );
			Eigen::VectorXcd edge_densities = edgeDensity_reconstruct(C);
			
            observer::results res;
            res.time = t;
            res.edgeDensitity_fts = edge_densities;

			output_results.push_back( res );
	
			return;
		}

//////////////////////
// Time-evolution

    void edgeDensityResponse_compute( 	uint particlesNumber, uint inverseFilling, 
										std::vector<int> angular_momenta, 
										std::function<double(double)> temporal_excitation_profile, 
										double end_time, double save_step, 
										int fileNumber){		
		// Import the stuff
		std::string dir_path_M = fmt::format("../output/N={}_m={}/ExternalExcitation/Statistics/Metric", particlesNumber, inverseFilling );
		std::string dir_path_H = fmt::format("../output/N={}_m={}/ExternalExcitation/Statistics/Hamiltonian", particlesNumber, inverseFilling );
		std::string dir_path_U = fmt::format("../output/N={}_m={}/ExternalExcitation/Statistics/ExcitationPotential", particlesNumber, inverseFilling );

		std::string file_name = fmt::format("{}.tsv", fileNumber);

		auto read_matrix = [](std::string path_to_file){

			std::ifstream in(path_to_file);
			if(!in){
				throw std::runtime_error("Cannot open file: " + path_to_file);
			}

			int i, j;
			double re, im;

			int max_i = -1, max_j = -1;

			std::vector<std::tuple<int,int,std::complex<double>>> entries;

			while(in >> i >> j >> re >> im) {
				entries.emplace_back(i, j, std::complex<double>(re, im));

				if (i > max_i) max_i = i;
				if (j > max_j) max_j = j;
			}

			if(!in.eof()){
				throw std::runtime_error("Parsing error in " + path_to_file);
			}

			Eigen::MatrixXcd mat(max_i + 1, max_j + 1);
			mat.setZero();

			for (const auto& [row, col, val] : entries){
				mat(row, col) = val;
			}

			return mat;
		};

		Eigen::MatrixXcd M = read_matrix( dir_path_M + "/" + file_name );
		Eigen::MatrixXcd H = read_matrix( dir_path_H + "/" + file_name );
		Eigen::MatrixXcd U = read_matrix( dir_path_U + "/" + file_name );

		// Get edgespace dimension and number of angular momentum sectors
		uint angularMomentumSectors = angular_momenta.size();
		uint edgeSpaceDimension = H.rows();

		std::cout << "Time-evoluion in the [ ";
		for(uint l=0; l<angularMomentumSectors; l++){
			std::cout << angular_momenta[l] << " ";
		}
		std::cout << "] angular momentum subspaces. Number of edge states: " << edgeSpaceDimension << "\n";

		// Set the initial condition
		if( angular_momenta[0] != 0 ){
			std::cout << "The Laughlin state (assumed to be the initial state) is not included in the subspace!" << std::endl;
			exit(1);
		}

		std::vector<std::complex<double>> C_iv(edgeSpaceDimension, 0);
		C_iv[0] = 1.;

		// Importing the edge density matrix elements
		auto read_edgeDensity_matrix = [](std::string path_to_file){

			std::ifstream in(path_to_file);
			if(!in){
				throw std::runtime_error("Cannot open file: " + path_to_file);
			}

			int i, j;
			double reCos, imCos;
			double reSin, imSin;

			int max_i = -1, max_j = -1;

			std::vector<std::tuple<int,int,std::complex<double>,std::complex<double>>> entries;

			while(in >> i >> j >> reCos >> imCos >> reSin >> imSin){
				entries.emplace_back(i, j, std::complex<double>(reCos, imCos), std::complex<double>(reSin, imSin));

				if (i > max_i) max_i = i;
				if (j > max_j) max_j = j;
			}

			if(!in.eof()){
				throw std::runtime_error("Parsing error in " + path_to_file);
			}

			Eigen::MatrixXcd mat(max_i + 1, max_j + 1);
			mat.setZero();

			for (const auto& [row, col, cosVal, sinVal] : entries){
				mat(row, col) = cosVal + 1.i*sinVal; // Reconstruct e^{i l \theta}
			}

			return mat;
		};

		std::vector<Eigen::MatrixXcd> rho_l(angularMomentumSectors);

		for(uint l=0; l<angularMomentumSectors; l++){
			std::string dir_path_rhol = fmt::format("../output/N={}_m={}/ExternalExcitation/Statistics/EdgeDenstiy_FourierTransform/dL={}", particlesNumber, inverseFilling, angular_momenta[l] );
			
			rho_l[l] = read_edgeDensity_matrix( dir_path_rhol + "/" + file_name );
		}

		// Integration	
		std::cout << "\tSetting up the Schrodinger system...\n";
		
		schrodinger diff_eq_system( edgeSpaceDimension );
		diff_eq_system.set_operators( M, H, U );
		diff_eq_system.set_timeDependence( temporal_excitation_profile );
								
		observer obs( edgeSpaceDimension, angularMomentumSectors );
		obs.set_metric( M );
		obs.set_edgeDensityMatrices( rho_l );
		obs.set_maxTime( end_time );
		obs.memory_reserve( (uint)(end_time / save_step) );
			
		// Integrate with adaptive timesteps, but call the observer every dt
		std::vector<double> times( std::ceil( end_time / save_step ) );
		for(uint t=0; t<times.size(); t++){
			times[t] = t*save_step;
		}
		
		double err_abs = 1.e-10;
		double err_rel = 1.e-8;

		boost::numeric::odeint::integrate_times( make_controlled( err_abs , err_rel , boost::numeric::odeint::runge_kutta_cash_karp54<std::vector<std::complex<double>>>() ), 
																	diff_eq_system, 
																	C_iv, 
																	times.begin(), times.end(), save_step, 
																	std::ref(obs) );

		std::cout << "\n\tDone. Saving..." << std::endl;

		// Retrieve the output and save															
		std::vector<observer::results> output = obs.getResults();

        for( uint l=0; l<angularMomentumSectors; l++){
            std::string dir_path_out = fmt::format("../output/N={}_m={}/ExternalExcitation/Statistics/edgeDynamics/dL={}", particlesNumber, inverseFilling, angular_momenta[l] );
            std::filesystem::create_directories(dir_path_out);           
            std::ofstream out_file( dir_path_out + "/" + file_name );

            for( auto elem : output ){
				std::complex<double> z = elem.edgeDensitity_fts[l];
				out_file << fmt::format("{:.6f}\t{:.6f}\t{:.6f}\n", elem.time, std::real(z), std::imag(z));
            }
        }

		return;
	}

}