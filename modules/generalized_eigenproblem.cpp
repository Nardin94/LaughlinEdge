#include "./generalized_eigenproblem.h"

#include <complex>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
// #define EIGEN_NO_CUDA 1

namespace ges {

	using namespace std::complex_literals;

	////////////////////////////////////////////////////////////////////
	// Solves the generalized eigenproblem H\psi = E M\psi
	// Rotates the various observables (feeded in the same basis as those of H, M) 
	// onto the eigenvector basis

    Eigen::MatrixXcd generalizedEigenproblem::mat2eigenMat(cmatrix<double> &mat) const{
		int rows = mat.rows();
		int cols = mat.cols();

		Eigen::MatrixXcd eigenMat(rows, cols);

		for(int row=0; row<rows; row++){
			for(int col=0; col<cols; col++){
				eigenMat(row, col) = mat(row, col);
			}
		}

		return eigenMat;
	}
    
	std::vector<double> generalizedEigenproblem::eigenVec2stdVec(Eigen::VectorXd &vec) const{
		std::vector<double> evals(vec.data(), vec.data() + vec.size());

		return evals;
	}

    generalizedEigenproblem::generalizedEigenproblem(cmatrix<double> &H, cmatrix<double> &M){
		H0 = mat2eigenMat( H );
		M0 = mat2eigenMat( M );

		return;
	}

    std::vector<double> generalizedEigenproblem::extract_spectrum(){
		Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcd> es(H0, M0, Eigen::EigenvaluesOnly);
		eigenvalues = es.eigenvalues();

		std::vector<double> out = eigenVec2stdVec( eigenvalues );

		return out;
	}

	std::vector<std::pair<double,double>> generalizedEigenproblem::extract_dsf_coefficients(cmatrix<double> &VC, cmatrix<double> &VS){
		// Remove the Laughlin state sub-block
		Eigen::MatrixXcd h0 = H0.block(1, 1, H0.rows()-1, H0.cols()-1);
		Eigen::MatrixXcd m0 = M0.block(1, 1, M0.rows()-1, M0.cols()-1);
	
		// Get the size
		uint size = h0.rows();
		
		// Solve the generalized eigenproblem
		Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcd> es(h0, m0);
		
		eigenvectors = es.eigenvectors();
		eigenvalues = es.eigenvalues();

		// The combination V_dL = cos(dL \theta) - i sin(dL \theta) = e^{-i dL \theta} : it reduces the angular momentum of the state
		// The matrix elements we are interested in are therefore dsf_j = |(0,0| V_dL |dL,j)|^2
		// We rotate onto the power sum symmetric polynomials
		// |0,0) has a trivial decomposition (it is still a Laughlin state)
		// |dL,j) = \sum_{\alpha \in PSSP} \psi_{\alpha, j} |\alpha)
		// therfore
		// (0,0| V_dL |dL,j) = \sum_{\alpha \in PSSP} (0,0| V_dL |\alpha) \psi_{\alpha, j}
		// But we sampled (0,0| V_dL |\alpha): it's saved in V_dL(0,alpha)
		Eigen::RowVectorXcd Vc = mat2eigenMat( VC ).row(0).segment(1, size);
		Eigen::RowVectorXcd Vs = mat2eigenMat( VS ).row(0).segment(1, size);
		Eigen::RowVectorXcd V = Vc - 1i * Vs;
		
		std::vector<std::pair<double,double>> energy_dsf(size);

		for(uint edgeModeIndex=0; edgeModeIndex<size; edgeModeIndex++){
			//Eigen::VectorXcd edgeMode = eigenvectors.col(edgeModeIndex);
			
			//std::complex<double> matrix_element = 0;
			//for(int pssp=0; pssp<size; pssp++){
			//	matrix_element += V(pssp) * edgeMode(pssp);
			//}

			std::complex<double> matrix_element = V * eigenvectors.col(edgeModeIndex);
			energy_dsf[edgeModeIndex] = std::make_pair( eigenvalues[edgeModeIndex], std::norm(matrix_element) );			
		}

		return energy_dsf;
	}

}
