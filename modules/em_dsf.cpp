#include "./edge_montecarlo.h"

#include <complex>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace edgeMC {

	////////////////////////////////////////////////////////////////////
	// Compute the spectrum and the edge dynamic structure factor only -- i.e. samples the \sum_k(f|e^{i dM \theta_k}g)
	// Two different angular momenta are required: the first is restricted to be 0 (i.e. the Laughlin state), the second dM.
	// The code here exploits the symmetries to set the off-diagonal blocks of the Hamiltonian and overlaps
	// and the diagonal blocks of the edge density operator to 0.

	std::vector<double> dsf_rotate(cmatrix<double> &H, cmatrix<double> &M, cmatrix<double> &Vcos, cmatrix<double> &Vsin){
		// Convert matrices to eigen
		int rows = H.rows();
		int cols = H.cols();

		Eigen::MatrixXcd Ham(rows, cols);
		Eigen::MatrixXcd Met(rows, cols);

		for(int row=0; row<rows; row++){
			for(int col=0; col<cols; col++){
				Ham(row, col) = H(row, col);
				Met(row, col) = M(row, col);
			}
		}

		// Use eigen libraries to diagonalize
		Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcd> es(Ham, Met, Eigen::EigenvaluesOnly);
		Eigen::VectorXd eigenvalues = es.eigenvalues();

		// Convert eigenvalues to a std::vector
		std::vector<double> evals(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());

		// Return
		return evals;
	}
}
