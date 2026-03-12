#include "./complex_numbers.h"
#include "./tensors.h"

#include <vector>
#include <Eigen/Dense>
// #define EIGEN_NO_CUDA 1

#ifndef GENEIG_H
#define GENEIG_H

    // Exact-diagonalization and eigenbasis rotation
    namespace ges{

        class generalizedEigenproblem{		
            private:
                Eigen::MatrixXcd H0;
                Eigen::MatrixXcd M0;

                Eigen::VectorXd eigenvalues;
                Eigen::MatrixXcd eigenvectors;

                Eigen::MatrixXcd mat2eigenMat(cmatrix<double> &mat) const;
                std::vector<double> eigenVec2stdVec(Eigen::VectorXd &vec) const;

            public:
                generalizedEigenproblem(cmatrix<double> &H, cmatrix<double> &M);
                    
                std::vector<double> extract_spectrum(); // Returns the eigenvalues of the generalized eigenproblem H\psi = e M\psi
                std::vector<std::pair<double,double>> extract_dsf_coefficients(cmatrix<double> &VC, cmatrix<double> &VS); // Returns the energies and edge dynamic structure factor

        };
    }

#endif