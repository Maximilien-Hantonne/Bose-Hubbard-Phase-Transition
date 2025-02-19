#pragma once

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>


/**
 * @brief Class representing an operator in a quantum system.
 * 
 * This class provides methods for initializing, manipulating, and diagonalizing operators represented as sparse matrices.
 */

class Operator {
private:

// DIAGONALIZATION : 

    /* implement the Full Orthogonalization Lanczos Method for a sparse matrix for nb_iter iterations starting with vector v_0 */
    static void FOLM_diag(Eigen::SparseMatrix<double> O, int nb_iter, Eigen::VectorXd& v_0, Eigen::MatrixXd& T, Eigen::MatrixXd& V);

    /* sort eigenvalues and eigenvectors in descending order */
    void sort_eigen(Eigen::VectorXd& eigenvalues, Eigen::MatrixXd& eigenvectors) const;

public:

// DIAGONALIZATION : 

    /**
     * @brief Calculate the approximate eigenvalues and eigenvectors of the Hamiltonian using the Implicitly Restarted Lanczos Method.
     * 
     * @param O The sparse matrix to diagonalize.
     * @param nb_eigen The number of eigenvalues to calculate.
     * @param eigenvectors An empty matrix to store the eigenvectors.
     * @return Eigen::Matrix<std::complex<double>> The vector of eigenvalues.
     * @warning Ensure that nb_eigen is greater than 1.
     */
    static Eigen::VectorXcd IRLM_eigen(Eigen::SparseMatrix<double> O, int nb_eigen, Eigen::MatrixXcd& eigenvectors);

    /**
    * @brief Calculate the approximate eigenvalues and eigenvectors of the Hamiltonian using the Full Orthogonalization Lanczos Method.
    * 
    * @param O The sparse matrix to diagonalize.
    * @param nb_iter The number of iterations.
    * @param eigenvectors An empty matrix to store the eigenvectors.
    * @return Eigen::VectorXd The vector of eigenvalues.
    * @warning Ensure that nb_iter is greater than 1. The calculation might be wrong if the number of iterations is too low.
    */
    static Eigen::VectorXd FOLM_eigen(Eigen::SparseMatrix<double> O, int nb_iter, Eigen::MatrixXd& eigenvectors);

    /**
     * @brief Calculate the exact eigenvalues and eigenvectors of the Hamiltonian by an exact diagonalization.
     * 
     * @param O The sparse matrix to diagonalize.
     * @param eigenvectors An empty matrix to store the eigenvectors.
     * @return Eigen::Matrix<double> The vector of eigenvalues.
     * @warning This function is computationally expensive and should be used with caution.
     */
    static Eigen::VectorXd exact_eigen(Eigen::SparseMatrix<double> O, Eigen::MatrixXd& eigenvectors);

};