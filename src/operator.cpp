#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

#include "operator.hpp"
#include "Eigen/src/Core/Matrix.h"

using namespace Spectra;



///// DIAGONALIZATION /////


    /* IMPLICITLY RESTARTED LANCZOS METHOD (IRLM) */

/* implement the IRLM for a sparse matrix to find the smallest nb_eigen eigenvalues of a sparse matrix */
Eigen::VectorXcd Op::IRLM_eigen(Eigen::SparseMatrix<double> O,int nb_eigen, Eigen::MatrixXcd& eigenvectors) {
    SparseGenMatProd<double> op(O); // create a compatible matrix object
    GenEigsSolver<SparseGenMatProd<double>> eigs(op, nb_eigen, 2 * nb_eigen+1); // create an eigen solver object
    eigs.init();
    [[maybe_unused]] int nconv = eigs.compute(Spectra::SortRule::SmallestReal); // solve the eigen problem
    if (eigs.info() != Spectra::CompInfo::Successful) { // verify if the eigen search is a success
        throw std::runtime_error("Eigenvalue computation failed.");
    }
    Eigen::VectorXcd eigenvalues = eigs.eigenvalues(); // eigenvalues of the hamiltonian
    eigenvectors = eigs.eigenvectors(); // eigenvectors of the hamiltonian
    return eigenvalues;
}


    /* FULL ORTHOGONALIZATION LANCZOS METHOD (FOLM) */

/* implement the FOLM for a sparse matrix for nb_iter iterations starting with vector v_0 */
void Op::FOLM_diag(Eigen::SparseMatrix<double> O, int nb_iter, Eigen::VectorXd& v_0, Eigen::MatrixXd& T, Eigen::MatrixXd& V) {
    T.resize(nb_iter, nb_iter); //resize the matrix T to a matching size
    V.resize(O.size(), nb_iter); //resize the matrix V to a matching size

    v_0.normalize(); // normalize the starting vector v_0
    V.col(0) = v_0; // put the first vector v_0 in the matrix V

    double alpha, beta;

    // Lanczos algorithm for nb_iter iterations
    for (int i = 0; i < nb_iter; i++) {
        Eigen::VectorXd w = O * V.col(i); // calculate the next vector w
        alpha = (V.col(i)).dot(w);
        for (int j = 0; j < i; j++) {
            w = w - (V.col(j)).dot(w) * V.col(j); // orthogonalize the vector w with respect to the previous vectors of the Krylov basis
        }
        beta = w.norm(); // calculate the norm of the vector w
        if (beta < 1e-6) {
            break; // if beta is null or almost null the algorithm stops
        }
        else {
            w = w / beta; // normalize the vector w
            if (i + 1 < nb_iter) {
                V.col(i + 1) = w; // add the vector w to the matrix V of vectors of the Krylov basis
            }
            T(i, i) = alpha; // add the ith diagonal element of the tridiagonal matrix T
            if (i > 0) {
                T(i, i - 1) = beta; // add the ith non-diagonal element of the tridiagonal matrix T
                T(i - 1, i) = beta; // add the ith non-diagonal element of the tridiagonal matrix T
            }
        }
    }
}

/* Calculate the approximate eigenvalues and eigenvectors of the hamiltonian using the Lanczos algorithm */
Eigen::VectorXd Op::FOLM_eigen(Eigen::SparseMatrix<double> O, int nb_iter, Eigen::MatrixXd& eigenvectors){
    Eigen::MatrixXd V(O.size(), nb_iter); // initialize the matrix V of vectors of the Krylov basis
    Eigen::MatrixXd T(nb_iter,nb_iter); // initialize the tridiagonal matrix T
    Eigen::VectorXd v_0 = Eigen::VectorXd::Random(O.size()); // initialize a random vector v_0
    FOLM_diag(O, nb_iter, v_0, T, V); // tridiagonalize the hamiltonian using Lanczos algorithm
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(T); // solve the eigen problem for T
    if (eigensolver.info() != Eigen::Success) { // verify if the eigen search is a success
        throw std::runtime_error("Eigenvalue computation failed.");
    }
    eigenvectors = V * eigensolver.eigenvectors(); // eigenvectors of the hamiltonian
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues(); // eigenvalues of the hamiltonian
    return eigenvalues;
}


    /* EXACT DIAGONALIZATION */

/* Calculate the exact eigenvalues and eigenvectors of the hamiltonian by an exact diagonalization */
Eigen::VectorXd Op::exact_eigen(Eigen::SparseMatrix<double> O, Eigen::MatrixXd& eigenvectors) {
    Eigen::MatrixXd dense_smat = Eigen::MatrixXd(O); // convert sparse matrix to dense matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(dense_smat); // solve the eigen problem for the hamiltonian
    if (eigensolver.info() != Eigen::Success) { // verify if the eigen search is a success
        throw std::runtime_error("Eigenvalue computation failed.");
    }
    eigenvectors = eigensolver.eigenvectors(); // eigenvectors of the hamiltonian
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues(); // eigenvalues of the hamiltonian
    return eigenvalues;
}