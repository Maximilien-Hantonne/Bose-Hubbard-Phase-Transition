#include <stdexcept>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

#include "operator.h"
#include "Eigen/src/Core/Matrix.h"

using namespace Spectra;

// REMARQUE : TROUVER LES FONCTIONS A RECODER AVEC CTRL + F "TODO"


    /* CONSTRUCTOR */

Operator::Operator(Eigen::SparseMatrix<double>&& smatrix) : O(std::move(smatrix)), D(smatrix.rows()), ref(1e-6) {}


    /* UTILITY FUNCTIONS */

int Operator::size() const {
	return D;
}


///// BASIS FUNCTIONS /////


    /* IDENTITY OPERATOR */

/* create the operator Identity*/
Operator Operator::Identity(int size) {
    Eigen::SparseMatrix<double> identity(size, size);
    identity.setIdentity();
    return identity;
}


    /* ADDITION AND MULTIPLICATION */

/* add a matrix to an operand of type SparseMatrix with same size */
Operator Operator::operator + (const Operator& operand) const {
    if (this->O.rows() != operand.O.rows() || this->O.cols() != operand.O.cols()) { // verify that the operands have matching size
        throw std::invalid_argument("Matrix should have matching size.");
    }
    Eigen::SparseMatrix<double> result_matrix = (this->O + operand.O).pruned(ref); // removes elements smaller than ref
    return result_matrix;
}

/* multiply a sparse matrix by a multiplicand of type SparseMatrix with same size */
Operator Operator::operator * (const Operator& multiplicand) const {
    if (this->O.cols() != multiplicand.O.rows()) {
        throw std::invalid_argument("Number of columns of multiplier must equal number of rows of multiplicand."); // verify that the number of columns of multiplier equals number of rows of multiplicand
    }
    Eigen::SparseMatrix<double> result_matrix = (this->O * multiplicand.O).pruned(ref); // removes elements smaller than ref
    return result_matrix;
}

/* multiply a sparse matrix by a vector with concomitant size */
Eigen::VectorXd Operator::operator * (const Eigen::VectorXd& vector) const {
    if (this->O.cols() != vector.size()) { // verify that number of matrix columns equals vector size
        throw std::invalid_argument("Number of matrix columns must equal vector size.");
    }
    return this->O * vector;
}

/* multiply a sparse matrix by a scalar */
Operator Operator::operator * (double scalar) const {
    Eigen::SparseMatrix<double> result_matrix = (this->O * scalar).pruned(ref); // removes elements smaller than ref
    return result_matrix;
}




///// DIAGONALIZATION /////


    /* IMPLICITLY RESTARTED LANCZOS METHOD (IRLM) */

/* implement the IRLM for a sparse matrix to find the smallest nb_eigen eigenvalues of a sparse matrix */
Eigen::VectorXcd Operator::IRLM_eigen(int nb_eigen, Eigen::MatrixXcd& eigenvectors) const {
    SparseGenMatProd<double> op(this->O); // create a compatible matrix object
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
void Operator::FOLM_diag(int nb_iter, Eigen::VectorXd& v_0, Eigen::MatrixXd& T, Eigen::MatrixXd& V) const {
    T.resize(nb_iter, nb_iter); //resize the matrix T to a matching size
    V.resize(D, nb_iter); //resize the matrix V to a matching size

    v_0.normalize(); // normalize the starting vector v_0
    V.col(0) = v_0; // put the first vector v_0 in the matrix V

    double alpha, beta;

    // Lanczos algorithm for nb_iter iterations
    for (int i = 0; i < nb_iter; i++) {
        Eigen::VectorXd w = this->O * V.col(i); // calculate the next vector w
        alpha = (V.col(i)).dot(w);
        for (int j = 0; j < i; j++) {
            w = w - (V.col(j)).dot(w) * V.col(j); // orthogonalize the vector w with respect to the previous vectors of the Krylov basis
        }
        beta = w.norm(); // calculate the norm of the vector w
        if (beta < ref) {
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
Eigen::VectorXd Operator::FOLM_eigen(int nb_iter, Eigen::MatrixXd& eigenvectors) const {
    Eigen::MatrixXd V(D, nb_iter); // initialize the matrix V of vectors of the Krylov basis
    Eigen::MatrixXd T(nb_iter,nb_iter); // initialize the tridiagonal matrix T
    Eigen::VectorXd v_0 = Eigen::VectorXd::Random(D); // initialize a random vector v_0
    FOLM_diag(nb_iter, v_0, T, V); // tridiagonalize the hamiltonian using Lanczos algorithm
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
Eigen::VectorXd Operator::exact_eigen(Eigen::MatrixXd& eigenvectors) const {
    Eigen::MatrixXd dense_smat = Eigen::MatrixXd(this->O); // convert sparse matrix to dense matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(dense_smat); // solve the eigen problem for the hamiltonian
    if (eigensolver.info() != Eigen::Success) { // verify if the eigen search is a success
        throw std::runtime_error("Eigenvalue computation failed.");
    }
    eigenvectors = eigensolver.eigenvectors(); // eigenvectors of the hamiltonian
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues(); // eigenvalues of the hamiltonian
    return eigenvalues;
}




///// PHASE TRANSITION CHARACTERIZATION /////


// TODO: Implement the order parameter calculation
/* Calculate the order parameter of the system */
// double Operator::order_parameter(const Eigen::VectorXd& eigenvalues, const Eigen::MatrixXd& eigenvectors) const {
//     throw std::logic_error("This function has not been implemented yet.");
// }

/* Calculate the energy gap ratios of the system */
Eigen::VectorXd Operator::gap_ratios() const {
    Eigen::MatrixXcd eigenvectors;
    int nb_eigen = std::min(15, D/2);
    auto eigenvalues = this->IRLM_eigen(nb_eigen, eigenvectors);
    std::vector<double> sorted_eigenvalues(nb_eigen);
    for (int i = 0; i < nb_eigen; ++i) {
        sorted_eigenvalues[i] = std::real(eigenvalues[i]);
    }
    std::sort(sorted_eigenvalues.begin(), sorted_eigenvalues.end());
    Eigen::VectorXd gap_ratios(nb_eigen - 2);
    for (int i = 1; i < nb_eigen - 1; ++i) {
        double E_prev = sorted_eigenvalues[i - 1];
        double E_curr = sorted_eigenvalues[i];
        double E_next = sorted_eigenvalues[i + 1];
        double min_gap = std::min(E_next - E_curr, E_curr - E_prev);
        double max_gap = std::max(E_next - E_curr, E_curr - E_prev);
        gap_ratios[i - 1] = (max_gap != 0) ? (min_gap / max_gap) : 0;
    }
    return gap_ratios;
}

/* Calculate the average energy gap ratio of the system */
double Operator::average_gap_ratio() const {
    Eigen::VectorXd gap_ratios = this->gap_ratios();
    double sum = gap_ratios.sum();
    int count = gap_ratios.size();
    return count > 0 ? sum / count : 0.0;
}


    /* SPECIFIC CALCULATIONS */


//TODO : A ECRIRE
/* Add a potential mu to the operator */
void Operator::add_chemical_potential(double mu, int n) {
    *this = *this + Operator::Identity(D) * (mu * n);
}

//TODO : A ECRIRE
/* Add an interaction U to the operator */
// void Operator::add_interaction(double U, const Eigen::VectorXd& interaction_matrix) {
//     throw std::logic_error("This function has not been implemented yet.");
// }


///// THERMODYNAMICAL FUNCTIONS /////


/* Calculate the partition function Z for an ALREADY diagonalized hamiltonian */
double Operator::partition_function(const Eigen::VectorXd& eigenvalues, double temperature) const {
    const double k_B = 1.380649e-23; // Boltzmann constant
    double beta = 1 / (k_B * temperature);
    return (-beta * eigenvalues.array()).exp().sum();
}

// TODO: Implement the canonical density matrix calculation
/* Calculate the canonical density matrix for an ALREADY diagonalized hamiltonian */
// void Operator::canonical_density_matrix(const Eigen::VectorXd& eigenvalues, double temperature) const {
//     throw std::logic_error("This function has not been implemented yet.");
// }

//TODO : A OPTIMISER
/* Calculate the mean boson density of the system */
double Operator::boson_density(double dmu, int n) {
    [[maybe_unused]] Eigen::MatrixXcd eigenvectors;
    int nb_eigen = std::min(20, D/2);
    std::complex<double> E0 = this->IRLM_eigen(nb_eigen, eigenvectors)[0]; // ground state energy at mu
    *this = *this + Operator::Identity(D) * dmu* n;
    std::complex<double> E1 = this->IRLM_eigen(nb_eigen, eigenvectors)[0]; // ground state energy at mu + dmu
    *this = *this + Operator::Identity(D) * (-dmu * n);
    return -(std::real(E1) - std::real(E0)) / dmu;
}

//TODO : A OPTIMISER
/* Calculate the isothermal compressibility of the system */
double Operator::compressibility(double dmu, int n) {
    double n1 = this->boson_density(dmu, n); // density at mu
    *this = *this + Operator::Identity(D) * dmu * n;
    double n2 = this->boson_density(dmu, n); // density at mu + dmu
    *this = *this + Operator::Identity(D) * (-dmu * n);
    return std::pow(n1,-2) * (n2-n1)/dmu;
}