#pragma once

#include<vector>
#include<Eigen/Dense>
#include<Eigen/SparseCore> 
#include "Eigen/src/Core/Matrix.h"


/**
 * @brief Class representing the Bose-Hubbard Hamiltonian.
 * 
 * This class implements the Hamiltonian for the Bose-Hubbard model, which describes interacting bosons on a lattice.
 */

class BH {
private:

    friend class Analysis;

// DIMENSION OF THE HILBERT SPACE

    /* calculate the dimension of the Hilbert space for n bosons on m sites */
    static int binomial(int n, int k); // Binomial coefficient
    static int dimension(int m, int n); // Dimension of the Hilbert space

// ELEMENTARY FUNCTIONS

    /* calculate the sum of the elements of a vector between 2 index */
    static int sum(const Eigen::VectorXd& state, int index1, int index2); 

// INITIALIZE THE HILBERT SPACE BASIS

    /* calculate the next state of the Hilbert space in lexicographic order */
    static bool next_lexicographic(Eigen::VectorXd& state, int m, int n); 

    /* creates a matrix that has the vectors of the Hilbert space basis in columns */
    static Eigen::MatrixXd init_lexicographic(int m, int n); 

// SORT THE HILBERT SPACE BASIS TO FACILITATE CALCULUS

    /* calculate the unique tag of the kth column of the matrix */
    static double calculate_tag(const Eigen::MatrixXd& basis, const std::vector<int>& primes, int k);

    /* calculate and store the tags of each state of the Hilbert space basis */
    static Eigen::VectorXd calculate_tags(const Eigen::MatrixXd& basis, const std::vector<int>& primes);

    /* sort the states of the Hilbert space by ascending order compared by their tags */
    static void sort_basis(Eigen::VectorXd& tags, Eigen::MatrixXd& basis); 

    /* gives the index of the wanted tag x by the Newton method */
    static int search_tag(const Eigen::VectorXd& tags, double x);

// FILL THE HAMILTONIAN OF THE SYSTEM

    /* fill the hopping term of the Hamiltonian */
    static void fill_hopping(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const std::vector<std::vector<int>>& neighbours, const std::vector<int>& primes, Eigen::SparseMatrix<double>& hmatrix, double J);

    /* fill the interaction term of the Hamiltonian */
    static void fill_interaction(const Eigen::MatrixXd& basis, Eigen::SparseMatrix<double>& hmatrix, double U); 

    /* fill the chemical potential term of the Hamiltonian */
    static void fill_chemical(const Eigen::MatrixXd& basis, Eigen::SparseMatrix<double>& hmatrix, double mu); 

public:

// BASIS

    /**
    * @brief Set the Fock states basis of the Hilbert space with a fixed number of bosons.
    * 
    * @param m Number of sites in the lattice.
    * @param n Number of bosons in the lattice.
    * @return std::pair<Eigen::VectorXd, Eigen::MatrixXd> The tags and the basis.
    */
    static std::pair<Eigen::VectorXd, Eigen::MatrixXd> fixed_set_basis(int m, int n);

    /**
    * @brief Set the Fock states basis of the Hilbert space with a varying number of bosons.
    *
    * @param m Number of sites in the lattice.
    * @param n The maximum number of bosons in the lattice.
    * @return std::pair<Eigen::VectorXd, Eigen::MatrixXd> The tags and the basis.
    */
    static std::pair<Eigen::VectorXd, Eigen::MatrixXd> max_set_basis(int m, int n);
    
// HAMILTONIAN MATRICES

    /**
    * @brief Create the Hamiltonian matrix for the Bose-Hubbard model with a fixed number of bosons.
    * 
    * @param neighbours Vector that contains the neighbours of each site of the lattice.
    * @param m Number of sites in the lattice.
    * @param n Number of bosons in the lattice.
    * @param J Hopping parameter of the BH model.
    * @param U Interaction parameter of the BH model.
    * @param mu Chemical potential of the BH model.
    */
    static Eigen::SparseMatrix<double> fixed_bosons_hamiltonian(const std::vector<std::vector<int>>& neighbours, const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, int m, int n, double J, double U, double mu);

    /**
    * @brief Create the Hamiltonian matrix for the Bose-Hubbard model with a varying number of bosons.
    *
    * @param neighbours Vector that contains the neighbours of each site of the lattice.
    * @param m Number of sites in the lattice.
    * @param n The maximum number of bosons in the lattice.
    * @param J Hopping parameter of the BH model.
    * @param U Interaction parameter of the BH model.
    * @param mu Chemical potential of the BH model.
    * @return Eigen::SparseMatrix<double> The Hamiltonian matrix.
    */
    static Eigen::SparseMatrix<double> max_bosons_hamiltonian(const std::vector<std::vector<int>>& neighbours, int m, int n_min, int n_max, double J, double U, double mu);

    /**
    * @brief Create the mean-field Hamiltonian.
    *
    * @param psi Mean-field parameter.
    * @param p Number of bosons.
    * @param mu Chemical potential.
    * @param J Hopping parameter.
    * @param q Coordination number.
    * @param h Matrix to store the Hamiltonian.
    */
    static void h_MF(double psi, int p, double mu, double J, int q, Eigen::MatrixXd& h);

}; 

