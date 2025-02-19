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
protected:

// PARAMETERS OF THE BH MODEL

    std::vector<std::vector<int>> neighbours; // Vector that contains the neighbours of each site of the chain
    
    int m; // Number of sites in the chain
    int n; // Number of bosons in the chain
    int D; // Dimension of the Hilbert space of the system

    double J; // Hopping parameter of the BH model
    double U; // Interaction parameter of the BH model 
    double mu; // Chemical potential of the BH model

    Eigen::SparseMatrix<double> H; // Sparse matrix representation of the Hamiltonian

// DIMENSION OF THE HILBERT SPACE

    /* calculate the dimension of the Hilbert space for n bosons on m sites */
    int binomial(int n, int k) const; // Binomial coefficient
    int dimension(int m, int n) const; // Dimension of the Hilbert space

// ELEMENTARY FUNCTIONS

    /* calculate the sum of the elements of a vector between 2 index */
    int sum(const Eigen::VectorXd& state, int index1, int index2) const; 

// INITIALIZE THE HILBERT SPACE BASIS

    /* calculate the next state of the Hilbert space in lexicographic order */
    bool next_lexicographic(Eigen::VectorXd& state, int m, int n) const; 

    /* creates a matrix that has the vectors of the Hilbert space basis in columns */
    Eigen::MatrixXd init_lexicographic(int m, int n) const; 

// SORT THE HILBERT SPACE BASIS TO FACILITATE CALCULUS

    /* calculate the unique tag of the kth column of the matrix */
    double calculate_tag(const Eigen::MatrixXd& basis, const std::vector<int>& primes, int k) const;

    /* calculate and store the tags of each state of the Hilbert space basis */
    Eigen::VectorXd calculate_tags(const Eigen::MatrixXd& basis, const std::vector<int>& primes) const;

    /* sort the states of the Hilbert space by ascending order compared by their tags */
    void sort_basis(Eigen::VectorXd& tags, Eigen::MatrixXd& basis) const; 

    /* gives the index of the wanted tag x by the Newton method */
    int search_tag(const Eigen::VectorXd& tags, double x) const;

// FILL THE HAMILTONIAN OF THE SYSTEM

    /* fill the hopping term of the Hamiltonian */
    void fill_hopping(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const std::vector<std::vector<int>>& neighbours, const std::vector<int>& primes, Eigen::SparseMatrix<double>& hmatrix, double J) const;

    /* fill the interaction term of the Hamiltonian */
    void fill_interaction(const Eigen::MatrixXd& basis, Eigen::SparseMatrix<double>& hmatrix, double U) const; 

    /* fill the chemical potential term of the Hamiltonian */
    void fill_chemical(const Eigen::MatrixXd& basis, Eigen::SparseMatrix<double>& hmatrix, double mu) const; 

public:
    
    Eigen::VectorXd interaction_matrix; // Vector that contains the interaction matrix elements

// CONSTRUCTOR

    /**
    * @brief Constructor for the Bose-Hubbard Hamiltonian.
    * 
    * @param neighbours Vector that contains the neighbours of each site of the lattice.
    * @param m Number of sites in the lattice.
    * @param n Number of bosons in the lattice.
    * @param J Hopping parameter of the BH model.
    * @param U Interaction parameter of the BH model.
    * @param mu Chemical potential of the BH model.
    */
    BH(const std::vector<std::vector<int>>& neighbours, int m_, int n_, double J_, double U_, double mu_);

    /**
    * @brief Create the Hamiltonian matrix for the Bose-Hubbard model.
    *
    * @param neighbours Vector that contains the neighbours of each site of the lattice.
    * @param m Number of sites in the lattice.
    * @param n The maximum number of bosons in the lattice.
    * @param J Hopping parameter of the BH model.
    * @param U Interaction parameter of the BH model.
    * @param mu Chemical potential of the BH model.
    * @return Eigen::SparseMatrix<double> The Hamiltonian matrix.
    */
    static Eigen::SparseMatrix<double> create_combined_hamiltonian(const std::vector<std::vector<int>>& neighbours, int m, int n, double J, double U, double mu);

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

// UTILITY FUNCTIONS

    /**
    * @brief Return the Hamiltonian sparse matrix.
    * 
    * @return Eigen::SparseMatrix<double> The Hamiltonian sparse matrix.
    */
    Eigen::SparseMatrix<double> getHamiltonian() const; 
}; 

