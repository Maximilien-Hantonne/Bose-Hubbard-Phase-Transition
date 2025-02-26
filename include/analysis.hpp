#pragma once

#include <complex>
#include <Eigen/Dense>
#include <Eigen/SparseCore>



/** 
 * @brief Class for the analysis of the phase transition.
 * 
 * This class contains methods for the analysis of the characteristic quantities in the Bose-Hubbard model.
 */

class Analysis {
private:

// MEAN-FIELD CALCULATIONS

    /* calculate the mean value of the annihilation operator <phi0|a|phi0> */
    static double SF_density(Eigen::VectorXd& phi0, int p);

    /* Self-consistent mean-field method to highlight the Superfluid-Mott insulator transition */
    static double SCMF(double mu, double J, int q ,double psi0);

// EXACT CALCULATIONS

    /* Calculate the gap ratios, spdm, boson density, and compressibility for a range of parameters */
    static void calculate_and_save(std::string fixed_param, double fixed_value, double param1_min, double param1_max, double param2_min, double param2_max, double param1_step, double param2_step, Eigen::SparseMatrix<double> H_fixed, Eigen::SparseMatrix<double> H1, Eigen::SparseMatrix<double> H2);

// GAP RATIOS

    /* Calculate the gap ratios of the system */
    static Eigen::VectorXd gap_ratios(Eigen::VectorXcd eigenvalues, int nb_eigen);

// THERMODYNAMIC FUNCTIONS

    /* Calculate the grand partition function of the system */
    static double partition_function(Eigen::VectorXcd eigenvalues, double beta);

    /* Calculate the probability of the system being in a given state in the grand canonical ensemble */
    static Eigen::VectorXd probability(Eigen::VectorXcd eigenvalues, double beta, double mu);

    /* Calculate the single-particle density matrix of the system */
    static Eigen::MatrixXcd SPDM(const Eigen::MatrixXcd& eigenvectors, int nb_eigen);

    /* Calculate the compressibility of the system */
    static double compressibility(const Eigen::MatrixXcd& spdm);
    
// MEAN VALUE CALCULATIONS

    /* Calculate the mean value of the annihilation operator <n|ai+ aj|n> */
    static std::complex<double> braket(const Eigen::MatrixXcd& eigenvectors, int i, int j);

// PCA FUNCTIONS

    /* calculate the dispersion of the projected points */
    static double calculate_dispersion(const Eigen::MatrixXd& projected_data);

    /* k-means clustering */
    static Eigen::VectorXi kmeans_clustering(const Eigen::MatrixXd& data, int num_clusters);

// UTILITY FUNCTIONS

    /* standardize a matrix */
    static Eigen::MatrixXd standardize_matrix(const Eigen::MatrixXd& matrix);

public:
    
    /** 
     * @brief Compute the phase diagram of the Bose_Hubbard model in the mean-field approximation, given parameters.
     * 
     * @param n number of different values for both the chemical potential mu and the hopping parameter J.
     * @param J Hopping parameter.
     * @param mu Chemical potential.
     * @param r Range for varying parameters.
     */
    static void mean_field_parameters(int n, double J, double mu, double r);

    /**
     * @brief Calculate the mean gap ratio, boson density, and compressibility with exact methods, given parameters.
     *
     * @param J Hopping parameter.
     * @param U Interaction parameter.
     * @param mu Chemical potential.
     * @param s Range for varying parameters.
     * @param r Step for varying parameters (with s < r).
     * @param fixed_param Fixed parameter (J, U, or mu).
     */
    static void exact_parameters(int m, int n, double J, double U, double mu, double s, double r,std::string fixed_param);
    
};