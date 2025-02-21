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
    static void calculate_and_save(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::SparseMatrix<double> H_fixed,const Eigen::SparseMatrix<double> H1, const Eigen::SparseMatrix<double> H2, std::string fixed_param, double fixed_value, double param1_min, double param1_max, double param2_min, double param2_max, double param1_step, double param2_step);

// GAP RATIOS

    /* Calculate the gap ratios of the system */
    static Eigen::VectorXd gap_ratios(Eigen::VectorXcd eigenvalues, int nb_eigen);

// THERMODYNAMIC FUNCTIONS

    // Grand canonical ensemble

    /* Calculate the partition function of the system */
    static double partition_function(Eigen::VectorXcd eigenvalues, double beta);

    /* Calculate the probability of the system being in a given state in the canonical ensemble */
    static double partition_proba(double energy, double beta);

    // Thermodynamic functions

    /* Calculate the single-particle density matrix of the system */
    static Eigen::MatrixXcd SPDM(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::MatrixXcd& eigenvectors);

    /* Calculate the density matrix of the system */
    static Eigen::MatrixXcd density_matrix(const Eigen::MatrixXcd& eigenvectors, const Eigen::VectorXcd eigenvalues, double temperature);

    /* Calculate the compressibility of the system */
    static double compressibility(const Eigen::MatrixXcd& spdm);
    
// MEAN VALUE CALCULATIONS

    /* Calculate the mean value of the annihilation operator <n|ai+ aj|n> */
    static std::complex<double> braket(const Eigen::VectorXcd& phi0, const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, int i, int j);

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
     * @brief Calculate the mean-field parameters with the self-consistent mean-field method.
     * 
     * @param n Number of bosons.
     * @param J Hopping parameter.
     * @param mu Chemical potential.
     * @param r Range for varying parameters.
     */
    static void mean_field_parameters(int n, double J, double mu, double r);

    /**
     * @brief Calculate the mean gap ratio, boson density, and compressibility with exact methods
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