#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "operator.hpp"

/** 
 * @brief Class for the analysis of the phase transition.
 * 
 * This class contains methods for the analysis of the characteristic quantities in the Bose-Hubbard model.
 */

class Analysis {
private:

    static Eigen::MatrixXd standardize_matrix(const Eigen::MatrixXd& matrix);
    static double calculate_dispersion(const Eigen::MatrixXd& projected_data);
    static Eigen::VectorXi kmeans_clustering(const Eigen::MatrixXd& data, int num_clusters);
    static double SF_density(Eigen::VectorXd& phi0, int p);
    static double SCMF(double mu, double J, int q ,double psi0);
    static void calculate_and_save(std::string fixed_param, double fixed_value, double param1_min, double param1_max, double param2_min, double param2_max, double param1_step, double param2_step, Operator& H_fixed, Operator& H1, Operator& H2);

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