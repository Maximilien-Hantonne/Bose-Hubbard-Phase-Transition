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

public:

    /** 
     * @brief Calculate the phase transition parameters and save them to a file.
     * 
     * @param fixed_param Fixed parameter (J, U, or u).
     * @param fixed_value Value of the fixed parameter.
     * @param param1_min Minimum value of the first parameter.
     * @param param1_max Maximum value of the first parameter.
     * @param param2_min Minimum value of the second parameter.
     * @param param2_max Maximum value of the second parameter.
     * @param param1_step Step for the first parameter.
     * @param param2_step Step for the second parameter.
     * @param H_fixed Fixed Hamiltonian.
     * @param H1 Hamiltonian with the first parameter.
     * @param H2 Hamiltonian with the second parameter.
     */
    static void calculate_and_save(std::string fixed_param, double fixed_value, double param1_min, double param1_max, double param2_min, double param2_max, double param1_step, double param2_step, Operator& H_fixed, Operator& H1, Operator& H2);

};