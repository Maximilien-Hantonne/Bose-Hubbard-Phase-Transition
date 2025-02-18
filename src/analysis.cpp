#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <fstream>
#include <numeric>
#include <iostream>

#include "operator.hpp"
#include "analysis.hpp"



/* calculate and save the phase transition parameters */
void Analysis::calculate_and_save(std::string fixed_param, double fixed_value, double param1_min, double param1_max, double param2_min, double param2_max, double param1_step, double param2_step, Operator& H_fixed, Operator& H1, Operator& H2) {
    std::ofstream file("phase.txt");
        file << fixed_param << " ";
        if (fixed_value == 0) {
            std::cerr << "Error: Fixed parameter " << fixed_value << " cannot be zero.\n";
        }
    file << fixed_value << std::endl;
    int nb_eigen = 15;
    int num_param1 = static_cast<int>((param1_max - param1_min) / param1_step) + 1;
    int num_param2 = static_cast<int>((param2_max - param2_min) / param2_step) + 1;
    std::vector<double> param1_values(num_param1 * num_param2);
    std::vector<double> param2_values(num_param1 * num_param2);
    std::vector<double> gap_ratios_values(num_param1 * num_param2);
    std::vector<double> boson_density_values(num_param1 * num_param2);
    std::vector<double> compressibility_values(num_param1 * num_param2);
    int size = H_fixed.gap_ratios(nb_eigen).size();
    Eigen::MatrixXd matrix_ratios(num_param1 * num_param2, size);
    double variance_threshold_percent = 1e-9;

    while (true) {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < num_param1; ++i) {
            for (int j = 0; j < num_param2; ++j) {
                double param1 = param1_min + i * param1_step;
                double param2 = param2_min + j * param2_step;
                Operator H = H_fixed + H1 * param1 + H2 * param2;
                Eigen::VectorXd vec_ratios = H.gap_ratios(nb_eigen);
                double gap_ratio = vec_ratios.size() > 0 ? vec_ratios.sum() / vec_ratios.size() : 0.0;
                double boson_density = 0;
                double compressibility = 0;
                int index = i * num_param2 + j;
                
                #pragma omp critical
                {
                    param1_values[index] = param1;
                    param2_values[index] = param2;
                    gap_ratios_values[index] = gap_ratio;
                    boson_density_values[index] = boson_density;
                    compressibility_values[index] = compressibility;
                    matrix_ratios.row(i * num_param2 + j) = vec_ratios;
                }
            }
        }

        double mean_gap_ratio = std::accumulate(gap_ratios_values.begin(), gap_ratios_values.end(), 0.0) / gap_ratios_values.size();
        double variance_gap_ratio = 0.0;
        for (const auto& value : gap_ratios_values) {
            variance_gap_ratio += (value - mean_gap_ratio) * (value - mean_gap_ratio);
        }
        variance_gap_ratio /= gap_ratios_values.size();
        if (variance_gap_ratio > variance_threshold_percent * mean_gap_ratio) {
            break;
        }

        nb_eigen += 5;
        size = size + 5;
        matrix_ratios.resize(num_param1 * num_param2, size);  
    }

    for (int i = 0; i < num_param1 * num_param2; ++i) {
        file << param1_values[i] << " " << param2_values[i] << " " << gap_ratios_values[i] << " " << boson_density_values[i] << " " << compressibility_values[i] << std::endl;
    }

    file.close();

    // PCA
    matrix_ratios = standardize_matrix(matrix_ratios);
    matrix_ratios = (matrix_ratios.adjoint() * matrix_ratios) / double(matrix_ratios.rows() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(matrix_ratios);
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues().real().reverse();
    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors().real().rowwise().reverse();
    Eigen::MatrixXd projected_data = matrix_ratios * eigenvectors.leftCols(2);
    std::ofstream projected_file("projected_data.txt");
    projected_file << projected_data << std::endl;
    projected_file.close();

    // Dispersion
    std::ofstream dispersion_file("dispersion.txt");
    double dispersion = calculate_dispersion(projected_data);
    dispersion_file << "Dispersion: " << dispersion << std::endl;
    dispersion_file.close();

    // Clustering
    int num_clusters = 2;
    Eigen::MatrixXd projected_data_copy = projected_data;
    Eigen::VectorXi labels = kmeans_clustering(projected_data_copy, num_clusters);
    std::ofstream clusters_file("clusters.txt");
    for (int i = 0; i < labels.size(); ++i) {
        clusters_file << labels[i] << std::endl;
    }
    clusters_file.close();
    }


/* calculate the dispersion of the projected points */
double Analysis::calculate_dispersion(const Eigen::MatrixXd& projected_data) {
    Eigen::VectorXd mean = projected_data.colwise().mean();
    Eigen::MatrixXd centered = projected_data.rowwise() - mean.transpose();
    Eigen::VectorXd variances = (centered.array().square().colwise().sum() / (centered.rows() - 1)).matrix();
    double dispersion = variances.sum();
    return dispersion;
}


/* k-means clustering */
Eigen::VectorXi Analysis::kmeans_clustering(const Eigen::MatrixXd& data, int num_clusters) {
    int num_points = data.rows();
    int num_dimensions = data.cols();
    Eigen::MatrixXd centroids = Eigen::MatrixXd::Zero(num_clusters, num_dimensions);
    Eigen::VectorXi labels = Eigen::VectorXi::Zero(num_points);
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(num_clusters);

    // Initialize centroids randomly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_points - 1);
    for (int i = 0; i < num_clusters; ++i) {
        centroids.row(i) = data.row(dis(gen));
    }

    bool converged = false;
    while (!converged) {
        converged = true;

        // Assign points to the nearest centroid
        for (int i = 0; i < num_points; ++i) {
            double min_distance = std::numeric_limits<double>::max();
            int best_cluster = 0;
            for (int j = 0; j < num_clusters; ++j) {
                double distance = (data.row(i) - centroids.row(j)).squaredNorm();
                if (distance < min_distance) {
                    min_distance = distance;
                    best_cluster = j;
                }
            }
            if (labels[i] != best_cluster) {
                labels[i] = best_cluster;
                converged = false;
            }
        }

        // Update centroids
        centroids.setZero();
        counts.setZero();
        for (int i = 0; i < num_points; ++i) {
            centroids.row(labels[i]) += data.row(i);
            counts[labels[i]] += 1;
        }
        for (int j = 0; j < num_clusters; ++j) {
            if (counts[j] > 0) {
                centroids.row(j) /= counts[j];
            }
        }
    }

    return labels;
}


/* standardize a matrix */
Eigen::MatrixXd Analysis::standardize_matrix(const Eigen::MatrixXd& matrix) {
    Eigen::MatrixXd standardized_matrix = matrix;
    Eigen::VectorXd mean = matrix.colwise().mean();
    Eigen::VectorXd stddev = ((matrix.rowwise() - mean.transpose()).array().square().colwise().sum() / (matrix.rows() - 1)).sqrt();
    standardized_matrix = (matrix.rowwise() - mean.transpose()).array().rowwise() / stddev.transpose().array();
    return standardized_matrix;
}