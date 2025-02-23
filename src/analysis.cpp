#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <complex>
#include <fstream>
#include <numeric>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

#include "Eigen/src/Core/Matrix.h"
#include "operator.hpp"
#include "analysis.hpp"
#include "hamiltonian.hpp"
#include "resource.hpp"
#include "neighbours.hpp"
#include "tqdm.h"





                   ////// MEAN-FIELD CALCULATIONS /////


/* Main function for the mean-field calculations */
void Analysis::mean_field_parameters(int n, double J, double mu, double r){

    double J_min = J; 
    double J_max = J + r;
    double dJ = (J_max - J_min) / n;
    double mu_min = mu;
    double mu_max = mu + r;
    double dmu = (mu_max - mu_min) / n;
    double q = 1;

    std::random_device rd; // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister generator
    std::uniform_real_distribution<> dis(0.0, 1.0); // Uniform distribution in [0, 1]

    std::ofstream file("mean_field.txt"); 

    Resource::timer(); // start the timer

    for (double mu : tqdm::range(mu_min, mu_max, dmu)) {
        for (double J : tqdm::range(J_min, J_max, dJ)) {
            double psi0 = dis(gen); 
            file << mu << " " << J << " " << SCMF(mu, J, q, psi0) << std::endl;
        }
    }
    
    Resource::timer(); // stop the timer
    Resource::get_memory_usage(true); // get the memory usage

    file.close();
}


/* calculate the mean value of the annihilation operator <phi0|a|phi0> */
double Analysis::SF_density(Eigen::VectorXd& phi0, int p)
/* Returns the mean value of the annihilation operator <phi0|a|phi0>, that is the superfluid density of the state phi0 */
{

    double a = 0; 
    for(int i=1; i<2*p+1; i++)
    {
        a += sqrt(i)*phi0[i]*phi0[i-1];
    }
    return a; 
}

/* Self-consistent mean-field method to highlight the Superfluid-Mott insulator transition
Parameters: 
- mu = mu/U (without dimension)
- J = J/U 
- q : number of neighbours of a site i in the specific lattice studied
- psi0: initial ansatz for the superfluid order parameter */
double Analysis::SCMF(double mu, double J, int q ,double psi0)
{
    double psi = psi0; // initial ansatz for the superfluid order parameter
    double psi_new = psi0; // new ansatz for the superfluid order parameter
    Eigen::VectorXd phi0; // GS eigenvector
    double e0; // ground state energy
    double e0_new; // new ground state energy
    double tmp = 0; //temporary variable
    int N_itt=1; 
    int N_itt_inner; // number of iterations
    int p; 
    double eps = 1e-6; // precision of the convergence

    Eigen::MatrixXd h(1000, 1000); // alllocation of a huge dense matrix, in which we will store the iterative
                                   // single particle hamiltonian in the mean-field approximation
    h.setZero();

    // std::cout << "*** Computing the superfluid order parameter Psi up to " << eps << " precision ***" << std::endl; 
    do
    {
        // std::cout << "*** Inner Loop: Computing the ground state e0 and phi0 up to " << eps << " precision ***" << std::endl;
        N_itt_inner = 0; 
        p = 1; 
        BH::h_MF(psi, p, mu, J, q, h); // single particle hamiltonian in the mean-field approximation
        // Define a submatrix view of the matrix h; without allocating new memory 
        Eigen::Block<Eigen::MatrixXd> sub_h = h.block(0, 0, 2*p+1, 2*p+1);
        Spectra::DenseSymMatProd<double> op(sub_h); 
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, 2*p);
        eigs.init();
        [[maybe_unused]] int nconv = eigs.compute(Spectra::SortRule::SmallestAlge);
        if (eigs.info() != Spectra::CompInfo::Successful) { // verify if the eigen search is a success
            // throw std::runtime_error("Eigenvalue computation for the mean-field single particle hamiltonian failed.");
            return -1.0;
        }
        else{
            phi0 = eigs.eigenvectors().col(0); // GS eigenvector
            e0 = eigs.eigenvalues()[0]; // GS eigenvalue 
        }
        
        do
        {
            p++; 
            BH::h_MF(psi, p, mu, J, q, h); // single particle hamiltonian in the mean-field approximation
            Eigen::Block<Eigen::MatrixXd> sub_h = h.block(0, 0, 2*p+1, 2*p+1);
            Spectra::DenseSymMatProd<double> op(sub_h); 
            Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, 2*p);
            eigs.init();
            [[maybe_unused]] int nconv = eigs.compute(Spectra::SortRule::SmallestAlge);
            if (eigs.info() != Spectra::CompInfo::Successful) { // verify if the eigen search is a success
                throw std::runtime_error("Eigenvalue computation for the mean-field single particle hamiltonian failed.");
            }
            else{
                phi0 = eigs.eigenvectors().col(0); // GS eigenvector
                e0_new = eigs.eigenvalues()[0]; // GS eigenvalue 
            }
            N_itt_inner++; 
            tmp = std::abs(e0-e0_new);
            e0 = e0_new;
        }while(tmp>eps);

        N_itt++; 
        psi_new = SF_density(phi0, p); // new ansatz for the superfluid order parameter
        tmp = std::abs(psi_new - psi);
        psi = psi_new;
    }while(tmp>eps);
    
    return psi;
}





                    ///// EXACT CALCULATIONS /////


        /* MAIN FUNCTIONS */

/*main function for exact calculations parameters*/
void Analysis::exact_parameters(int m, int n, double J,double U, double mu, double s, double r, std::string fixed_param) {
    Resource::timer();

    Neighbours neighbours(m);
    neighbours.chain_neighbours();
    const std::vector<std::vector<int>>& nei = neighbours.getNeighbours();
    
    auto [tags, basis] = BH::set_basis(m, n);

    if (std::abs(J-0.0) < std::numeric_limits<double>::epsilon() && std::abs(U-0.0) < std::numeric_limits<double>::epsilon() && std::abs(mu-0.0) < std::numeric_limits<double>::epsilon()) {
        std::cerr << "Error: At least one of the parameters J, U, mu must be different from zero." << std::endl;
        return;
    }

    int n_min = 1, n_max = n;

    Eigen::SparseMatrix<double> JH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 1, 0, 0);
    Eigen::SparseMatrix<double> UH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 0, 1, 0);
    Eigen::SparseMatrix<double> uH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 0, 0, 1);

    // Eigen::SparseMatrix<double> JH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 1, 0, 0);
    // Eigen::SparseMatrix<double> UH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 0, 1, 0);
    // Eigen::SparseMatrix<double> uH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 0, 0, 1);

    Resource::set_omp_threads(JH, 3);

    double J_min = J, J_max = J + r, mu_min = mu, mu_max = mu + r, U_min = U, U_max = U + r;

    if (fixed_param == "J") {
        JH = JH * J;
        calculate_and_save(basis, tags, JH, UH, uH, fixed_param, J, J_min, J_max, mu_min, mu_max, s, s);
    }
    else if (fixed_param == "U") {
        UH = UH * U;
        calculate_and_save(basis, tags, UH, JH, uH, fixed_param, U, J_min, J_max, U_min, U_max, s, s);
    }
    else{
        uH = uH * mu;
        calculate_and_save(basis, tags, uH, JH, UH, fixed_param, mu, J_min, J_max, mu_min, mu_max, s, s);
    }
    Resource::timer();
    Resource::get_memory_usage(true);
}


/* calculate and save gap ratio and other quantities */
void Analysis::calculate_and_save(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::SparseMatrix<double> H_fixed, const Eigen::SparseMatrix<double> H1, const Eigen::SparseMatrix<double> H2, std::string fixed_param, double fixed_value, double param1_min, double param1_max, double param2_min, double param2_max, double param1_step, double param2_step) {
    std::ofstream file("phase.txt");
    file << fixed_param << " ";
    if (fixed_value == 0) {
        std::cerr << "Error: Fixed parameter " << fixed_value << " cannot be zero.\n";
    }
    file << fixed_value << std::endl;

    // PARAMETERS

    int nb_eigen = 10;
    double temperature = 0.0;

    int num_param1 = static_cast<int>((param1_max - param1_min) / param1_step) + 1;
    int num_param2 = static_cast<int>((param2_max - param2_min) / param2_step) + 1;
    std::vector<double> param1_values(num_param1 * num_param2);
    std::vector<double> param2_values(num_param1 * num_param2);
    std::vector<double> gap_ratios_values(num_param1 * num_param2);
    std::vector<double> boson_density_values(num_param1 * num_param2);
    std::vector<double> compressibility_values(num_param1 * num_param2);
    Eigen::MatrixXcd eigenvectors;
    Eigen::MatrixXd matrix_ratios(num_param1 * num_param2, nb_eigen -2);
    std::vector<Eigen::MatrixXd> spdm_matrices(num_param1 * num_param2);
    double variance_threshold_percent = 1e-8;

    while (true) {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < num_param1; ++i) {
            for (int j = 0; j < num_param2; ++j) {
                double param1 = param1_min + i * param1_step;
                double param2 = param2_min + j * param2_step;
                Eigen::SparseMatrix<double> H = H_fixed + H1 * param1 + H2 * param2;
                Eigen::VectorXcd eigenvalues = Operator::IRLM_eigen(H, nb_eigen, eigenvectors);
                Eigen::VectorXd vec_ratios = gap_ratios(eigenvalues, nb_eigen);
                double gap_ratio = vec_ratios.size() > 0 ? vec_ratios.sum() / vec_ratios.size() : 0.0;
                Eigen::MatrixXcd spdm;
                double density, K;
                if (temperature > 1e-3) {
                    spdm = density_matrix(eigenvalues, eigenvectors, temperature);
                }
                else {
                    spdm = SPDM(basis, tags, eigenvectors);
                }
                density = std::real(spdm.trace());
                K = fluctuations(spdm);

                normalize_spdm(spdm);

                int index = i * num_param2 + j;
                
                #pragma omp critical
                {
                    param1_values[index] = param1;
                    param2_values[index] = param2;
                    gap_ratios_values[index] = gap_ratio;
                    boson_density_values[index] = density;
                    compressibility_values[index] = K;
                    matrix_ratios.row(index) = vec_ratios;
                    spdm_matrices[index] = spdm.real();
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
        matrix_ratios.resize(num_param1 * num_param2, nb_eigen - 2);  
    }

    for (int i = 0; i < num_param1 * num_param2; ++i) {
        file << param1_values[i] << " " << param2_values[i] << " " << gap_ratios_values[i] << " " << boson_density_values[i] << " " << compressibility_values[i] << std::endl;
    }

    file.close();

    // PCA, dispersion, and clustering

    std::vector<Eigen::MatrixXd> pca_matrices;
    std::vector<double> dispersions;
    std::vector<Eigen::VectorXi> cluster_labels;

    for (int start_row = 0; start_row < matrix_ratios.rows()-6; start_row += 6) {
        int end_row = std::min(start_row + 6, static_cast<int>(matrix_ratios.rows()));
        Eigen::MatrixXd sub_matrix = matrix_ratios.block(start_row, 0, end_row - start_row, matrix_ratios.cols());

        sub_matrix = standardize_matrix(sub_matrix);
        sub_matrix = (sub_matrix.adjoint() * sub_matrix) / double(sub_matrix.rows() - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(sub_matrix);
        Eigen::MatrixXd eigenvectors2 = eigensolver.eigenvectors().real().rowwise().reverse();
        Eigen::MatrixXd projected_data = sub_matrix * eigenvectors2.leftCols(2);
        pca_matrices.push_back(projected_data);

        double dispersion = calculate_dispersion(projected_data);
        dispersions.push_back(dispersion);

        int num_clusters = 2;
        Eigen::MatrixXd projected_data_copy = projected_data;
        Eigen::VectorXi labels = kmeans_clustering(projected_data_copy, num_clusters);
        cluster_labels.push_back(labels);
    }

    save_matrices_to_csv("spdm_matrices.csv", spdm_matrices, "Matrix");
    save_matrices_to_csv("pca_results.csv", pca_matrices, "PCA");
    save_dispersions("dispersions.csv", dispersions);
    save_cluster_labels("cluster_labels.csv", cluster_labels);
}


        /* GAP RATIOS */

/* Calculate the energy gap ratios of the system */
Eigen::VectorXd Analysis::gap_ratios(Eigen::VectorXcd eigenvalues,int nb_eigen) {
    Eigen::VectorXd gap_ratios(nb_eigen - 2);
    std::vector<double> sorted_eigenvalues(nb_eigen);
    for (int i = 0; i < nb_eigen; ++i) {
        sorted_eigenvalues[i] = std::real(eigenvalues[i]);
    }
    std::sort(sorted_eigenvalues.begin(), sorted_eigenvalues.end());
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


        /* THERMODYNAMIC FUNCTIONS */

/* Calculate the partition function of the system */
double Analysis::partition_function(Eigen::VectorXcd eigenvalues, double beta) {
    double Z = 0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        Z += exp(-beta * (std::real(eigenvalues[i])));
    }
    return Z;
}

/* Calculate the probability of the system being in a given state in the canonical ensemble */
double Analysis::partition_proba(double energy, double beta) {
    return exp(- beta * energy);
}

/* Calculate the density matrix of the system */
Eigen::MatrixXcd Analysis::density_matrix(const Eigen::VectorXcd eigenvalues, Eigen::MatrixXcd& eigenvectors, double temperature) {
    int nb_eigen = eigenvalues.size();
    Eigen::VectorXd normalized_eigenvalues = eigenvalues.real()/eigenvalues.real().maxCoeff();
    Eigen::MatrixXcd density_matrix = Eigen::MatrixXcd::Zero(nb_eigen, nb_eigen);
    double Z = partition_function(normalized_eigenvalues, 1.0 / temperature);
    for (int i = 0; i < nb_eigen; ++i) {
        density_matrix(i, i) = partition_proba(std::real(normalized_eigenvalues[i]), 1.0 / temperature) / Z;
    }
    Eigen::MatrixXcd t_eigenvectors = eigenvectors.transpose();
    density_matrix = eigenvectors * density_matrix * t_eigenvectors;
    return density_matrix;
}

/* Calculate the single-particle density matrix of the system */
Eigen::MatrixXcd Analysis::SPDM(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, Eigen::MatrixXcd& eigenvectors) {
    int m = basis.rows();
    Eigen::MatrixXcd spdm = Eigen::MatrixXcd::Zero(m, m);
    Eigen::VectorXd phi0 = eigenvectors.col(0).real();
    // for (int i = 0; i < m; ++i) {
    //     for (int j = i; j < m; ++j) {
    //         spdm(i, j) = braket(phi0, basis, tags, i, j);
    //     }
    //     for (int j = 0; j < i; ++j) {
    //         spdm(i, j) = std::conj(spdm(j, i));
    //     }
    // }
    for (int k = 0; k < eigenvectors.cols(); ++k) {
        Eigen::VectorXd phi = eigenvectors.col(k).real();
        for (int i = 0; i < m; ++i) {
            for (int j = i; j < m; ++j) {
                spdm(i, j) += braket(phi, basis, tags, i, j);
            }
            for (int j = 0; j < i; ++j) {
                spdm(i, j) += std::conj(spdm(j, i));
            }
        }
    }
    return spdm/ eigenvectors.cols();
}

/* Normalize the spdm to the distance between each site */
void Analysis::normalize_spdm(Eigen::MatrixXcd& spdm) {
    for (int i = 0; i < spdm.rows(); ++i) {
        for (int j = 0; j < spdm.cols(); ++j) {
            spdm(i, j) /= (std::abs(i - j) + 1);
        }
    }
}


        /* COMPRESSIBILITY */
/* Calculate the compressibility of the system */
double Analysis::fluctuations(const Eigen::MatrixXcd& spdm) {
    double K = 0;
    for (int i = 0; i < spdm.rows(); ++i) {
        for (int j = 0; j < spdm.cols(); ++j) {
            K += std::real(spdm(i, j) * spdm(i, j));
        };
    }
    // std::cout << "K = " << K << " - trace = " << spdm.trace().real() << std::endl;
    return (K-std::pow(spdm.trace().real(),2))/(spdm.trace().real());
}


        /* MEAN VALUE CALCULATIONS */

/* Calculate the mean value of the annihilation operator <n|ai+ aj|n> */
std::complex<double> Analysis::braket(const Eigen::VectorXcd& phi0, const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, int i, int j) {
    std::complex<double> braket_value = 0 ;
    std::vector<int> primes = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 };
    for (int k = 0; k < basis.cols(); ++k) {
        if (std::abs(phi0[k]) > std::numeric_limits<double>::epsilon()) {
            Eigen::VectorXd state = basis.col(k);
            if (state[i] >= 0 && state[j] >= 1) {
                state[i] += 1;
                state[j] -= 1;
                double x = BH::calculate_tag(state, primes, i);
                int index = BH::search_tag(tags, x);
                if (std::abs(phi0[index]) > 1e-6) {
                    braket_value += std::conj(phi0[k]) * phi0[index] * sqrt(state[i] * state[j]);
                }
            }
        }
    }
    return braket_value;
}


                    ///// UTILITY FUNCTIONS /////


        /* PCA FUNCTIONS */

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

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_points - 1);
    for (int i = 0; i < num_clusters; ++i) {
        centroids.row(i) = data.row(dis(gen));
    }

    bool converged = false;
    while (!converged) {
        converged = true;

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


        /* STANDARDIZATION */

/* standardize a matrix */
Eigen::MatrixXd Analysis::standardize_matrix(const Eigen::MatrixXd& matrix) {
    Eigen::MatrixXd standardized_matrix = matrix;
    Eigen::VectorXd mean = matrix.colwise().mean();
    Eigen::VectorXd stddev = ((matrix.rowwise() - mean.transpose()).array().square().colwise().sum() / (matrix.rows() - 1)).sqrt();
    standardized_matrix = (matrix.rowwise() - mean.transpose()).array().rowwise() / stddev.transpose().array();
    return standardized_matrix;
}


        /* SAVE TO CSV */

/* Save the real part of matrices to a CSV file */
void Analysis::save_matrices_to_csv(const std::string& filename, const std::vector<Eigen::MatrixXd>& matrices, const std::string& label) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t k = 0; k < matrices.size(); ++k) {
            file << label << " " << k << std::endl;
            const Eigen::MatrixXd& matrix = matrices[k];
            for (int i = 0; i < matrix.rows(); ++i) {
                for (int j = 0; j < matrix.cols(); ++j) {
                    file << matrix(i, j);
                    if (j < matrix.cols() - 1) {
                        file << ",";
                    }
                }
                file << std::endl;
            }
            file << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

void Analysis::save_dispersions(const std::string& filename, const std::vector<double>& dispersions) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t k = 0; k < dispersions.size(); ++k) {
            file << "Dispersion " << k << std::endl;
            file << dispersions[k] << std::endl;
            file << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

void Analysis::save_cluster_labels(const std::string& filename, const std::vector<Eigen::VectorXi>& cluster_labels) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t k = 0; k < cluster_labels.size(); ++k) {
            file << "Clusters " << k << std::endl;
            const Eigen::VectorXi& labels = cluster_labels[k];
            for (int i = 0; i < labels.size(); ++i) {
                file << labels[i] << std::endl;
            }
            file << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}