#include <set>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <getopt.h>
#include <omp.h>

#include "hamiltonian.h"
#include "operator.h"
#include "neighbours.h"


// Standardize the matrix using Eigen functions
Eigen::MatrixXd standardize_matrix(const Eigen::MatrixXd& matrix) {
    Eigen::MatrixXd standardized_matrix = matrix;
    Eigen::VectorXd mean = matrix.colwise().mean();
    Eigen::VectorXd stddev = ((matrix.rowwise() - mean.transpose()).array().square().colwise().sum() / (matrix.rows() - 1)).sqrt();
    standardized_matrix = (matrix.rowwise() - mean.transpose()).array().rowwise() / stddev.transpose().array();
    return standardized_matrix;
}

/** 
 * @brief Get the memory usage of the program in KB and optionally print it.
 * @param print If true, print the memory usage.
 * @return The memory usage in KB.
 */
long get_memory_usage(bool print = false) {
    std::ifstream statm_file("/proc/self/statm");
    long memory_usage = -1;

    if (statm_file.is_open()) {
        long size, resident, share, text, lib, data, dt;
        statm_file >> size >> resident >> share >> text >> lib >> data >> dt;
        memory_usage = resident * (sysconf(_SC_PAGESIZE) / 1024); // Convert pages to KB
    }

    if (memory_usage == -1) {
        std::cerr << "Error reading memory usage from /proc/self/statm." << std::endl;
    } else if (print) {
        std::cout << "Memory usage: " << memory_usage << " KB" << std::endl;
    }

    return memory_usage;
}


/**
 * @brief Get the available memory in KB.
 */
long get_available_memory() {
    struct sysinfo info;
    if (sysinfo(&info) != 0) {
        std::cerr << "Error getting system info." << std::endl;
        return -1;
    }
    return info.freeram / 1024;
}


/** 
 * @brief Estimate the memory usage of a sparse matrix.
 * @param matrix The sparse matrix.
 * @return The estimated memory usage in bytes.
 */
size_t estimateSparseMatrixMemoryUsage(const Eigen::SparseMatrix<double>& matrix) {
    size_t numNonZeros = matrix.nonZeros();
    size_t numCols = matrix.cols();
    size_t memoryUsage = numNonZeros * sizeof(double);
    memoryUsage += numNonZeros * sizeof(int); 
    memoryUsage += (numCols + 1) * sizeof(int);
    return memoryUsage;
}

/**
 * @brief Timer function to measure the duration of the calculation.
 */
void timer() {
    static std::chrono::high_resolution_clock::time_point start_time;
    static bool is_running = false;
    if (!is_running) {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    } else {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        is_running = false;
        if (duration.count() > 60) {
            int minutes = static_cast<int>(duration.count()) / 60;
            double seconds = duration.count() - (minutes * 60);
            std::cout << "Calculation duration: " << minutes << " minutes " << seconds << " seconds." << std::endl;
        } else {
            std::cout << "Calculation duration: " << duration.count() << " seconds." << std::endl;
        }
    }
}

/**
 * @brief Print the usage information for the program.
 */
void print_usage() {
    std::cout << "Usage: program [options]\n"
              << "Options:\n"
              << "  -m, --sites       Number of sites\n"
              << "  -n, --bosons      Number of bosons\n"
              << "  -J, --hopping     Hopping parameter\n"
              << "  -U, --interaction On-site interaction\n"
              << "  -u, --potential   Chemical potential\n"
              << "  -r, --range     Range for varying parameters\n"
              << "  -s, --step      Step for varying parameters (with s < r)\n"
              << "  -f --fixed      Fixed parameter (J, U or u) \n";
}


/**
 * @brief Main function for the Bose-Hubbard Phase Transition program.
 * 
 * This function parses command-line arguments to set up the parameters for the Bose-Hubbard model.
 * 
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * 
 * Command-line options:
 * - `-m, --sites`: Number of sites in the lattice.
 * - `-n, --bosons`: Number of bosons in the lattice.
 * - `-J, --hopping`: Hopping parameter.
 * - `-U, --interaction`: On-site interaction parameter.
 * - `-u, --potential`: Chemical potential.
 * - `-r, --range`: Range for chemical potential and interaction.
 * - `-s, --step`: Step for chemical potential and interaction.
 * - `-f, --fixed`: Fixed parameter (J, U, or u).
 * - `-h, --help`: Display usage information.
 * 
 * @return int Exit status of the program.
 * @warning s must be smaller than r.
 */

int main(int argc, char *argv[]) {

    // // INITIAL MEMORY USAGE
    // long base_memory = get_memory_usage();

    // PARAMETERS OF THE MODEL
    int m, n;
    double J, U, mu, s, r;
    [[maybe_unused]] double J_min, J_max, mu_min, mu_max, U_min, U_max;
    std::string fixed_param;

    const char* const short_opts = "m:n:J:U:u:r:s:f:h";
    const option long_opts[] = {
        {"sites", required_argument, nullptr, 'm'},
        {"bosons", required_argument, nullptr, 'n'},
        {"hopping", required_argument, nullptr, 'J'},
        {"interaction", required_argument, nullptr, 'U'},
        {"potential", required_argument, nullptr, 'u'},
        {"range", required_argument, nullptr, 'r'},
        {"step", required_argument, nullptr, 's'},
        {"fixed", required_argument, nullptr, 'f'},
		{"help", no_argument, nullptr, 'h'},
        {nullptr, no_argument, nullptr, 0}
    };

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
        if (-1 == opt) break;
        switch (opt) {
            case 'm':
                m = std::stoi(optarg);
                break;
            case 'n':
                n = std::stoi(optarg);
                break;
            case 'J':
                J = std::stod(optarg);
                break;
            case 'U':
                U = std::stod(optarg);
                break;
            case 'u':
                mu = std::stod(optarg);
                break;
            case 'r':
                r = std::stod(optarg);
                break;
            case 's':
                s = std::stod(optarg);
                break;
            case 'f':
                fixed_param = optarg;
                break;
			case 'h':
            default:
                print_usage();
                return 0;
        }
    }

    J_min = J;
    J_max = J + r;
	mu_min = mu;
    mu_max = mu + r;
    U_min = U;
    U_max = U + r;

    [[maybe_unused]] double dmu = 0.05 * std::min({J > 0 ? J : std::numeric_limits<double>::max(),
                                  U > 0 ? U : std::numeric_limits<double>::max(),
                                  mu > 0 ? mu : std::numeric_limits<double>::max(),
                                  s > 0 ? s : std::numeric_limits<double>::max()});


	// GEOMETRY OF THE LATTICE
	Neighbours neighbours(m);
	neighbours.chain_neighbours(); // list of neighbours
	const std::vector<std::vector<int>>& nei = neighbours.getNeighbours();

    // HAMILTONIAN INITIALIZATION
    timer();
    BH jmatrix(nei, m, n, 1, 0, 0);
    Eigen::SparseMatrix<double> jsmatrix = jmatrix.getHamiltonian();
    Operator JH(std::move(jsmatrix));
    
    BH Umatrix(nei, m, n, 0, 1, 0);
    Eigen::SparseMatrix<double> Usmatrix = Umatrix.getHamiltonian();
    Operator UH(std::move(Usmatrix));

    BH umatrix(nei, m, n, 0, 0, 1);
    Eigen::SparseMatrix<double> usmatrix = umatrix.getHamiltonian();
    Operator uH(std::move(usmatrix));
    
    
    // SETTING THE NUMBER OF THREADS FOR PARALLELIZATION
    long available_memory = get_available_memory();
    size_t jsmatrixMemoryUsage = estimateSparseMatrixMemoryUsage(jsmatrix);
    size_t usmatrixMemoryUsage = estimateSparseMatrixMemoryUsage(Usmatrix);
    size_t umatrixMemoryUsage = estimateSparseMatrixMemoryUsage(usmatrix);
    size_t totalMemoryUsage = jsmatrixMemoryUsage + usmatrixMemoryUsage + umatrixMemoryUsage;
    int num_threads = std::min(available_memory / totalMemoryUsage, static_cast<size_t>(omp_get_max_threads()));
    omp_set_num_threads(num_threads);
    std::cout << "Using OpenMP with " << num_threads << " threads." << std::endl;
    

    // CALCULATING AND SAVING THE PHASE TRANSITION PARAMETERS
    auto calculate_and_save = [&](auto param1_min, auto param1_max, auto param2_min, auto param2_max, auto param1_step, auto param2_step, auto& H_fixed) {
        std::ofstream file("phase.txt");
        file << fixed_param << " ";
        double fixed_value;
        if (fixed_param == "J") {
            fixed_value = J;
        } else if (fixed_param == "U") {
            fixed_value = U;
        } else if (fixed_param == "u") {
            fixed_value = mu;
        } else {
            std::cerr << "Error: Invalid fixed parameter specified.\n";
            print_usage();
            return 1;
        }
        if (fixed_value == 0) {
            std::cerr << "Error: Fixed parameter " << fixed_param << " cannot be zero.\n";
            return 1;
        }
        file << fixed_value << std::endl;
        int nb_eigen = 12;
        int num_param1 = static_cast<int>((param1_max - param1_min) / param1_step) + 1;
        int num_param2 = static_cast<int>((param2_max - param2_min) / param2_step) + 1;
        std::vector<double> param1_values(num_param1 * num_param2);
        std::vector<double> param2_values(num_param1 * num_param2);
        std::vector<double> gap_ratios_values(num_param1 * num_param2);
        std::vector<double> boson_density_values(num_param1 * num_param2);
        std::vector<double> compressibility_values(num_param1 * num_param2);
        int size = H_fixed.gap_ratios(nb_eigen).size();
        Eigen::MatrixXd matrix_ratios(num_param1 * num_param2, size);
        while (true) {
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int i = 0; i < num_param1; ++i) {
                for (int j = 0; j < num_param2; ++j) {
                    double param1 = param1_min + i * param1_step;
                    double param2 = param2_min + j * param2_step;
                    Operator H = H_fixed + UH * param1 + uH * param2;
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
            std::set<double> unique_gap_ratios(gap_ratios_values.begin(), gap_ratios_values.end());
            if (unique_gap_ratios.size() >= 5) {
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
        matrix_ratios = standardize_matrix(matrix_ratios);
        matrix_ratios = (matrix_ratios.adjoint() * matrix_ratios) / double(matrix_ratios.rows() - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(matrix_ratios);
        Eigen::VectorXd eigenvalues = eigensolver.eigenvalues().real().reverse();
        Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors().real().rowwise().reverse();
        Eigen::MatrixXd projected_data = matrix_ratios * eigenvectors.leftCols(2);
        std::ofstream projected_file("projected_data.txt");
        projected_file << projected_data << std::endl;
        projected_file.close();
        return 0;
    };
    if (fixed_param == "J") {
        JH = JH * J;
        calculate_and_save(U_min, U_max, mu_min, mu_max, s, s, JH);
    } else if (fixed_param == "U") {
        UH = UH * U;
        calculate_and_save(J_min, J_max, mu_min, mu_max, s, s, UH);
    } else if (fixed_param == "u") {
        uH = uH * mu;
        calculate_and_save(J_min, J_max, U_min, U_max, s, s, uH);
    }


    // EFFICIENCY OF THE CALCULATION
    timer();
    get_memory_usage(true);


    // PLOT OF THE PHASE TRANSITION
	int result = system("python3 plot.py");
	if (result != 0) {
		std::cerr << "Error when executing Python script." << std::endl;
		return 1;
	}

	// /// DIAGONALIZATION OF THE HAMILTONIAN
	// BH hmatrix (nei, m, n, J, U, mu);
	// Eigen::SparseMatrix<double> smatrix = hmatrix.getHamiltonian();
	// Operator H(std::move(smatrix));

	// // USING THE FOLM 
	// int k = H.size(); // Number of eigenvalues to calculate
	// Eigen::VectorXd v_0 = Eigen::VectorXd::Random(H.size()); // Random initial vector
	// Eigen::MatrixXd eigenvectors1;
	// Eigen::MatrixXd V;
	// auto start = std::chrono::high_resolution_clock::now();
	// Eigen::VectorXd eigenvalues1 = H.FOLM_eigen(k, v_0,eigenvectors1); // FOLM
	// auto end = std::chrono::high_resolution_clock::now();
	// std::chrono::duration<double> duration = end - start;
	// std::cout << "FOLM execution time: " << duration.count() << " seconds" << std::endl;
    // std::cout << "smallest eigenvalue : " << eigenvalues1.transpose()[0] << std::endl;
    // std::cout << "number of calculated eigenvalues : " << eigenvalues1.size() << std::endl << std::endl;
	
	// // USING THE IRLM 
    // Eigen::MatrixXd eigenvectors2;
	// start = std::chrono::high_resolution_clock::now();
	// Eigen::VectorXcd eigenvalues2 = H.IRLM_eigen(1, eigenvectors2); // IRLM
	// end = std::chrono::high_resolution_clock::now();
	// duration = end - start;
	// std::cout << "IRLM execution time: " << duration.count() << " seconds" << std::endl;
    // std::cout << "smallest eigenvalue : " << std::real(eigenvalues2.transpose()[0]) << std::endl;
    // std::cout << "number of calculated eigenvalues : " << eigenvalues2.size() << std::endl << std::endl;

	// // USING EXACT DIAGONALIZATION
	// Eigen::MatrixXd eigenvectors3;
	// start = std::chrono::high_resolution_clock::now();
	// Eigen::VectorXd eigenvalues3 = H.exact_eigen(eigenvectors3); // Exact diagonalization
	// end = std::chrono::high_resolution_clock::now();
	// duration = end - start;
	// std::cout << "Exact diagonalization execution time: " << duration.count() << " seconds" << std::endl;
    // std::cout << "smallest eigenvalue : " << eigenvalues3.transpose()[0] << std::endl;
    // std::cout << "number of calculated eigenvalues : " << eigenvalues3.size() << std::endl << std::endl;


	/// PHASE TRANSITION CALCULATIONS
	// double boson_density = H.boson_density(0.1, n);
	// std::cout << "boson density : " << boson_density << std::endl;

	// double compressibility = H.compressibility(0.1, n);
	// std::cout << "isothermal compressibility : " << compressibility << std::endl << std::endl;

	return 0;
}
