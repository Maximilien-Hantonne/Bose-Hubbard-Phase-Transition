#include <cmath>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <getopt.h>
#include <omp.h>

#include "hamiltonian.hpp"
#include "neighbours.hpp"
#include "resource.hpp"
#include "operator.hpp"
#include "analysis.hpp"



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

    // PARAMETERS OF THE MODEL
    int m, n;
    double J, U, mu, s, r;
    double J_min, J_max, mu_min, mu_max, U_min, U_max;
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
    if (s >= r) {
        std::cerr << "Error: s must be smaller than r." << std::endl;
        return 1;
    }
    if(fixed_param != "J" && fixed_param != "U" && fixed_param != "u"){
        std::cerr << "Error: fixed parameter must be J, U or u." << std::endl;
        return 1;
    }
    Resource::timer();


	// GEOMETRY OF THE LATTICE
	Neighbours neighbours(m);
	neighbours.chain_neighbours(); // list of neighbours
	const std::vector<std::vector<int>>& nei = neighbours.getNeighbours();


    // HAMILTONIAN INITIALIZATION
    Eigen::SparseMatrix<double> jsmatrix = BH::create_combined_hamiltonian(nei, m, n, 1, 0, 0);
    Operator JH(std::move(jsmatrix));
    
    Eigen::SparseMatrix<double> Usmatrix = BH::create_combined_hamiltonian(nei, m, n, 0, 1, 0);
    Operator UH(std::move(Usmatrix));

    Eigen::SparseMatrix<double> usmatrix = BH::create_combined_hamiltonian(nei, m, n, 0, 0, 1);
    Operator uH(std::move(usmatrix));
    
    
    // SETTING THE NUMBER OF THREADS FOR PARALLELIZATION
    Resource::set_omp_threads(jsmatrix, 3);
    

    // CALCULATING AND SAVING THE PHASE TRANSITION PARAMETERS
    J_min = J; J_max = J + r; mu_min = mu; mu_max = mu + r; U_min = U; U_max = U + r;

    if (fixed_param == "J") {
        JH = JH * J;
        Analysis::calculate_and_save(fixed_param, J, U_min, U_max, mu_min, mu_max, s,s, JH, UH, uH);
    }
    else if (fixed_param == "U") {
        UH = UH * U;
        Analysis::calculate_and_save(fixed_param, U, J_min, J_max, mu_min, mu_max, s,s, UH, JH, uH);
    }
    else{
        uH = uH * mu;
        Analysis::calculate_and_save(fixed_param, mu, J_min, J_max, U_min, U_max, s,s, uH, JH, UH);
    }


    // EFFICIENCY OF THE CALCULATION
    Resource::timer();
    Resource::get_memory_usage(true);


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
