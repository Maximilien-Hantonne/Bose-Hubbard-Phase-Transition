#include <cmath>
#include <future>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <getopt.h>
#include <omp.h>

#include "analysis.hpp"

/**
 * @file main.cpp
 * @brief Main file for the Bose-Hubbard Phase Transition program.
 *
 * This file contains the main function and the command-line argument parsing
 * for the Bose-Hubbard Phase Transition program. The program calculates the
 * phase transition parameters for the Bose-Hubbard model using either exact
 * diagonalization or mean-field theory.
 *
 * The program accepts various command-line options to set up the parameters
 * for the model and to specify the type of calculation to be performed.
 *
 * The results of the calculations are plotted using Python scripts.
 *



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
              << "  -r, --range     Range for varying parameters (if range is the same for each)\n"
              << "  -s, --step      Step for varying parameters (with s < r)\n"
              << "  -f --fixed      Fixed parameter (J, U or u) \n"
              << "  -t, --type      Type of calculation (exact or mean)\n"
              << "  -i, --iterations  Number of iterations over the parameters in the mean-field approximation\n"
              << "  -e, --epsilon  Threshold for convergence in the mean-field approximation\n";
}

/**
 * @brief Print the copyright and warranty information.
 */
 void copyright_warranty() {
    std::cout << "======================================\n";
    std::cout << "Bose-Hubbard Phase Transition\n";
    std::cout << "Copyright (C) 2025 by Maximilien HANTONNE, Rémy LYSCAR and Alexandre MENARD\n";
    std::cout << "This program is licensed under the GNU General Public License v3.0.\n";
    std::cout << "For more details, see the LICENSE file.\n";
    std::cout << "======================================\n\n\n";
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
 * - `-r, --range`: Range for varying parameters
 * - `-s, --step`: Step for chemical potential and interaction.
 * - `-f, --fixed`: Fixed parameter (J, U, or u).
 * - `-t, --type`: Type of calculation (exact or mean).
 * - `-h, --help`: Display usage information.
 * - `-i, --iterations`: Number of iterations over parameters in the mean-field approximation.
 * - `-e, --epsilon`: Threshold for convergence in the mean-field approximation
 * 
 * @return int Exit status of the program.
 * @warning s must be smaller than r.
 */

int main(int argc, char *argv[]) {

    // Print the copyright and warranty information
    copyright_warranty();

    // PARAMETERS OF THE MODEL
    int m, n, i, eps;
    double J, U, mu, s, r;
    std::string fixed_param, calc_type;

    const char* const short_opts = "m:n:J:U:u:r:s:f:t:i:e:h";
    const option long_opts[] = {
        {"sites", required_argument, nullptr, 'm'},
        {"bosons", required_argument, nullptr, 'n'},
        {"hopping", required_argument, nullptr, 'J'},
        {"interaction", required_argument, nullptr, 'U'},
        {"potential", required_argument, nullptr, 'u'},
        {"range", required_argument, nullptr, 'r'},
        {"step", required_argument, nullptr, 's'},
        {"fixed", required_argument, nullptr, 'f'},
        {"type", required_argument, nullptr, 't'},
        {"iterations", required_argument, nullptr, 'i'},
        {"epsilon", required_argument, nullptr, 'e'},
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
            case 't':
                calc_type = optarg;
                break;
            case 'i':
                i = std::stod(optarg);
                break;
            case 'e': 
                eps = std::stod(optarg); 
                break;
            case 'h':
            default:
                print_usage();
                return 0;
        }
    }

    if(calc_type != "exact" && calc_type != "mean"){
        std::cerr << "Error: calculation type must be 'exact' or 'mean'." << std::endl;
        return 1;
    }

    if (calc_type == "exact") {
        if (s >= r) {
            std::cerr << "Error: s must be smaller than r." << std::endl;
            return 1;
        }
        if(fixed_param != "J" && fixed_param != "U" && fixed_param != "u"){
            std::cerr << "Error: fixed parameter must be J, U or u." << std::endl;
            return 1;
        }

        // Calculate the exact parameters
        Analysis::exact_parameters(m, n, J, U, mu, s, r, fixed_param);

        // Execute the Python script to plot the results
        auto run_python_script = []() -> int {
            return system("python3 plot.py");
        };
        std::future<int> result = std::async(std::launch::async, run_python_script);
        if (result.get() != 0) {
            std::cerr << "Error when executing Python script." << std::endl;
            return 1;
        }
    }
    else if (calc_type == "mean"){

        // Calculate the mean-field parameters
        Analysis::mean_field_parameters(i, eps);
        
        // Execute the Python script to plot the results
        auto run_python_script = []() -> int {
            return system("python3 plot_mean_field.py");
        };
        std::future<int> result = std::async(std::launch::async, run_python_script);
        if (result.get() != 0) {
            std::cerr << "Error when executing Python script." << std::endl;
            return 1;
        }
    }
    return 0;
}