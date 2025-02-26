#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <omp.h>

#include "resource.hpp"



// MEMORY USAGE :

/* Get the memory usage of the program in KB and optionally print it */
long Resource::get_memory_usage(bool print) {
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

/* Get the available memory in KB */
long Resource::get_available_memory() {
    struct sysinfo info;
    if (sysinfo(&info) != 0) {
        std::cerr << "Error getting system info." << std::endl;
        return -1;
    }
    return info.freeram / 1024;
}

/* Estimate the memory usage of a sparse matrix */
size_t Resource::estimateSparseMatrixMemoryUsage(const Eigen::SparseMatrix<double>& matrix) {
    size_t numNonZeros = matrix.nonZeros();
    size_t numCols = matrix.cols();
    size_t memoryUsage = numNonZeros * sizeof(double);
    memoryUsage += numNonZeros * sizeof(int); 
    memoryUsage += (numCols + 1) * sizeof(int);
    return memoryUsage;
}



// TIMER :
/* Timer function to measure the duration of the calculation */
void Resource::timer() {
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



// OPENMP THREADS :
/* Set the number of threads for OpenMP parallelization */
void Resource::set_omp_threads(const Eigen::SparseMatrix<double>& matrix, int nb_matrix) {
    size_t memoryUsage1 = estimateSparseMatrixMemoryUsage(matrix);
    size_t totalMemoryUsage = memoryUsage1 * nb_matrix;
    long available_memory = get_available_memory();
    int num_threads = std::min(available_memory / totalMemoryUsage, static_cast<size_t>(omp_get_max_threads()));
    omp_set_num_threads(num_threads);
}