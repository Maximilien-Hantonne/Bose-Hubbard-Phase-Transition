#pragma once

#include <chrono>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <Eigen/SparseCore>


/**
 * @brief Class for collecting statistics about the program.
 */
 
class Resource {
public:

    /**
     * @brief Get the memory usage of the program in KB and optionally print it.
     * @param print If true, print the memory usage.
     * @return The memory usage in KB.
     */
    static long get_memory_usage(bool print = false);

    /**
     * @brief Get the available memory in KB.
     */
    static long get_available_memory();

    /** 
     * @brief Estimate the memory usage of a sparse matrix.
     * @param matrix The sparse matrix.
     * @return The estimated memory usage in bytes.
     */
    static size_t estimateSparseMatrixMemoryUsage(const Eigen::SparseMatrix<double>& matrix);

    /**
     * @brief Timer function to measure the duration of the calculation.
     */
    static void timer();

    /**
     * @brief Set the number of threads for OpenMP parallelization.
     */
    static void set_omp_threads(const Eigen::SparseMatrix<double>& matrix1, int nb_matrix);

 private:
    static std::chrono::high_resolution_clock::time_point start_time;
    static bool is_running;

};  

