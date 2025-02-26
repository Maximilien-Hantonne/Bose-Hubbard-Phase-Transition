#pragma once

#include <chrono>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <Eigen/SparseCore>


namespace Resource
{
    /**
     * @brief Get the memory usage of the program in KB and optionally print it.
     * @param print If true, print the memory usage.
     * @return The memory usage in KB.
     */
    long get_memory_usage(bool print = false);

    /**
     * @brief Get the available memory in KB.
     */
    long get_available_memory();

    /** 
     * @brief Estimate the memory usage of a sparse matrix.
     * @param matrix The sparse matrix.
     * @return The estimated memory usage in bytes.
     */
    size_t estimateSparseMatrixMemoryUsage(const Eigen::SparseMatrix<double>& matrix);

    /**
     * @brief Timer function to measure the duration of the calculation.
     */
    void timer();

    /**
     * @brief Set the number of threads for OpenMP parallelization.
     */
    void set_omp_threads(const Eigen::SparseMatrix<double>& matrix1, int nb_matrix);

    static std::chrono::high_resolution_clock::time_point start_time;
    static bool is_running = false;

}

