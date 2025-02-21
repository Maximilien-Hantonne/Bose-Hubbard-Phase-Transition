#pragma once

#include <cmath>
#include <vector>


/**
 * @brief Class to generate the list of neighbours for a 1D chain, 2D square lattice, or 3D cubic lattice.
 * 
 * @warning This class is not thread-safe.
 */

class Neighbours {
public:

    /**
     * @brief Constructor for the Neighbours class.
     * 
     * @param m Number of sites in the lattice.
     */
    Neighbours(int m);

    /**
     * @brief Destructor for the Neighbours class.
     */
    ~Neighbours();

    /**
     * @brief Generate the list of neighbours for a 1D chain.
     * 
     * @param m Number of sites in the chain.
     * @param closed By default, closed = true for periodic boundary conditions, closed = false for open boundary conditions.
     * @warning Ensure that the number of sites is greater than 1.
     */
    void chain_neighbours(bool closed = true);

    /**
     * @brief Generate the list of neighbours for a 2D square lattice.
     * 
     * @param m Number of sites in the square.
     * @param closed By default, closed = true for periodic boundary conditions, closed = false for open boundary conditions.
     * @warning Ensure that the number of sites is a perfect square.
     */
    void square_neighbours(bool closed = true);

    /**
     * @brief Generate the list of neighbours for a 2D square lattice.
     * 
     * @param m Number of sites in the square.
     * @param closed By default, closed = true for periodic boundary conditions, closed = false for open boundary conditions.
     * @warning Ensure that the number of sites is a perfect cube.
     */
    void cube_neighbours(bool closed = true);
    
    /** 
    * @brief Return the list of neighbours
    * @return std::vector<std::vector<int>> The list of neighbours for each site of the lattice. 
    */
    std::vector<std::vector<int>> getNeighbours() const;
    
private:
    int m;
    std::vector<std::vector<int>> neighbours;
};