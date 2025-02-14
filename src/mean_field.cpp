#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <array>
#include <iomanip>
#include <complex>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

#include "clock.hpp"

using namespace Spectra;


const double eps = 1e-6; // threshold for the convergence of the self-consistent mean-field method

// Eigen::SparseMatrix<std::complex<double>> h_MF (std::complex<double> psi, int p, double mu, double J, int q)
// /* Returns the single particle hamiltonian in the mean-field approximation*/
// {
//     Eigen::SparseMatrix<std::complex<double>> h(2*p+1, 2*p+1);

//     // diagonal elements 
//     for (int i = 0; i < 2*p+1; i++)
//     {
//         h.insert(i, i) = -mu*i + (0.5)*i*(i-1) + q*J*psi*psi;
//     }

//     //off diagonal elements 
//     for (int j = 1; j < 2*p; j++)
//     {
//         h.insert(j+1,j) = -q*J*psi*sqrt(j+1);
//         h.insert(j-1, j) = -q*J*psi*sqrt(j);
//     }

//     h.insert(1,0) = -q*J*psi*sqrt(1);
//     h.insert(2*p-1, 2*p) = -q*J*psi*sqrt(2*p);

//     return h;
// }



void h_MF (double psi, int p, double mu, double J, int q, Eigen::MatrixXd& h)
{
    // fill diagonal elements
    for (int i=0; i<2*p+1; i++)
    {
        h(i,i) = -mu*i + 0.5*i*(i-1) + q*J*psi*psi;
    }

    // fill off-diagonal elements
    for (int j=0; j<2*p+1; j++)
    {
        if (j == 0) {
            h(1, 0) = -q * J * psi * sqrt(1);
        } else if (j == 2 * p) {
            h(2 * p - 1, 2 * p) = -q * J * psi * sqrt(2 * p);
        } else {
            h(j + 1, j) = -q * J * psi * sqrt(j + 1);
            h(j - 1, j) = -q * J * psi * sqrt(j);
        }
    }
}





double SF_density(Eigen::VectorXd& phi0, int p)
/* Returns the mean value of the annihilation operator <phi0|a|phi0>, that is the superfluid density of the state phi0 */
{

    double a = 0; 
    for(int i=1; i<2*p+1; i++)
    {
        a += sqrt(i)*phi0[i]*phi0[i-1];
    }
    return a; 
}



double SCMF(double mu, double J, int q ,double psi0)
/* Self-consistent mean-field method to highlight the Superfluid-Mott insulator transition
Parameters: 
- mu = mu/U (without dimension)
- J = J/U 
- q : number of neighbours of a site i in the specific lattice studied
- psi0: initial ansatz for the superfluid order parameter */

{

    // std::cout << " *** Start: Self-consistent mean-field method ***" << std::endl;
    Clock clock;
    clock.start();

    double psi = psi0; // initial ansatz for the superfluid order parameter
    double psi_new = psi0; // new ansatz for the superfluid order parameter
    Eigen::VectorXd phi0; // GS eigenvector
    double e0; // ground state energy
    double e0_new; // new ground state energy
    double tmp = 0; //temporary variable
    int N_itt=1; 
    int N_itt_inner; // number of iterations
    int p; 


    Eigen::MatrixXd h(1000, 1000); // alllocation of a huge dense matrix, in which we will store the iterative
                                   // single particle hamiltonian in the mean-field approximation
    h.setZero();

    // std::cout << "*** Computing the superfluid order parameter Psi up to " << eps << " precision ***" << std::endl; 
    do
    {
        // std::cout << "*** Inner Loop: Computing the ground state e0 and phi0 up to " << eps << " precision ***" << std::endl;
        N_itt_inner = 0; 
        p = 1; 
        h_MF(psi, p, mu, J, q, h); // single particle hamiltonian in the mean-field approximation
        // Define a submatrix view of the matrix h; without allocating new memory 
        Eigen::Block<Eigen::MatrixXd> sub_h = h.block(0, 0, 2*p+1, 2*p+1);
        Spectra::DenseSymMatProd<double> op(sub_h); 
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, 2*p);
        eigs.init();
        int nconv = eigs.compute(SortRule::SmallestAlge);
        if (eigs.info() != Spectra::CompInfo::Successful) { // verify if the eigen search is a success
            throw std::runtime_error("Eigenvalue computation for the mean-field single particle hamiltonian failed.");
        }
        else{
            phi0 = eigs.eigenvectors().col(0); // GS eigenvector
            e0 = eigs.eigenvalues()[0]; // GS eigenvalue 
        }
        
        do
        {
            p++; 
            h_MF(psi, p, mu, J, q, h); // single particle hamiltonian in the mean-field approximation
            Eigen::Block<Eigen::MatrixXd> sub_h = h.block(0, 0, 2*p+1, 2*p+1);
            Spectra::DenseSymMatProd<double> op(sub_h); 
            Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, 2*p);
            eigs.init();
            int nconv = eigs.compute(SortRule::SmallestAlge);
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
 
        // std::string message = "Try " + std::to_string(N_itt) + ": e0 converged in " + std::to_string(N_itt_inner) + " iterations, and it took";
        // clock.time_us(message); 
        N_itt++; 

        psi_new = SF_density(phi0, p); // new ansatz for the superfluid order parameter
        tmp = std::abs(psi_new - psi);
        psi = psi_new;
    }while(tmp>eps);
    
    // std::string message = "The superfluid order parameter Psi converged in " + std::to_string(N_itt) + " iterations, and it took";
    // clock.time_ms(message); 
    // std::cout << " *** End: Self-consistent mean-field method ***" << std::endl;
    
    return psi;
}


