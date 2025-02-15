#include <iostream>
#include <chrono>
#include <iomanip>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

#include "operator.h"




const double eps = 1e-3; // threshold for the convergence of the self-consistent mean-field method

Eigen::SparseMatrix<std::complex<double>> h_MF (std::complex<double> psi, int p, double mu, double J, int q)
/* Returns the single particle hamiltonian in the mean-field approximation*/
{
    Eigen::SparseMatrix<std::complex<double>> h(2*p+1, 2*p+1);

    // diagonal elements 
    for (int i = 0; i < 2*p+1; i++)
    {
        h.insert(i, i) = -mu*i + (0.5)*i*(i-1) + q*J*psi*psi;
    }

    //off diagonal elements 
    for (int j = 1; j < 2*p; j++)
    {
        h.insert(j+1,j) = -q*J*psi*sqrt(j+1);
        h.insert(j-1, j) = -q*J*psi*sqrt(j);
    }

    h.insert(1,0) = -q*J*psi*sqrt(1);
    h.insert(2*p-1, 2*p) = -q*J*psi*sqrt(2*p);

    return h;
}

std::complex<double> a_mean(Eigen::MatrixXcd& phi, int p)
/* Returns the mean value of the annihilation operator <phi0|a|phi0> */
{
    Eigen::VectorXcd phi0 = phi.col(0);

    std::complex<double> a = 0;

    for(int i=1; i<2*p+1; i++)
    {
        a += sqrt(i)*phi0[i]*(std::conj(phi0[i-1]));
    }
    return a; 
}




void SCMF (double mu, double J, int q ,double psi0)
/* Self-consistent mean-field method to highlight the Superfluid-Mott insulator transition
Parameters: 
- mu = mu/U (without dimension)
- J = J/U 
- q : number of neighbours of a site i in the specific lattice studied
- psi0: initial ansatz for the superfluid order parameter */

{

    std::cout << " *** Starting: Self-consistent mean-field method ***" << std::endl;
    auto start = std::chrono::high_resolution_clock::now(); // Start the chronometer
    

    std::complex<double> psi = psi0; // initial ansatz for the superfluid order parameter
    std::complex<double> psi_new = psi0; // new ansatz for the superfluid order parameter
    Eigen::MatrixXcd phi; // eigenvectors
    Eigen::VectorXcd E; // eigenvalues
    double e0; // ground state energy
    double e0_new; // new ground state energy
    double tmp = 0; //temporary variable



    do
    {
        int p = 1; 
        Eigen::SparseMatrix<double> h = h_MF(psi, p, mu, J, q); // single particle hamiltonian in the mean-field approximation
        Operator H(std::move(h));
        E = H.IRLM_eigen(1, phi); // IRLM computation of the lowest eigenvalue e0
        e0 = E[0].real(); // ground state energy
        p++;

        do
        {
            Eigen::SparseMatrix<double> h = h_MF(psi, p, mu, J, q); // single particle hamiltonian in the mean-field approximation
            Operator H(std::move(h));
            E = H.IRLM_eigen(1, phi); // IRLM computation of the lowest eigenvalue e0
            e0_new = E[0].real(); // new ground state energy
            tmp = std::abs(e0_new - e0);
            e0 = e0_new;
            p++;
        }while(tmp>eps); 

        psi_new = a_mean(phi, p); // new ansatz for the superfluid order parameter
    }while(std::abs(psi_new - psi)>eps); 

    auto end = std::chrono::high_resolution_clock::now(); // Stop the chronometer
    std::chrono::duration<double> duration = end - start; // Calculate the duration
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    duration -= minutes;
    auto seconds = duration;

    std::cout << " *** End: Self-consistent mean-field method *** \n *** Duration of the function:  " 
              << std::setw(2) << std::setfill('0') << hours.count() << ":"
              << std::setw(2) << std::setfill('0') << minutes.count() << ":"
              << std::setw(2) << std::setfill('0') << seconds.count() << " ***" << std::endl;
              
}