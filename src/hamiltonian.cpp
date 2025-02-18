#include<vector>
#include<Eigen/Dense>
#include<Eigen/SparseCore>

#include "hamiltonian.hpp"


/////  IMPLEMENTATION OF THE BH CLASS METHODS  /////

    
    /* ELEMENTARY FUNCTIONS */

/* Calculate the sum of the elements of a vector between 2 index */
int BH::sum(const Eigen::VectorXd& state, int index1, int index2) const { 
	int s = 0;
	for (int i = index1; i <= index2; i++) {
		s += state[i];
	}
	return s;
}


    /* DIMENSION OF THE HILBERT SPACE */

/* Calculate the binomial coefficient */
int BH::binomial(int n, int k) const{
	if (k==0 || k==n){
		return 1;
	}
	if (k > n/2) {
		return binomial(n,n-k);
	}
	else{
		return n*binomial(n-1,k-1)/k;
	}
}

/* Calculate the dimension of the Hilbert space for n bosons on m sites */
int BH::dimension(int m, int n) const{
	return binomial(m + n - 1, n);
}


    /* INITIALIZE THE HILBERT SPACE BASIS */

/* Calculate the next Fock state of the Hilbert space in lexicographic order */
bool BH::next_lexicographic(Eigen::VectorXd& state, int m, int n) const {
	for (int k = m - 2; k > -1; k--) {
		if (state[k] != 0) {
			state[k] -= 1;
			state[k + 1] = n - sum(state, 0, k);
			for (int i = k + 2; i < m; i++) {
				state[i] = 0;
			}
			return true;
		}
	}
	return false;
}

/* Create the matrix that has the Fock states of the Hilbert space basis in columns */
Eigen::MatrixXd BH::init_lexicographic(int m, int n) const {
    Eigen::MatrixXd basis(m, D);
    Eigen::VectorXd state = Eigen::VectorXd::Zero(m);
    state(0) = n;
    int col = 0;
    do {
        basis.col(col++) = state;
    } while (next_lexicographic(state, m, n));
    return basis;
}


    /* SORT THE HILBERT SPACE BASIS TO FACILITATE CALCULUS */

/* Calculate the unique tag of the kth column of the matrix */
double BH::calculate_tag(const Eigen::MatrixXd& basis, const std::vector<int>& primes, int k) const {
	double tag = 0;
	for (int i = 0; i < basis.rows(); i++) {
		tag += basis.coeff(i, k) * log(primes[i]);
	}
	return tag;
}

/* Calculate and store the tags of each state of the Hilbert space basis */
Eigen::VectorXd BH::calculate_tags(const Eigen::MatrixXd& basis, const std::vector<int>& primes) const {
	Eigen::VectorXd tags(basis.cols());
	for (int i = 0; i < basis.cols(); i++) {
		tags[i] = calculate_tag(basis, primes, i);
	}
	return tags;
}

/* Sort the states of the Hilbert space by ascending order compared by their tags*/
void BH::sort_basis(Eigen::VectorXd& tags, Eigen::MatrixXd& basis) const {
    std::vector<int> indices(tags.size());
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&tags](int a, int b) {return tags[a] < tags[b];});
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        while (indices[i] != i) {
            int j = indices[i];
            std::swap(tags[i], tags[j]);
            basis.col(i).swap(basis.col(j));
            std::swap(indices[i], indices[j]);
        }
    }
}

/* Gives the index of the wanted tag x by the Newton method */
int BH::search_tag(const Eigen::VectorXd& tags, double x) const {
	int a = 0;
	int b = tags.size() - 1;
	int m = (a + b) / 2;
	while (fabs(tags[m] - x) > 1e-3 && a <= b) {
		if (tags[m] < x) {
			a = m + 1;
		}
		else {
			b = m - 1;
		}
		m = (a + b) / 2;
	}
	return m;
}


    /* FILL THE HAMILTONIAN OF THE SYSTEM */

/* Fill the hopping term of the Hamiltonian */
void BH::fill_hopping(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const std::vector<std::vector<int>>& neighbours, const std::vector<int>& primes, Eigen::SparseMatrix<double>& hmatrix, double J) const {
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(basis.cols() * basis.rows() * neighbours.size());
    for (int k = 0; k < basis.cols(); k++) {
        for (int i = 0; i < static_cast<int>(neighbours.size()); i++) {
            for (int j = 0; j < static_cast<int>(neighbours[i].size()); j++) {
                Eigen::VectorXd state = basis.col(k);
                if (basis.coeff(i, k) >= 1 && basis.coeff(j, k) >= 1) {
                    state[i] += 1;
                    state[j] -= 1;
                    double x = calculate_tag(state, primes, i);
                    int index = search_tag(tags, x);
                    assert(index >= 0 && index < tags.size()); // Add assertion to check index bounds
                    double value = sqrt((basis.coeff(i, k) + 1) * basis.coeff(j, k));
                    tripletList.push_back(Eigen::Triplet<double>(index, k, -J * value));
                    tripletList.push_back(Eigen::Triplet<double>(k, index, -J * value));
                }
            }
        }
    }
    hmatrix.setFromTriplets(tripletList.begin(), tripletList.end());
}

/* Fill the interaction term of the Hamiltonian */
void BH::fill_interaction(const Eigen::MatrixXd& basis, Eigen::SparseMatrix<double>& hmatrix, double U) const {
	std::vector<Eigen::Triplet<double>> tripletList;
	tripletList.reserve(hmatrix.nonZeros() + basis.cols());
	for (int k = 0; k < hmatrix.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(hmatrix, k); it; ++it) {
			tripletList.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
		}
	}
	for (int k = 0; k < basis.cols(); k++) {
		double value = 0;
		for (int i = 0; i < basis.rows(); i++) {
			double ni = basis.coeff(i, k);
			value += (ni + 1) * ni;
		}
		tripletList.push_back(Eigen::Triplet<double>(k, k, U * value));
	}
	hmatrix.setFromTriplets(tripletList.begin(), tripletList.end());
}

/* Fill the chemical potential term of the Hamiltonian */
void BH::fill_chemical(const Eigen::MatrixXd& basis, Eigen::SparseMatrix<double>& hmatrix, double mu) const {
	std::vector<Eigen::Triplet<double>> tripletList;
	tripletList.reserve(hmatrix.nonZeros() + basis.cols());
	for (int k = 0; k < hmatrix.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(hmatrix, k); it; ++it) {
			tripletList.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
		}
	}
    std::vector<Eigen::Triplet<double>> tripletList2;
	for (int k = 0; k < basis.cols(); k++) {
		double value = 0;
		for (int i = 0; i < basis.rows(); i++) {
			double ni = basis.coeff(i, k);
			value += ni;
		}
		tripletList.push_back(Eigen::Triplet<double>(k, k, -mu * value));
	}
	hmatrix.setFromTriplets(tripletList.begin(), tripletList.end());
}


    /* CONSTRUCTOR */

/* Constructor for the Bose-Hubbard model */
BH::BH(const std::vector<std::vector<int>>& neighbours, int m, int n, double J, double U, double mu) : neighbours(neighbours), m(m), n(n), D(dimension(m,n)), J(J), U(U), mu(mu), H(D,D) {
	Eigen::MatrixXd basis = init_lexicographic(m, n);
    H.setZero();
    if (J != 0) {
        std::vector<int> primes = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 };
        Eigen::VectorXd tags = calculate_tags(basis, primes);
        sort_basis(tags, basis);
        fill_hopping(basis, tags, neighbours, primes, H, J);
    }
    if (U != 0) {
        fill_interaction(basis, H, U);
    }
    if (mu != 0) {
        fill_chemical(basis, H, mu);
    }
}


Eigen::SparseMatrix<double> BH::create_combined_hamiltonian(const std::vector<std::vector<int>>& neighbours, int m, int n, double J, double U, double mu) {
    int total_dimension = 0;
    std::vector<Eigen::SparseMatrix<double>> hamiltonians;
    for (int bosons = 1; bosons <= n; ++bosons) {
        BH hamiltonian(neighbours, m, bosons, J, U, mu);
        Eigen::SparseMatrix<double> hmatrix = hamiltonian.getHamiltonian();
        hamiltonians.push_back(hmatrix);
        total_dimension += hmatrix.rows();
    }
    Eigen::SparseMatrix<double> combined_hamiltonian(total_dimension, total_dimension);
    std::vector<Eigen::Triplet<double>> tripletList;
    int offset = 0;
    for (const auto& hmatrix : hamiltonians) {
        for (int k = 0; k < hmatrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(hmatrix, k); it; ++it) {
                tripletList.push_back(Eigen::Triplet<double>(it.row() + offset, it.col() + offset, it.value()));
            }
        }
        offset += hmatrix.rows();
    }
    combined_hamiltonian.setFromTriplets(tripletList.begin(), tripletList.end());
    return combined_hamiltonian;
}


    /* UTILITY FUNCTIONS */

/* get the Hamiltonian matrix */
Eigen::SparseMatrix<double> BH::getHamiltonian() const {
	return H;
}