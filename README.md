<div align="center">
    <a href="https://github.com/Maximilien-Hantonne/Bose-Hubbard-Phase-Transition">
        <img src="https://github.com/Maximilien-Hantonne/Bose-Hubbard-Phase-Transition/blob/main/figures/mean_field/mean_field_plot.svg" alt="Bose-Hubbard Model Diagram" width="350">
    </a>
    <h3 align="center">Mott insulator to Superfluid transition</h3>
    <p align="center">
        An implementation of the Bose Hubbard model with exact and mean field calculations to characterize the phase of the system.
    </p>
</div>

# Bose-Hubbard Phase Transition

This repository contains the implementation of the Bose-Hubbard model, which describes interacting bosons on a lattice. The aim of this project is to study the phase transition in the Bose-Hubbard model using various numerical methods.

## Project Structure

- `include/`: Contains the header files for the project.
  - `hamiltonian.hpp`: Defines the `BH` class representing the Bose-Hubbard Hamiltonian.
  - `neighbours.hpp`: Defines the `Neighbours` class for generating the list of neighbours for different lattice structures.
  - `operator.hpp`: Defines the `Operator` class for various matrix diagonalization methods.
  - `analysis.hpp`: Defines the `Analysis` class for calculating and saving physical quantities with an exact or a mean field approach.
  - `resource.hpp`: Defines the `Resource` class for utility functions for timing, memory usage and parallelization.

- `src/`: Contains the source files for the project.

- `external/`: Contains external dependencies.
  - `spectra/`: Header-only C++ library for large scale eigenvalue problems.

- `CMakeLists.txt`: CMake configuration file for building the project.

- `figures/`: Directory to store the generated plots.

## Dependencies

- [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html): A C++ template library for linear algebra.
- [Spectra](https://spectralib.org/): A header-only C++ library for large scale eigenvalue problems.
- [TkInter](https://docs.python.org/3/library/tkinter.html): A standard GUI toolkit for Python (required for viewing plots).
- [OpenMP](https://www.openmp.org/): An API for parallel programming in C, C++, and Fortran.
- [tqdm](https://github.com/tqdm/tqdm.cpp): A header-only C++ port of the popular python module tqdm.
- (Optional) [Doxygen](http://www.doxygen.nl/): A tool for generating documentation from annotated C++ sources.
  
## Building the Project (Linux)

To build the project on a Linux system, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/Maximilien-Hantonne/Bose-Hubbard-Phase-Transition.git
    cd Bose-Hubbard-Phase-Transition
    ```

2. Create a build directory and navigate to it:
    ```sh
    mkdir build
    cd build
    ```

3. Run CMake to configure the project:
    ```sh
    cmake ..
    ```

4. Build the project:
    ```sh
    cmake --build .
    ```

## (Optional) Generating Documentation (Linux)

To generate the documentation using Doxygen, follow these steps:

### HTML Documentation

To generate the HTML documentation using Doxygen on a Linux system, follow these steps:

1. Ensure Doxygen is installed on your system.

2. Run Doxygen with the provided `Doxyfile`:
    ```sh
    doxygen Doxyfile
    ```

The documentation will be generated in the `docs/html` directory. The main page can be opened by navigating to `docs/html/index.html` in your web browser:

```sh
firefox docs/html/index.html
```

### LaTeX Documentation

To generate the LaTeX documentation using Doxygen on a Linux system, follow these steps:

1. Ensure Doxygen and LaTeX are installed on your system. You can install LaTeX using:
    ```sh
    sudo apt-get install texlive-full
    ```

2. Run Doxygen with the provided `Doxyfile`:
    ```sh
    doxygen Doxyfile
    ```

3. Navigate to the `docs/latex` directory:
    ```sh
    cd docs/latex
    ```

4. Compile the LaTeX documentation:
    ```sh
    make
    ```

The PDF documentation will be generated in the `docs/latex` directory.

## Usage (Linux)

The project provides classes to represent the Bose-Hubbard Hamiltonian and to perform various operations and calculations related to the model. The main classes are:

- `BH`: Represents the Bose-Hubbard Hamiltonian.
- `Neighbours`: Generates the list of neighbours for different lattice structures i.e. the geometry of the lattice.
- `Operator`: Provides various matrix diagonalization methods.
- `Analysis`: Computes the physical quantities for exact or mean-field approach of Bose-Hubbard model.

### Command-Line Options

The `main` function parses command-line arguments to set up the parameters for the Bose-Hubbard model.

Command-line options:
- `-m, --sites`: Number of sites in the lattice.
- `-n, --bosons`: Number of bosons in the lattice.
- `-J, --hopping`: Hopping parameter.
- `-U, --interaction`: On-site interaction parameter.
- `-u, --potential`: Chemical potential.
- `-r, --range`: Range for chemical potential and interaction.
- `-s, --step`: Step for chemical potential and interaction.
- `-f, --fixed`: Fixed parameter (J, U, or u)
- `-t, --type` : type of calculation (exact or mean)
- `-h, --help`: Display usage information.

### Viewing Plots (Linux)
To view the plots generated by the `plot.py` or `plot_mean_field.py` script, you need to have the TkInter windowing system installed. You can install it using:
```sh
sudo apt-get install python3-tk
```

The `plot.py` script will generate plots for the gap ratio, boson density, and compressibility and then the projection of the PCA on the data and visualize the projected points along with their k-means clustering labels.

The `plot_mean_field.py` will generate the map of the order parameter to visualize the transition Mott insulator/superfluid. 

Some already calculated plots can be seen in the [figures](https://github.com/Maximilien-Hantonne/Bose-Hubbard-Phase-Transition/tree/main/figures) folder.

### Launching the program (Linux)
Make sure you have the required dependencies installed. You can then launch the program by typing for example :
#### For exact calculations :
```sh
./QuantumProject -m 5 -n 5 -J 100 -U 0 -u 0 -r 100 -s 5 -f "J" --t "exact"
```

#### For mean-field calculations:
The parameters are not required in the simulation and don't need to be specified. You only need to precise the `mean` in the command-line options and choose the other parameters randomly. For example :
```sh
./QuantumProject -n 1000 -J 0 -rJ 0.25 -u 0 -ru 4 --t "mean"
```

## Authors

- [Maximilien HANTONNE](https://github.com/Maximilien-Hantonne)
- [Rémy LYSCAR](https://github.com/Remy-Lyscar)
- [Alexandre MENARD](https://github.com/alexandremnd)
  
## Contributing

Feel free to contribute to our project

## License

This project is licensed under the GNU GPLv3 - see [LICENCE](https://github.com/Maximilien-Hantonne/Bose-Hubbard-Phase-Transition/blob/main/LICENSE) file for details.
