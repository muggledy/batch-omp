The implementation of Batch OMP. I have written omp algorithm with Numpy, including Origin OMP、Cholesky OMP and Batch OMP in */code/omp.py*, but it's a bit slow. You can see the excellent performance of Cython version in */demo.py*, however, it is still very slow compared to ompbox10 (matlab). You can clone this repository with `git clone --depth 1 git@github.com:muggledy/batch-omp.git`.

# Run

First, add <a href="https://gitlab.com/libeigen/eigen">Eigen</a> library to */third-party* by executing `git submodule init` and `git submodule update`.

Second, you need to prepare Cython for compiling */code/batch_omp.pyx* to `.cpp` file and gcc(C++), so the `.cpp` file can be compiled to `.pyd`(on Windows), which can be directly imported in a Python session. To accomplish the installation, cd into sub-directory */code* and run `python setup.py build_ext --inplace`, you will get a `.pyd` file.

# References

[1] M. Elad, R. Rubinstein, and M. Zibulevsky, "Efficient Implementation of the K-SVD Algorithm 
using Batch Orthogonal Matching Pursuit", Technical Report - CS, Technion, April 2008.
