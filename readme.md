The implementation of Batch OMP.

# Run

First, you need to prepare VS and Cython for compiling code, then download <a href="https://github.com/RLovelett/eigen.git">Eigen</a> into */third-party/eigen/*. In fact only */main.cpp*、*/main.h*、*omp_wrapper.pyx*、*setup.py* must be necessary. I also have written omp algorithm with Numpy, including Origin OMP、Cholesky OMP and Batch OMP in */omp.py*, but it's a bit slow. You can see the excellent performance of Cython in */demo.py*, however, it is still very slow compared to ompbox10 (matlab).

# References

[1] M. Elad, R. Rubinstein, and M. Zibulevsky, "Efficient Implementation of the K-SVD Algorithm 
using Batch Orthogonal Matching Pursuit", Technical Report - CS, Technion, April 2008.