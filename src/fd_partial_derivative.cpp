#include "fd_partial_derivative.h"
#include <vector>
void fd_partial_derivative(
        const int nx,
        const int ny,
        const int nz,
        const double h,
        const int dir,
        Eigen::SparseMatrix<double> & D) {

    //Set the number of rows by the direction of partial derivative.
    int m = 0;
    if(dir == 0) {
        m = (nx - 1) * ny * nz;
    } else if(dir == 1) {
        m = nx * (ny - 1) * nz;
    } else {
        m = nx * ny * (nz - 1);
    }

    //Resize the output matrix
    D.resize(m, nx * ny * nz);
    //Build a list of Triplet to store the change of matrix D
    std::vector<Eigen::Triplet<double>> tripletList;

    if(dir == 0) {
        for(int i = 0; i < nx - 1; i++) {
            for(int j = 0; j < ny; j++) {
                for(int k = 0; k < nz; k++) {
                    int index = i + nx * j + nx * ny * k;
                    tripletList.push_back(Eigen::Triplet<double>(index, index, -1.0 / h));
                    tripletList.push_back(Eigen::Triplet<double>(index, index + 1, 1.0 / h));
                }
            }
        }
    } else if(dir == 1) {
        for(int i = 0; i < nx; i++) {
            for(int j = 0; j < ny - 1; j++) {
                for(int k = 0; k < nz; k++) {
                    int index = i + nx * j + nx * ny * k;
                    tripletList.push_back(Eigen::Triplet<double>(index, index, -1.0 / h));
                    tripletList.push_back(Eigen::Triplet<double>(index, index + nx, 1.0 / h));
                }
            }
        }
    } else {
        for(int i = 0; i < nx; i++) {
            for(int j = 0; j < ny; j++) {
                for(int k = 0; k < nz - 1; k++) {
                    int index = i + nx * j + nx * ny * k;
                    tripletList.push_back(Eigen::Triplet<double>(index, index, -1.0 / h));
                    tripletList.push_back(Eigen::Triplet<double>(index, index + nx * ny, 1.0 / h));
                }
            }
        }
    }

    //Use the triplet list to set matrix D
    D.setFromTriplets(tripletList.begin(), tripletList.end());

}
