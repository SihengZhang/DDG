#include "fd_grad.h"

void fd_grad(
        const int nx,
        const int ny,
        const int nz,
        const double h,
        Eigen::SparseMatrix<double> & G) {

    G.resize((nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1), nx * ny * nz);

    std::vector<Eigen::Triplet<double>> tripletList;

    for(int i = 0; i < nx - 1; i++) {
        for(int j = 0; j < ny; j++) {
            for(int k = 0; k < nz; k++) {
                int index = i + nx * j + nx * ny * k;
                tripletList.push_back(Eigen::Triplet<double>(index, index, -1.0 / h));
                tripletList.push_back(Eigen::Triplet<double>(index, index + 1, 1.0 / h));
            }
        }
    }

    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny - 1; j++) {
            for(int k = 0; k < nz; k++) {
                int index = i + nx * j + nx * ny * k;
                tripletList.push_back(Eigen::Triplet<double>(index + (nx - 1) * ny * nz, index, -1.0 / h));
                tripletList.push_back(Eigen::Triplet<double>(index + (nx - 1) * ny * nz, index + nx, 1.0 / h));
            }
        }
    }

    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            for(int k = 0; k < nz - 1; k++) {
                int index = i + nx * j + nx * ny * k;
                tripletList.push_back(Eigen::Triplet<double>(index + (nx - 1) * ny * nz + nx * (ny - 1) * nz, index, -1.0 / h));
                tripletList.push_back(Eigen::Triplet<double>(index + (nx - 1) * ny * nz + nx * (ny - 1) * nz, index + nx * ny, 1.0 / h));
            }
        }
    }

    G.setFromTriplets(tripletList.begin(), tripletList.end());

}
