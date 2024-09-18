#include "massmatrix.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <igl/doublearea.h>
void massmatrix(
        const Eigen::MatrixXd & l,
        const Eigen::MatrixXi & F,
        Eigen::DiagonalMatrix<double,Eigen::Dynamic> & M) {

    //get the number of vertices
    int n = F.maxCoeff() + 1;

    //set the size of diagonal matrix M to n
    M.diagonal().resize(n);

    //calculate doubled areas of all triangle
    Eigen::VectorXd a;
    igl::doublearea(l, 0, a);

    for(int i = 0; i < F.rows(); i++) {
        M.diagonal()(F(i, 0)) += a(i);
        M.diagonal()(F(i, 1)) += a(i);
        M.diagonal()(F(i, 2)) += a(i);
    }

    //multiply M by a scalar to get 1/3 of the sum of areas
    double scalar_factor = 1.0 / 6;
    M = M * scalar_factor;
}

