#include "../include/mean_curvature.h"
#include "igl/cotmatrix.h"
#include "igl/massmatrix.h"
#include "igl/invert_diag.h"
#include "igl/per_vertex_normals.h"
#include <cmath>

void mean_curvature(
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        Eigen::VectorXd & H){

    const int n = (int)V.rows();

    Eigen::SparseMatrix<double> L, M, M_inverted;
    igl::cotmatrix(V,F,L);
    igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,M);
    igl::invert_diag(M, M_inverted);

    Eigen::MatrixXd N;
    igl::per_vertex_normals(V, F, N);

    Eigen::MatrixXd HN = M_inverted * -L * V;

    H.resize(n);
    for(int i = 0; i < n; i++) {
        const double value = std::sqrt(HN(i, 0) * HN(i, 0) +
                                       HN(i, 1) * HN(i, 1) +
                                       HN(i, 2) * HN(i, 2)) / 2;

        if(value == 0)
        {
            H(i) = 0.0;
        } else if(HN(i, 0) * N(i, 0) > 0 ||
                  HN(i, 1) * N(i, 1) > 0 ||
                  HN(i, 2) * N(i, 2) > 0)
        {
            H(i) = value;
        } else
        {
            H(i) = -1.0 * value;
        }
    }

}
