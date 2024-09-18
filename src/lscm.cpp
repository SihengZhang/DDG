#include "lscm.h"
#include "vector_area_matrix.h"
#include "igl/cotmatrix.h"
#include "igl/repdiag.h"
#include "igl/massmatrix.h"
#include "igl/eigs.h"
#include "igl/vector_area_matrix.h"

void lscm(
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        Eigen::MatrixXd & U) {


    Eigen::SparseMatrix<double> A;
    vector_area_matrix(F,A);

    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(V,F,L);
    Eigen::SparseMatrix<double> L2;
    igl::repdiag(L,2,L2);

    Eigen::SparseMatrix<double> Q;
    Q = -L2 - 2.0 * A;

    Eigen::SparseMatrix<double> M;
    igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,M);
    Eigen::SparseMatrix<double> M2;
    igl::repdiag(M,2,M2);

    Eigen::MatrixXd sU;
    Eigen::VectorXd sS;
    igl::eigs(Q,M2,3,igl::EIGS_TYPE_SM,sU,sS);

    U.resize(V.rows(),2);
    U << sU.col(0).head(V.rows()), sU.col(0).tail(V.rows());


}
