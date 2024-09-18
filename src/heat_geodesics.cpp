#include "heat_geodesics.h"
#include "igl/cotmatrix.h"
#include "igl/massmatrix.h"
#include "igl/doublearea.h"
#include "igl/grad.h"
#include "igl/boundary_facets.h"
#include "igl/unique.h"
#include "igl/min_quad_with_fixed.h"


void heat_geodesics(
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        const Eigen::VectorXi & gamma,
        const double t,
        Eigen::VectorXd & D) {

    Eigen::SparseMatrix<double> L, M, G;
    Eigen::VectorXd dblA;

    igl::cotmatrix(V,F,L);
    igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,M);
    igl::doublearea(V,F,dblA);
    igl::grad(V,F,G);


    Eigen::SparseMatrix<double> Div = -0.25 * G.transpose() * dblA.colwise().replicate(3).asDiagonal();
    Eigen::SparseMatrix<double> Q = M - t * L;

    Eigen::VectorXi boundary_indices;
    Eigen::MatrixXi O;
    igl::boundary_facets(F,O);
    igl::unique(O,boundary_indices);

    Eigen::VectorXd u0 = Eigen::VectorXd::Zero(V.rows(), 1);
    for(int g = 0;g<gamma.size();g++)
    {
        u0(gamma(g)) = 1;
    }


    Eigen::VectorXd u;
    {
        igl::min_quad_with_fixed(Q, u0, Eigen::VectorXi(), Eigen::VectorXd(), Eigen::SparseMatrix<double>(), Eigen::VectorXd(), true, u);
    }

    if(boundary_indices.size()>0){
        Eigen::VectorXd ud;
        igl::min_quad_with_fixed(Q, u0, boundary_indices, Eigen::VectorXd::Zero(boundary_indices.size()).eval(), Eigen::SparseMatrix<double>(), Eigen::VectorXd(), true, ud);
        u = 0.5 * (u + ud);
    }

    Eigen::VectorXd gu = G * u;
    const int m = (int)F.rows();
    for(int i = 0; i < m; i++)
    {
        Eigen::VectorXd v(3);
        v << gu(i), gu(m + i), gu(2 * m + i);
        double norm = v.stableNorm();

        gu(i) /= norm;
        gu(m + i) /= norm;
        gu(2 * m + i) /= norm;
    }


    L *= -1;
    const Eigen::VectorXd dX = -Div * gu;
    const Eigen::SparseMatrix<double> Aeq = M.diagonal().transpose().sparseView();
    const Eigen::VectorXd Beq = Eigen::VectorXd::Zero(1);
    igl::min_quad_with_fixed(L, -dX, Eigen::VectorXi(), Eigen::VectorXd(), Aeq, Beq, true, D);


}
