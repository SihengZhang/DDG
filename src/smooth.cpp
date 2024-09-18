#include "smooth.h"
#include "cotmatrix.h"
#include "massmatrix.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <igl/edge_lengths.h>
void smooth(
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        const Eigen::MatrixXd & G,
        double lambda,
        Eigen::MatrixXd & U) {

    //calculate the lengths of edges
    Eigen::MatrixXd l;
    igl::edge_lengths(V, F, l);

    //calculate cotangent matrix L
    Eigen::SparseMatrix<double> L;
    cotmatrix(l, F, L);
    //igl::cotmatrix(V, F, L);

    //calculate mass matrix M
    Eigen::DiagonalMatrix<double,Eigen::Dynamic> M;
    massmatrix(l, F, M);

    Eigen::SparseMatrix<double> A = Eigen::MatrixXd(M).sparseView() - lambda * L;

    //Cholesky decomposition
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    U = solver.solve(M * G);

}
