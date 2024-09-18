#include "tutte.h"
#include "igl/boundary_loop.h"
#include "igl/map_vertices_to_circle.h"
#include "igl/min_quad_with_fixed.h"

void tutte(
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        Eigen::MatrixXd & U) {

    std::vector<std::vector<int>> Loops;
    Eigen::VectorXi b;
    igl::boundary_loop(F, Loops);
    assert(Loops.size() > 0);
    b = Eigen::Map<Eigen::VectorXi>(Loops[0].data(), (int)Loops[0].size());

    Eigen::MatrixXd UV;
    igl::map_vertices_to_circle(V, b, UV);

    Eigen::SparseMatrix<double> L;
    L.resize(V.rows(), V.rows());

    std::vector<Eigen::Triplet<double>> triplets;

    Eigen::MatrixXi edges;
    edges.resize(3, 2);
    edges << 1,2,
             2,0,
             0,1;

    for(int f = 0; f < F.rows(); f++) {
        for(int e = 0; e < edges.rows(); e++) {

            const int i = F(f, edges(e, 0));
            const int j = F(f, edges(e, 1));

            const double length = (V.row(i) - V.row(j)).norm();

            triplets.emplace_back(i, j, 0.5 * length);
            triplets.emplace_back(j, i, 0.5 * length);
            triplets.emplace_back(i, i, -0.5 * length);
            triplets.emplace_back(j, j, -0.5 * length);
        }
    }

    L.setFromTriplets(triplets.begin(), triplets.end());

    igl::min_quad_with_fixed_data<double> data;
    igl::min_quad_with_fixed_precompute(L,b,Eigen::SparseMatrix<double>(),false,data);
    igl::min_quad_with_fixed_solve(data, Eigen::MatrixXd::Zero(data.n,2), UV, Eigen::VectorXd(), U);

}

