#include "vector_area_matrix.h"
#include "igl/boundary_loop.h"

void vector_area_matrix(
        const Eigen::MatrixXi & F,
        Eigen::SparseMatrix<double>& A) {

    int V_size = F.maxCoeff()+1;

    std::vector<std::vector<int>> loops;
    std::vector<std::vector<int>> edges;
    igl::boundary_loop(F, loops);
    for(const auto& loop : loops) {
        int n = (int)loop.size();
        for(int i = 0; i < n; i++) {
            edges.push_back({loop[i], loop[(i + 1) % n]});
        }
    }
    assert(edges[0].size() == 2);

    Eigen::MatrixXi boundary_edges;
    boundary_edges.resize((int)edges.size(), 2);
    for(int i = 0; i < (int)edges.size(); i++) {
        boundary_edges(i, 0) = edges[i][0];
        boundary_edges(i, 1) = edges[i][1];
    }

    std::vector<Eigen::Triplet<double>> triplets;

    for(int e = 0; e < boundary_edges.rows(); e++) {

        const int i = boundary_edges(e, 0);
        const int j = boundary_edges(e, 1);

        triplets.emplace_back(i + V_size, j, -0.25);
        triplets.emplace_back(j, i + V_size, -0.25);
        triplets.emplace_back(i, j + V_size, 0.25);
        triplets.emplace_back(j + V_size, i, 0.25);
    }


    A.resize(V_size * 2,V_size * 2);
    A.setFromTriplets(triplets.begin(), triplets.end());
}

