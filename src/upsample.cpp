#include "upsample.h"
#include <vector>
#include <igl/unique_edge_map.h>

void upsample(
        const int n_verts,
        const Eigen::MatrixXi & F,
        Eigen::SparseMatrix<double> & S,
        Eigen::MatrixXi & newF) {

    //use igl::unique_edge_map to calculate the list of undirected edges and the map from vertices to edges
    Eigen::MatrixXi E;
    Eigen::MatrixXi uE;
    Eigen::VectorXi EMAP;
    std::vector<std::vector<int>> uE2E;
    igl::unique_edge_map(F, E, uE,  EMAP, uE2E);

    //use triplets to initialize sparse matrix
    std::vector<Eigen::Triplet<double>> tripletList;

    //we have already known the number of non-zero elements
    tripletList.reserve(n_verts + uE.rows() * 2);

    //the old vertices' position
    for(int i = 0; i < n_verts; i++) {
        tripletList.emplace_back(i, i, 1.0);
    }

    //the new vertices' position
    for(int i = 0; i < uE.rows(); i++) {
        tripletList.emplace_back(i + n_verts, uE(i, 0), 0.5);
        tripletList.emplace_back(i + n_verts, uE(i, 1), 0.5);
    }

    //set S using triplets
    S.resize(n_verts + uE.rows(), n_verts);
    S.setFromTriplets(tripletList.begin(), tripletList.end());

    //resize the new faces matrix to 4 times of the original
    newF.resize(F.rows() * 4, 3);

    //for each old triangle, fill 4 new triangles in the new faces matrix
    for(int i = 0; i < F.rows(); i++) {

        //calculate the indexes of 6 vertices in new vertices matrix
        int oldVertex_0 = F(i, 0);
        int oldVertex_1 = F(i, 1);
        int oldVertex_2 = F(i, 2);
        int newVertex_0 = n_verts + EMAP(i + F.rows() * 0);
        int newVertex_1 = n_verts + EMAP(i + F.rows() * 1);
        int newVertex_2 = n_verts + EMAP(i + F.rows() * 2);

        //fill newF
        newF.row(i * 4 + 0) << oldVertex_0, newVertex_2, newVertex_1;
        newF.row(i * 4 + 1) << newVertex_0, newVertex_2, newVertex_1;
        newF.row(i * 4 + 2) << newVertex_2, newVertex_0, oldVertex_1;
        newF.row(i * 4 + 3) << newVertex_1, oldVertex_2, newVertex_0;
    }
}

