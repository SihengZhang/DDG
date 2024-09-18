#include "loop.h"
#include <vector>
#include <igl/unique_edge_map.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/adjacency_list.h>

void loop(
        const int n_verts,
        const Eigen::MatrixXi & F,
        Eigen::SparseMatrix<double> & S,
        Eigen::MatrixXi& newF) {

    //use igl::unique_edge_map to calculate the list of undirected edges and the map from vertices to edges
    Eigen::MatrixXi E;
    Eigen::MatrixXi uE;
    Eigen::VectorXi EMAP;
    std::vector<std::vector<int>> uE2E;
    igl::unique_edge_map(F, E, uE,  EMAP, uE2E);

    //get the list of lists of adjacency vertices of each vertex in F
    std::vector<std::vector<int>> A;
    igl::adjacency_list(F, A);
    assert(A.size() == n_verts);

    //construct the triangle-triangle adjacency matrix for a given mesh F.
    Eigen::MatrixXi TT;
    Eigen::MatrixXi TTi;
    igl::triangle_triangle_adjacency(F, TT, TTi);

    //use triplets to initialize sparse matrix
    std::vector<Eigen::Triplet<double>> tripletList;

    //we have already known the number of non-zero elements
    tripletList.reserve(n_verts * 7 + uE.rows() * 4);

    //the old vertices' position
    for(int i = 0; i < A.size(); i++) {
        //the number of adjacent vertices
        int n = (int)A[i].size();


        if(n == 2) { //the vertex is on the boundary
            tripletList.emplace_back(i, A[i][0], 1.0 / 8.0);
            tripletList.emplace_back(i, A[i][1], 1.0 / 8.0);
            tripletList.emplace_back(i, i, 3.0 / 4.0);
        } else { // inside the mesh

            //determine the value of beta
            double beta;
            if(n == 3)
                beta = 3.0 / 16.0;
            else
                beta = 3.0 / (8.0 * n);

            //iterate all adjacent vertices
            for(int j = 0; j < n; j++) {
                tripletList.emplace_back(i, A[i][j], beta);
            }

            //the vertex itself part
            tripletList.emplace_back(i, i, 1.0 - beta * n);
        }
    }

    //the new vertices' position
    for(int i = 0; i < uE.rows(); i++) {
        if(uE2E[i].size() == 1) { //on the boundary
            tripletList.emplace_back(i + n_verts, uE(i, 0), 0.5);
            tripletList.emplace_back(i + n_verts, uE(i, 1), 0.5);

        } else { //inside the mesh

            //near adjacent vertices
            tripletList.emplace_back(i + n_verts, uE(i, 0), 3.0 / 8.0);
            tripletList.emplace_back(i + n_verts, uE(i, 1), 3.0 / 8.0);

            //calculate the indices of directed edges
            int e1 = uE2E[i][0];
            int e2 = uE2E[i][1];

            //calculate the indices of faces
            int f1 = e1 % (int)F.rows();
            int f2 = e2 % (int)F.rows();

            //calculate the indices of vertices in a particular face
            int c1 = e1 / (int)F.rows();
            int c2 = e2 / (int)F.rows();

            //far adjacent vertices
            tripletList.emplace_back(i + n_verts, F(f1, c1), 1.0 / 8.0);
            tripletList.emplace_back(i + n_verts, F(f2, c2), 1.0 / 8.0);

        }
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
