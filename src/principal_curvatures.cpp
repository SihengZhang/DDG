#include "../include/principal_curvatures.h"
#include <Eigen/Eigenvalues>
#include "igl/adjacency_list.h"
#include "igl/pinv.h"
#include <cmath>
#include <vector>
#include <set>
#include <iostream>

void principal_curvatures(
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        Eigen::MatrixXd & D1,
        Eigen::MatrixXd & D2,
        Eigen::VectorXd & K1,
        Eigen::VectorXd & K2) {

    //resize the output vectors and matrices
    K1 = Eigen::VectorXd::Zero(V.rows());
    K2 = Eigen::VectorXd::Zero(V.rows());
    D1 = Eigen::MatrixXd::Zero(V.rows(),3);
    D2 = Eigen::MatrixXd::Zero(V.rows(),3);

    //get adjacency list of all vertices
    std::vector<std::vector<int>> AL;
    igl::adjacency_list(F, AL);

    //for all vertices in V
    for(int i = 0; i < V.rows(); i++) {

        //get the unique set of two-ring vertices
        std::set<int> unique_neighbor_vertices;
        for(const auto v1 : AL[i]) {
            unique_neighbor_vertices.insert(v1);
            for(const auto v2 : AL[v1]) {
                if(v2 != i) unique_neighbor_vertices.insert(v2);
            }
        }

        //get relative position matrix P
        Eigen::MatrixXd P;
        P.resize((int)unique_neighbor_vertices.size(), 3);
        int index = 0;
        for(const auto v : unique_neighbor_vertices) {
            P(index, 0) = V(v, 0) - V(i, 0);
            P(index, 1) = V(v, 1) - V(i, 1);
            P(index, 2) = V(v, 2) - V(i, 2);
            index++;
        }
        const int k = (int)P.rows();


        Eigen::MatrixXd S;
        Eigen::VectorXd B;
        S.resize(k, 2);
        B.resize(k);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(P.transpose() * P);
        Eigen::MatrixXd Eigen_vectors = solver.eigenvectors();
        for(int p = 0; p < k; p++) {
            S(p, 0) = P.row(p).dot(Eigen_vectors.row(2));
            S(p, 1) = P.row(p).dot(Eigen_vectors.row(1));
            B(p) = P.row(p).dot(Eigen_vectors.row(0));
        }

        Eigen::MatrixXd A;
        Eigen::MatrixXd APlus;
        A.resize(k, 5);
        for(int p = 0; p < k; p++) {
            A(p, 0) = S(p, 0);
            A(p, 1) = S(p, 1);
            A(p, 2) = S(p, 0) * S(p, 0);
            A(p, 3) = S(p, 0) * S(p, 1);
            A(p, 4) = S(p, 1) * S(p, 1);
        }
        igl::pinv(A, APlus);

        Eigen::VectorXd alpha = APlus * B;

        double dE = 1 + alpha(0) * alpha(0);
        double dF = alpha(0) * alpha(1);
        double dG = 1 + alpha(1) * alpha(1);
        double de = 2 * alpha(2) / std::sqrt(alpha(0) * alpha(0) + 1 + alpha(1) * alpha(1));
        double df = alpha(3) / std::sqrt(alpha(0) * alpha(0) + 1 + alpha(1) * alpha(1));
        double dg = 2 * alpha(4) / std::sqrt(alpha(0) * alpha(0) + 1 + alpha(1) * alpha(1));

        Eigen::MatrixXd Shape;
        Eigen::MatrixXd matrix1;
        Eigen::MatrixXd matrix2;
        matrix1.resize(2, 2);
        matrix2.resize(2, 2);
        matrix1 << de, df,
                   df, dg;
        matrix2 << dE, dF,
                   dF, dG;
        Shape = -1 * matrix1 * matrix2.inverse();

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_for_shape(Shape);
        Eigen::VectorXd principal_curvatures = solver_for_shape.eigenvalues();
        Eigen::MatrixXd tangent_vectors = solver_for_shape.eigenvectors();

        K1(i) = principal_curvatures(1);
        K2(i) = principal_curvatures(0);
        D1.row(i) = V.row(i) + tangent_vectors(1, 0) * Eigen_vectors.row(2) + tangent_vectors(1, 1) * Eigen_vectors.row(1);
        D2.row(i) = V.row(i) + tangent_vectors(0, 0) * Eigen_vectors.row(2) + tangent_vectors(0, 1) * Eigen_vectors.row(1);
    }


}
