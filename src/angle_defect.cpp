#include "../include/angle_defect.h"
#include "../include/internal_angles.h"
#include "igl/squared_edge_lengths.h"
#include "igl/adjacency_matrix.h"
#include "igl/vertex_triangle_adjacency.h"
#include <vector>
#include <cmath>


void angle_defect(
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        Eigen::VectorXd & D) {

    //number of vertices
    const int n = (int)V.rows();

    //get squared length of edges
    Eigen::MatrixXd l_sqr;
    igl::squared_edge_lengths(V, F, l_sqr);

    //get internal angles incident on respective corner of all triangles
    Eigen::MatrixXd A;
    internal_angles(l_sqr, A);

    //get list of adjacency triangles indices and list of corresponding corner indices
    std::vector<std::vector<int>> VF;
    std::vector<std::vector<int>> VFi;
    igl::vertex_triangle_adjacency(n, F, VF, VFi);

    //resize D
    D.resize(n);

    //set D
    for(int i = 0; i < n; i++) {

        //default value is 2 * pi
        double angleDefect = 2.0 * M_PI;

        //minus all surrounding angles
        for(int j = 0; j < VF[i].size(); j++) {
            angleDefect -= A(VF[i][j], VFi[i][j]);
        }

        D(i) = angleDefect;
    }
}
