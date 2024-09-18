#include "point_mesh_distance.h"
#include <Eigen/Core>
#include <point_triangle_distance.h>
#include <igl/per_face_normals.h>
void point_mesh_distance(
        const Eigen::MatrixXd & X,
        const Eigen::MatrixXd & VY,
        const Eigen::MatrixXi & FY,
        Eigen::VectorXd & D,
        Eigen::MatrixXd & P,
        Eigen::MatrixXd & N) {

    //the number of points
    int n = (int)X.rows();

    //the number of triangles in the mesh
    int m = (int)FY.rows();

    //resize output vectors and matrices
    D.resize(n);
    P.resize(n, 3);
    N.resize(n, 3);

    //compute all normals of the triangles in the mesh
    Eigen::MatrixXd meshNormals;
    igl::per_face_normals(VY, FY, meshNormals);

    //iterate all sampled points in X
    for(int i = 0; i < n; i++) {

        double minDistance;
        Eigen::RowVector3d minPoint;
        Eigen::RowVector3d minNormal = meshNormals.row(0);

        //initialize distance
        point_triangle_distance(X.row(i), VY.row(FY(0, 0)), VY.row(FY(0, 1)), VY.row(FY(0, 2)), minDistance, minPoint);

        //iterate all triangles in the mesh to get the minimum distance
        for(int j = 0; j < m; j++) {

            //get minimum distance and point from a single triangle
            double distance;
            Eigen::RowVector3d point;
            point_triangle_distance(X.row(i), VY.row(FY(j, 0)), VY.row(FY(j, 1)), VY.row(FY(j, 2)), distance, point);

            //if found new minimum, update 3 "min" results
            if(distance < minDistance) {
                minDistance = distance;
                minPoint = point;
                minNormal = meshNormals.row(j);
            }
        }

        //store the result for one iteration
        D(i) = minDistance;
        P.row(i) = minPoint;
        N.row(i) = minNormal;
    }

}
