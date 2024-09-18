#include "icp_single_iteration.h"
#include "random_points_on_mesh.h"
#include <Eigen/Core>
#include "point_mesh_distance.h"
#include "point_to_point_rigid_matching.h"
#include "point_to_plane_rigid_matching.h"
void icp_single_iteration(
        const Eigen::MatrixXd & VX,
        const Eigen::MatrixXi & FX,
        const Eigen::MatrixXd & VY,
        const Eigen::MatrixXi & FY,
        const int num_samples,
        const ICPMethod method,
        Eigen::Matrix3d & R,
        Eigen::RowVector3d & t) {


    //n by 3 matrix of sampled points
    Eigen::MatrixXd X;

    //n vector of distance
    Eigen::VectorXd D;

    //n by 3 matrix of the closest points on mesh
    Eigen::MatrixXd P;

    //n by 3 matrix of the normals of closest points
    Eigen::MatrixXd N;

    //random sampling
    random_points_on_mesh(num_samples, VX, FX, X);

    //calculate distance
    point_mesh_distance(X, VY, FY, D, P, N);

    if(method == ICP_METHOD_POINT_TO_POINT) {
        point_to_point_rigid_matching(X, P, R, t);
    } else {
        //igl::rigid_alignment(X, P, N, R, t);
        point_to_plane_rigid_matching(X, P, N, R, t);
    }
}
