#include "hausdorff_lower_bound.h"
#include <Eigen/Core>
#include "random_points_on_mesh.h"
#include "point_mesh_distance.h"
double hausdorff_lower_bound(
        const Eigen::MatrixXd & VX,
        const Eigen::MatrixXi & FX,
        const Eigen::MatrixXd & VY,
        const Eigen::MatrixXi & FY,
        const int n) {

    //n by 3 matrix of sampled points
    Eigen::MatrixXd X;

    //n vector of distance
    Eigen::VectorXd D;

    //n by 3 matrix of the closest points on mesh
    Eigen::MatrixXd P;

    //n by 3 matrix of the normals of closest points
    Eigen::MatrixXd N;

    //random sampling
    random_points_on_mesh(n, VX, FX, X);

    //calculate distance
    point_mesh_distance(X, VY, FY, D, P, N);

    //iterate through D to get the max distance, which is the hausdorff_lower_bound
    double maxDistance = 0;
    for(int i = 0; i < n; i++) {
        maxDistance = std::max(maxDistance, D(i));
    }

    return maxDistance;
}
