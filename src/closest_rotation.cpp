#include "closest_rotation.h"
#include <Eigen/Core>
#include <Eigen/Dense>
void closest_rotation(
        const Eigen::Matrix3d & M,
        Eigen::Matrix3d & R) {

    // compute the SVD of M, using Eigen::JacobiSVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Check for reflection
    if ((U * V.transpose()).determinant() < 0.0) {
        // Adjust the last column of U if the determinant of UV^T is -1
        U.col(2) *= -1.0;
    }

    // Compute the closest rotation matrix
    R = U * V.transpose();
}
