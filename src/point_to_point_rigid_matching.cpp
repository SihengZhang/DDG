#include "point_to_point_rigid_matching.h"
#include "closest_rotation.h"
#include <Eigen/Core>
void point_to_point_rigid_matching(
        const Eigen::MatrixXd & X,
        const Eigen::MatrixXd & P,
        Eigen::Matrix3d & R,
        Eigen::RowVector3d & t) {

    Eigen::Vector3d xAverage;
    Eigen::Vector3d pAverage;

    for(int i = 0; i < X.rows(); i++) {
        xAverage += X.row(i);
        pAverage += P.row(i);
    }
    xAverage = xAverage / X.rows();
    pAverage = pAverage / X.rows();

    Eigen::MatrixXd XA = X - Eigen::VectorXd::Ones(X.rows()) * xAverage.transpose();
    Eigen::MatrixXd PA = P - Eigen::VectorXd::Ones(X.rows()) * pAverage.transpose();

    Eigen::MatrixXd M = PA.transpose() * XA;

    closest_rotation(M.transpose(), R);

    t = pAverage - R * xAverage;
    t = t.transpose();

}

