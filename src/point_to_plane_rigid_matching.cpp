#include "point_to_plane_rigid_matching.h"
#include <Eigen/Core>
#include <igl/iterative_closest_point.h>
#include <igl/rigid_alignment.h>
#include <Eigen/Dense>
void point_to_plane_rigid_matching(
        const Eigen::MatrixXd & X,
        const Eigen::MatrixXd & P,
        const Eigen::MatrixXd & N,
        Eigen::Matrix3d & R,
        Eigen::RowVector3d & t) {

    igl::rigid_alignment(X, P, N, R, t);

    const int n = (int)X.rows();

    //initial value of R and t, use identity matrix to keep R as a rotation matrix
    R = Eigen::Matrix3d::Identity(3,3);
    t = Eigen::RowVector3d::Zero(1,3);
    Eigen::MatrixXd X_iter = X;

    //iterate to get optimal R and t
    const int iterations = 5;
    for(int i = 0; i < iterations; i++) {

        //calculate temp1
        Eigen::MatrixXd temp1(n, 6);
        for(int j = 0; j < n; j++) {
            Eigen::Vector3d xi(X_iter(j, 0), X_iter(j, 1), X_iter(j, 2));
            Eigen::Vector3d ni(N(j, 0), N(j, 1), N(j, 2));
            Eigen::Vector3d ci = xi.cross(ni);
            temp1.row(j) << ci(0),ci(1), ci(2), ni(0), ni(1), ni(2);
        }

        //calculate temp2
        Eigen::VectorXd temp2;
        temp2.resize(n);
        for(int j = 0; j < n; j++) {
            Eigen::Vector3d xi(X_iter(j, 0), X_iter(j, 1), X_iter(j, 2));
            Eigen::Vector3d pi(P(j, 0), P(j, 1), P(j, 2));
            Eigen::Vector3d ni(N(j, 0), N(j, 1), N(j, 2));
            temp2(j) = ni.dot(pi - xi);
        }

        Eigen::MatrixXd A = temp1.transpose() * temp1;
        Eigen::MatrixXd B = temp1.transpose() * temp2;

        //solve u, u.transpose() = [a.transpose() t.transpose()]
        Eigen::VectorXd u = (A.ldlt()).solve(B);

        //use u to get rotation matrix and shift vector
        Eigen::RowVector3d t_iter = u.tail(3).transpose();

        Eigen::Matrix3d W;
        W<<0, u(2), -u(1),
           -u(2), 0, u(0),
           u(1), -u(0), 0;

        Eigen::Matrix3d R_iter;

        const double x = u.head(3).stableNorm();
        if(x == 0)
        {
            R_iter = Eigen::Matrix3d::Identity(3,3);
        }else
        {
            R_iter =Eigen::Matrix3d::Identity(3,3) +
                    sin(x)/x * W +
                    (1.0-cos(x))/(x*x) * W * W;
        }

        //update R, t and pre-move X for next iteration
        R = R * R_iter;
        t = t * R_iter + t_iter;
        X_iter = (X * R).rowwise() + t;

    }
}
