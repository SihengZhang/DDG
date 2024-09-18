#include "../include/internal_angles.h"
#include <cmath>

void internal_angles(
        const Eigen::MatrixXd & l_sqr,
        Eigen::MatrixXd & A) {

    const int m = (int)l_sqr.rows();

    A.resize(m, 3);

    for(int i = 0; i < m; i++) {

        const double se0 = l_sqr(i, 0);
        const double se1 = l_sqr(i, 1);
        const double se2 = l_sqr(i, 2);

        A(i, 0) = std::acos((se1 + se2 - se0) / (2 * std::sqrt(se1 * se2)));
        A(i, 1) = std::acos((se0 + se2 - se1) / (2 * std::sqrt(se0 * se2)));
        A(i, 2) = std::acos((se0 + se1 - se2) / (2 * std::sqrt(se0 * se1)));
    }

}
