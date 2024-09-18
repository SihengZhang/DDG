#include "fd_interpolate.h"

void fd_interpolate(
        const int nx,
        const int ny,
        const int nz,
        const double h,
        const Eigen::RowVector3d & corner,
        const Eigen::MatrixXd & P,
        Eigen::SparseMatrix<double> & W) {

    //resize W to n by (nx*ny*nz), n is the rows of P
    W.resize(P.rows(), nx * ny * nz);

    //make a triplet to store none-zero elements in W
    std::vector<Eigen::Triplet<double>> tripletList;

    //For all the points in P, do trilinear interpolation
    for (int i = 0; i < P.rows(); i++) {

        //Calculate the relative position in our grid
        Eigen::Vector3d relativePosition = (P.row(i) - corner) / h;

        //The base index of vertices in our grid
        int xIndex = static_cast<int> (relativePosition[0]);
        int yIndex = static_cast<int> (relativePosition[1]);
        int zIndex = static_cast<int> (relativePosition[2]);

        //The shift value of three directions from base, [0, 1)
        double xShift = relativePosition[0] - xIndex;
        double yShift = relativePosition[1] - yIndex;
        double zShift = relativePosition[2] - zIndex;

        //Calculate the weights of all 8 adjacent vertices(this block comes from ChatGPT :) )
        for (int dz = 0; dz < 2; dz++) {
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int index = (xIndex + dx) + nx * ((yIndex + dy) + ny * (zIndex + dz));
                    double weight = (dx == 0 ? (1.0 - xShift) : xShift)
                            * (dy == 0 ? (1.0 - yShift) : yShift)
                            * (dz == 0 ? (1.0 - zShift) : zShift);
                    //insert the weight in triplets
                    tripletList.push_back(Eigen::Triplet<double>(i, index, weight));
                }
            }
        }
    }

    W.setFromTriplets(tripletList.begin(), tripletList.end());
}