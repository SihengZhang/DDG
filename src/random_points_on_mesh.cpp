#include "random_points_on_mesh.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>

void random_points_on_mesh(
        const int n,
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        Eigen::MatrixXd & X) {

    //the number of triangles on mesh
    int m = (int)F.rows();

    //build prefix sum vector for binary searching
    std::vector<double> prefixSum;

    //track the sum of triangles
    double sum = 0;

    //build prefix sum vector
    for(int i = 0; i < m; i++) {
        //calculate 2 edge vectors
        Eigen::Vector3d edge1 = V.row(F(i, 1)) - V.row(F(i, 0));
        Eigen::Vector3d edge2 = V.row(F(i, 2)) - V.row(F(i, 0));

        //use edge vectors to calculate area
        sum += 0.5 * edge1.cross(edge2).norm();

        prefixSum.push_back(sum);
    }

    //set the size of target output to n by 3, which means n random points' coordinates on the mesh
    X.resize(n,3);

    //iterate n times to generate n random points
    for(int i = 0; i < n; i++) {

        //c++ 11 random device
        std::random_device rd;
        std::mt19937 gen(rd());

        //generate double random number from 0 to the total area
        std::uniform_real_distribution<double> dis(0.0, sum);
        double randomArea = dis(gen);

        //binary search to get the index of target triangle
        auto targetTriangle = std::lower_bound(prefixSum.begin(), prefixSum.end(), randomArea);
        int index =  (int)std::distance(prefixSum.begin(), targetTriangle);

        //get three vertices of triangle
        Eigen::Vector3d v1 = V.row(F(index, 0));
        Eigen::Vector3d v2 = V.row(F(index, 1));
        Eigen::Vector3d v3 = V.row(F(index, 2));

        //generate 2 double random numbers from 0 to 1
        std::uniform_real_distribution<double> disZeroToOne(0.0, 1.0);
        double a1 = disZeroToOne(gen);
        double a2 = disZeroToOne(gen);

        //if a1 + a2 > 1, the point is out of the triangle, so, we keep a1 + a2 < 1
        if(a1 + a2 > 1.0) {
            a1 = 1 - a1;
            a2 = 1 - a2;
        }

        //get the absolute coordinate vector of the random point
        Eigen::Vector3d randomPoint = v1 + a1 * (v2 - v1) + a2 * (v3 - v1);

        //save the point to X
        X.row(i) = randomPoint;

    }



}

