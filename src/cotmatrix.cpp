#include "cotmatrix.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>
void cotmatrix(
        const Eigen::MatrixXd & l,
        const Eigen::MatrixXi & F,
        Eigen::SparseMatrix<double> & L) {

    //get the number of vertices
    int n = F.maxCoeff() + 1;

    //set the size of sparse matrix L to n by n
    L.resize(n, n);

    //initialize a list of triplets
    std::vector<Eigen::Triplet<double>> tripletList;


    //iterate through all triangles
    for(int i = 0; i < F.rows(); i++) {

        //calculate cos() of each angle
        double cos_v0 = (l(i, 1) * l(i, 1) + l(i, 2) * l(i, 2) - l(i, 0) * l(i, 0))
                        / (2 * l(i, 1) * l(i, 2));
        double cos_v1 = (l(i, 0) * l(i, 0) + l(i, 2) * l(i, 2) - l(i, 1) * l(i, 1))
                        / (2 * l(i, 0) * l(i, 2));
        double cos_v2 = (l(i, 0) * l(i, 0) + l(i, 1) * l(i, 1) - l(i, 2) * l(i, 2))
                        / (2 * l(i, 0) * l(i, 1));

        //calculate cot() of each angle
        double cot_v0 = cos_v0 / sqrt(1 - cos_v0 * cos_v0);
        double cot_v1 = cos_v1 / sqrt(1 - cos_v1 * cos_v1);
        double cot_v2 = cos_v2 / sqrt(1 - cos_v2 * cos_v2);

        //set values
        tripletList.push_back(Eigen::Triplet<double>(F(i, 1),F(i, 2),0.5 * cot_v0));
        tripletList.push_back(Eigen::Triplet<double>(F(i, 2),F(i, 1),0.5 * cot_v0));
        tripletList.push_back(Eigen::Triplet<double>(F(i, 1),F(i, 1),-0.5 * cot_v0));
        tripletList.push_back(Eigen::Triplet<double>(F(i, 2),F(i, 2),-0.5 * cot_v0));

        tripletList.push_back(Eigen::Triplet<double>(F(i, 0),F(i, 2),0.5 * cot_v1));
        tripletList.push_back(Eigen::Triplet<double>(F(i, 2),F(i, 0),0.5 * cot_v1));
        tripletList.push_back(Eigen::Triplet<double>(F(i, 0),F(i, 0),-0.5 * cot_v1));
        tripletList.push_back(Eigen::Triplet<double>(F(i, 2),F(i, 2),-0.5 * cot_v1));

        tripletList.push_back(Eigen::Triplet<double>(F(i, 0),F(i, 1),0.5 * cot_v2));
        tripletList.push_back(Eigen::Triplet<double>(F(i, 1),F(i, 0),0.5 * cot_v2));
        tripletList.push_back(Eigen::Triplet<double>(F(i, 0),F(i, 0),-0.5 * cot_v2));
        tripletList.push_back(Eigen::Triplet<double>(F(i, 1),F(i, 1),-0.5 * cot_v2));


    }


    //use triplet list to initialize sparse matrix L
    L.setFromTriplets(tripletList.begin(), tripletList.end());


}

