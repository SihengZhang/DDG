#include "euler_characteristic.h"

int euler_characteristic(const Eigen::MatrixXi &F) {
    int Chi = 0;
    // ADD YOUR CODE HERE
    //USE SET TO AVOID DUPLICATE VERTICES
    set<int> verticesSet;

    //ITERATE ALL VERTICES IN F
    for(int i = 0; i < F.rows(); i++) {
        for(int j = 0; j < 3; j++) {
            verticesSet.insert(F(i, j));
        }
    }

    //GET THE NUMBERS OF VERTICES, FACES AND EDGES.
    int Vertices = verticesSet.size();

    int Faces = F.rows();

    int Edges = edges(F).rows();

    //CALCULATE THE EULER CHARACTERISTIC
    Chi = Vertices - Edges + Faces;

    return Chi;
}
