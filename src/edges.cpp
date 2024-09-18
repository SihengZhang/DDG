#include "edges.h"

Eigen::MatrixXi edges(const Eigen::MatrixXi &F) {
    Eigen::MatrixXi E;
    // ADD YOUR CODE HERE

    // USE A SET TO ELIMINATE DUPLICATE EDGES
    set<pair<int, int>> edgesSet;
    // ITERATE THROUGH MATRIX F BY ROWS
    for(int i = 0; i < F.rows(); i++) {
        // INSERT ALL 3 EDGES IN THE TRIANGLE
        // MAKE SURE THE FIRST VERTEX HAS SMALLER INDEX THAN THE SECOND TO AVOID DUPLICATE EDGES
        if(F(i, 0) < F(i, 1))
            edgesSet.insert(make_pair(F(i, 0), F(i, 1)));
        else
            edgesSet.insert(make_pair(F(i, 1), F(i, 0)));

        if(F(i, 0) < F(i, 2))
            edgesSet.insert(make_pair(F(i, 0), F(i, 2)));
        else
            edgesSet.insert(make_pair(F(i, 2), F(i, 0)));

        if(F(i, 1) < F(i, 2))
            edgesSet.insert(make_pair(F(i, 1), F(i, 2)));
        else
            edgesSet.insert(make_pair(F(i, 2), F(i, 1)));
    }

    // RESIZE MATRIX E TO OCCUPY ALL EDGES
    E.resize(edgesSet.size(), 2);

    // COPY EDGES FROM THE SET TO MATRIX E
    int index = 0;
    for(auto edge : edgesSet) {
        E(index, 0) = edge.first;
        E(index, 1) = edge.second;
        index++;
    }

    return E;
}
