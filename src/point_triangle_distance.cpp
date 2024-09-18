#include "point_triangle_distance.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>

void point_triangle_distance(
        const Eigen::RowVector3d & x,
        const Eigen::RowVector3d & a,
        const Eigen::RowVector3d & b,
        const Eigen::RowVector3d & c,
        double & d,
        Eigen::RowVector3d & p) {

    //get vectors of 3 edges
    Eigen::RowVector3d AB = b - a;
    Eigen::RowVector3d BC = c - b;
    Eigen::RowVector3d CA = a - c;

    // calculate normal of the triangle
    Eigen::RowVector3d n = AB.cross(CA).normalized();

    // calculate normals of 3 edges
    Eigen::RowVector3d nAB = AB.cross(n).normalized();
    Eigen::RowVector3d nBC = BC.cross(n).normalized();
    Eigen::RowVector3d nCA = CA.cross(n).normalized();

    //get x's projection on the plane of triangle ABC
    Eigen::RowVector3d xProj = x - n.dot(x - a) * n;

    // Check if xProj is inside the triangle
    if ((nAB.dot(xProj - a) >= 0) && (nBC.dot(xProj - b) >= 0) && (nCA.dot(xProj - c) >= 0)) {
        //if xProj is in the triangle, p is xProj itself
        p = xProj;
        d = (x - p).norm();
        return;
    }

    // build a lambda function to compute closest point on edges
    auto closestPointOnLine = [](const Eigen::RowVector3d& v1, const Eigen::RowVector3d& v2, const Eigen::RowVector3d& p) {
        //get the edge vector
        Eigen::RowVector3d edgeVector = v2 - v1;

        //project p - v1 on the edge
        double t = edgeVector.dot(p - v1) / edgeVector.dot(edgeVector);

        //make sure the point is on the edge, including vertices
        t = std::max(0.0, std::min(1.0, t));

        Eigen::RowVector3d closestP = v1 + t * edgeVector;
        return closestP;
    };

    //get the closest points on each edges
    Eigen::RowVector3d pAB = closestPointOnLine(a, b, x);
    Eigen::RowVector3d pBC = closestPointOnLine(b, c, x);
    Eigen::RowVector3d pCA = closestPointOnLine(c, a, x);

    // Choose the closest point among pAB, pBC, pCA
    double distAB = (pAB - x).norm();
    double distBC = (pBC - x).norm();
    double distCA = (pCA - x).norm();


    if (distAB < distBC && distAB < distCA) {
        //p is on edge AB
        p = pAB;
        d = (x - p).norm();
        return;

    } else if (distBC < distCA) {
        //p is on corner B or on edge BC
        p = pBC;
        d = (x - p).norm();
        return;

    } else {
        //p is on corner A, C or on edge CA
        p = pCA;
        d = (x - p).norm();
        return;
    }

}