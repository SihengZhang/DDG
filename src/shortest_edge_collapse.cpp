#include "shortest_edge_collapse.h"
#include "decimate.h"
#include "igl/decimate_callback_types.h"
#include "igl/edge_flaps.h"
#include "igl/boundary_loop.h"

bool shortest_edge_collapse(
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        const int max_m,
        Eigen::MatrixXd & U,
        Eigen::MatrixXi & G,
        Eigen::VectorXi & J,
        Eigen::VectorXi & I) {

    //the number of faces at the beginning
    const int start_m = (int)F.rows();

    //track the current number of faces
    int m = start_m;

    //callback function to calculate cost and the placement position of an edge, marked by index e in E
    igl::decimate_cost_and_placement_callback cost_and_placement = [](
            const int e,
            const Eigen::MatrixXd & V,
            const Eigen::MatrixXi & /*F*/,
            const Eigen::MatrixXi & E,
            const Eigen::VectorXi & /*EMAP*/,
            const Eigen::MatrixXi & EF,
            const Eigen::MatrixXi & /*EI*/,
            double & cost,
            Eigen::RowVectorXd & p) {

        //cost is the length of the edge
        cost = (V.row(E(e,0))-V.row(E(e,1))).norm();

        //new position of vertex is the midpoint of the edge
        p = (V.row(E(e,0))+V.row(E(e,1))) * 0.5;

    };

    igl::decimate_stopping_condition_callback stopping_condition = [&m, max_m](
            const Eigen::MatrixXd & /*V*/,
            const Eigen::MatrixXi & /*F*/,
            const Eigen::MatrixXi & /*E*/,
            const Eigen::VectorXi & /*EMAP*/,
            const Eigen::MatrixXi & /*EF*/,
            const Eigen::MatrixXi & /*EI*/,
            const igl::min_heap< std::tuple<double,int,int> > & /*Q*/,
            const Eigen::VectorXi & /*EQ*/,
            const Eigen::MatrixXd & /*C*/,
            const int /*e*/,
            const int e1,
            const int e2,
            const int /*f1*/,
            const int /*f2*/)->bool {

        //reduce the number of faces after decimate
        if(e1 != -1 && e2 != -1)
            m -= 2;
        else if(e1 != -1 || e2 != -1)
            m -= 1;

        //if current number of faces smaller than max faces, stop
        return m <= max_m;
    };

    igl::decimate_pre_collapse_callback pre_collapse = [](
            const Eigen::MatrixXd & /*V*/,
            const Eigen::MatrixXi & /*F*/,
            const Eigen::MatrixXi & /*E*/,
            const Eigen::VectorXi & /*EMAP*/,
            const Eigen::MatrixXi & /*EF*/,
            const Eigen::MatrixXi & /*EI*/,
            const igl::min_heap< std::tuple<double,int,int> > & /*Q*/,
            const Eigen::VectorXi & /*EQ*/,
            const Eigen::MatrixXd & /*C*/,
            const int e)->bool {

        return true;
    };

    igl::decimate_post_collapse_callback post_collapse = [](
            const Eigen::MatrixXd & /*V*/,
            const Eigen::MatrixXi & /*F*/,
            const Eigen::MatrixXi & /*E*/,
            const Eigen::VectorXi & /*EMAP*/,
            const Eigen::MatrixXi & /*EF*/,
            const Eigen::MatrixXi & /*EI*/,
            const igl::min_heap< std::tuple<double,int,int> > & /*Q*/,
            const Eigen::VectorXi & /*EQ*/,
            const Eigen::MatrixXd & /*C*/,
            const int /*e*/,
            const int /*e1*/,
            const int /*e2*/,
            const int /*f1*/,
            const int /*f2*/,
            const bool /*collapsed*/
    )->void {

    };

    bool result = decimate(
            V, F,
            cost_and_placement,
            stopping_condition,
            pre_collapse,
            post_collapse,
            U, G, J, I);

    return result;

}
