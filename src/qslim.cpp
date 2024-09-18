#include "qslim.h"
#include "decimate.h"
#include "igl/decimate_callback_types.h"
#include "igl/edge_flaps.h"
#include "igl/boundary_loop.h"
#include "igl/per_vertex_point_to_plane_quadrics.h"

bool qslim(
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

    Eigen::VectorXi EMAP;
    Eigen::MatrixXi E,EF,EI;
    igl::edge_flaps(F,E,EMAP,EF,EI);

    std::vector<std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double>> quadrics;

    assert(V.cols() == 3);

    //igl::per_vertex_point_to_plane_quadrics(V,F,EMAP,EF,EI,quadrics);

    quadrics.resize(
            V.rows(),
            std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double>{Eigen::MatrixXd::Zero(3,3),Eigen::RowVectorXd::Zero(3),0}
            );

    const double w = 1e-10;
    for(int v = 0;v<V.rows();v++)
    {
        std::get<0>(quadrics[v]) = w * Eigen::MatrixXd::Identity(3,3);
        Eigen::RowVectorXd Vv = V.row(v);
        std::get<1>(quadrics[v]) = w*-Vv;
        std::get<2>(quadrics[v]) = w*Vv.dot(Vv);
    }

    for(int f = 0;f<F.rows();f++)
    {

        // Returns quadric triple {A,b,c} so that A-2*b+c measures the quadric
        const auto subspace_quadric = [](
                const Eigen::RowVectorXd & p,
                const Eigen::MatrixXd & S,
                const double  weight)->std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double>
        {
            // Dimension of subspace
            const int m = (int)S.rows();

            Eigen::MatrixXd A = Eigen::MatrixXd::Identity(3,3);;
            Eigen::RowVectorXd b = -p;
            double c = p.dot(p);

            for(int i = 0;i<m;i++)
            {
                Eigen::RowVectorXd ei = S.row(i);
                for(int j = 0;j<i;j++) assert(std::abs(S.row(j).dot(ei)) < 1e-10);
                A += -ei.transpose()*ei;
                b += p.dot(ei)*ei;
                c += -pow(p.dot(ei),2);
            }

            return std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double>{ weight*A, weight*b, weight*c };
        };

        // Finite (non-boundary) face
        Eigen::RowVectorXd v1 = V.row(F(f, 0));
        Eigen::RowVectorXd v2 = V.row(F(f, 1));
        Eigen::RowVectorXd v3 = V.row(F(f, 2));

        Eigen::RowVectorXd edge1 = v2 - v1;
        Eigen::RowVectorXd edge2 = v3 - v1;

        // Gram Determinant = squared area of parallelogram
        double area = sqrt(edge1.squaredNorm() * edge2.squaredNorm() - pow(edge2.dot(edge1), 2));

        Eigen::RowVectorXd e1 = edge1.normalized();
        Eigen::RowVectorXd e2 = (edge2 - e1.dot(edge2) * e1).normalized();
        Eigen::MatrixXd S(2,V.cols());
        S<<e1,e2;
        std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double> face_quadric = subspace_quadric(v1, S, area);

        //add face quadric on three corners
        for(int i = 0; i < 3; i++) {
            std::get<0>(quadrics[F(f,i)]) = std::get<0>(quadrics[F(f,i)]) + std::get<0>(face_quadric);
            std::get<1>(quadrics[F(f,i)]) = std::get<1>(quadrics[F(f,i)]) + std::get<1>(face_quadric);
            std::get<2>(quadrics[F(f,i)]) = std::get<2>(quadrics[F(f,i)]) + std::get<2>(face_quadric);
        }

        for(int i = 0; i < 3; i++) {

            if(EF(EMAP(f + F.rows() * i), 0) == -1 || EF(EMAP(f + F.rows() * i), 0) == -1) {

                int ev1 = (i + 1) % 3;
                int ev2 = (i + 2) % 3;


                Eigen::RowVectorXd edge_vector = V.row(F(f, ev2)) - V.row(F(f, ev1));
                double edge_length = edge_vector.norm();

                Eigen::RowVectorXd ev = edge_vector.normalized();

                const Eigen::RowVectorXd eu = V.row(F(f,i)) - V.row(F(f, ev1));

                Eigen::MatrixXd A(ev.size(),2);
                A<<ev.transpose(),eu.transpose();

                // Use QR decomposition to find basis for orthogonal space
                Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
                const Eigen::MatrixXd Q = qr.householderQ();
                const Eigen::MatrixXd N =
                        Q.topRightCorner(ev.size(),ev.size()-2).transpose();

                Eigen::MatrixXd S1(N.rows()+1,ev.size());
                S1<<ev,N;

                std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double> boundary_quadric = subspace_quadric(V.row(F(f, ev1)),S1,edge_length*edge_length);

                std::get<0>(quadrics[F(f,ev1)]) = std::get<0>(quadrics[F(f,ev1)]) + std::get<0>(boundary_quadric);
                std::get<1>(quadrics[F(f,ev1)]) = std::get<1>(quadrics[F(f,ev1)]) + std::get<1>(boundary_quadric);
                std::get<2>(quadrics[F(f,ev1)]) = std::get<2>(quadrics[F(f,ev1)]) + std::get<2>(boundary_quadric);

                std::get<0>(quadrics[F(f,ev2)]) = std::get<0>(quadrics[F(f,ev2)]) + std::get<0>(boundary_quadric);
                std::get<1>(quadrics[F(f,ev2)]) = std::get<1>(quadrics[F(f,ev2)]) + std::get<1>(boundary_quadric);
                std::get<2>(quadrics[F(f,ev2)]) = std::get<2>(quadrics[F(f,ev2)]) + std::get<2>(boundary_quadric);
            }
        }


    }


    int collapsed_v1 = -1;
    int collapsed_v2 = -1;

    //callback function to calculate cost and the placement position of an edge, marked by index e in E
    igl::decimate_cost_and_placement_callback cost_and_placement = [&quadrics](
            const int e,
            const Eigen::MatrixXd & V,
            const Eigen::MatrixXi & /*F*/,
            const Eigen::MatrixXi & E,
            const Eigen::VectorXi & /*EMAP*/,
            const Eigen::MatrixXi & EF,
            const Eigen::MatrixXi & /*EI*/,
            double & cost,
            Eigen::RowVectorXd & p) {

        // Combined quadric
        std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double> quadric_p;

        std::get<0>(quadric_p) = (std::get<0>(quadrics[E(e,0)]) + std::get<0>(quadrics[E(e,1)])).eval();
        std::get<1>(quadric_p) = (std::get<1>(quadrics[E(e,0)]) + std::get<1>(quadrics[E(e,1)])).eval();
        std::get<2>(quadric_p) = (std::get<2>(quadrics[E(e,0)]) + std::get<2>(quadrics[E(e,1)]));

        // Quadric: p'Ap + 2b'p + c
        // optimal point: Ap = -b, or rather because we have row vectors: pA=-b
        const auto & A = std::get<0>(quadric_p);
        const auto & b = std::get<1>(quadric_p);
        const auto & c = std::get<2>(quadric_p);
        if(b.array().isInf().any())
        {
            cost = std::numeric_limits<double>::infinity();
            p.resizeLike(b);
            p.setConstant(std::numeric_limits<double>::quiet_NaN());
        }else
        {
            p = -b*A.inverse();
            cost = p.dot(p*A) + 2*p.dot(b) + c;
        }

        if(std::isinf(cost) || cost!=cost)
        {
            cost = std::numeric_limits<double>::infinity();
            p.setConstant(0);
        }

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

    igl::decimate_pre_collapse_callback pre_collapse = [&collapsed_v1, &collapsed_v2](
            const Eigen::MatrixXd & /*V*/,
            const Eigen::MatrixXi & /*F*/,
            const Eigen::MatrixXi & E,
            const Eigen::VectorXi & /*EMAP*/,
            const Eigen::MatrixXi & /*EF*/,
            const Eigen::MatrixXi & /*EI*/,
            const igl::min_heap< std::tuple<double,int,int> > & /*Q*/,
            const Eigen::VectorXi & /*EQ*/,
            const Eigen::MatrixXd & /*C*/,
            const int e)->bool {
        collapsed_v1 = E(e,0);
        collapsed_v2 = E(e,1);

        return true;
    };

    igl::decimate_post_collapse_callback post_collapse = [&collapsed_v1, &collapsed_v2, &quadrics](
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
            const bool collapsed
    )->void {
        if(collapsed) {
            if(collapsed_v1 < collapsed_v2) {
                std::get<0>(quadrics[collapsed_v1]) = std::get<0>(quadrics[collapsed_v1]) + std::get<0>(quadrics[collapsed_v2]);
                std::get<1>(quadrics[collapsed_v1]) = std::get<1>(quadrics[collapsed_v1]) + std::get<1>(quadrics[collapsed_v2]);
                std::get<2>(quadrics[collapsed_v1]) = std::get<2>(quadrics[collapsed_v1]) + std::get<2>(quadrics[collapsed_v2]);
            } else {
                std::get<0>(quadrics[collapsed_v2]) = std::get<0>(quadrics[collapsed_v1]) + std::get<0>(quadrics[collapsed_v2]);
                std::get<1>(quadrics[collapsed_v2]) = std::get<1>(quadrics[collapsed_v1]) + std::get<1>(quadrics[collapsed_v2]);
                std::get<2>(quadrics[collapsed_v2]) = std::get<2>(quadrics[collapsed_v1]) + std::get<2>(quadrics[collapsed_v2]);
            }
        }

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
