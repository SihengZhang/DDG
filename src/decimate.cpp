#include "decimate.h"
#include <tuple>
#include <queue>
#include <vector>
#include <set>
#include "igl/edge_flaps.h"
#include "igl/remove_unreferenced.h"
#include "igl/edge_collapse_is_valid.h"
#include "igl/is_edge_manifold.h"
#include "igl/collapse_edge.h"

void get_neighbors(const int & e,
                   const Eigen::MatrixXi & F,
                   const Eigen::MatrixXi & uE,
                   const Eigen::VectorXi & EMAP,
                   const Eigen::MatrixXi & EF,
                   const Eigen::MatrixXi & EI,
                   std::vector<int> & Nv,
                   std::vector<int> & Nf,
                   std::vector<int> & Nsv,
                   std::vector<int> & Nsf,
                   std::vector<int> & Ndv,
                   std::vector<int> & Ndf);

bool collapse_one_edge(
        const igl::decimate_cost_and_placement_callback & cost_and_placement,
        const igl::decimate_pre_collapse_callback       & pre_collapse,
        const igl::decimate_post_collapse_callback      & post_collapse,
        Eigen::MatrixXd & V,
        Eigen::MatrixXi & F,
        Eigen::MatrixXi & E,
        Eigen::VectorXi & EMAP,
        Eigen::MatrixXi & EF,
        Eigen::MatrixXi & EI,
        igl::min_heap< std::tuple<double,int,int> > & Q,
        Eigen::VectorXi & EQ,
        Eigen::MatrixXd & C,
        int & e,
        int & e1,
        int & e2,
        int & f1,
        int & f2
        );

bool collapse_by_neighbor(
        const int & e,
        const Eigen::RowVectorXd & p,
        const std::vector<int> & Nsf,
        std::vector<int> & Nsv,
        const std::vector<int> & Ndf,
        std::vector<int> & Ndv,
        Eigen::MatrixXd & V,
        Eigen::MatrixXi & F,
        Eigen::MatrixXi & E,
        Eigen::VectorXi & EMAP,
        Eigen::MatrixXi & EF,
        Eigen::MatrixXi & EI,
        int & a_e1,
        int & a_e2,
        int & a_f1,
        int & a_f2);

bool decimate(
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        const igl::decimate_cost_and_placement_callback & cost_and_placement,
        const igl::decimate_stopping_condition_callback & stopping_condition,
        const igl::decimate_pre_collapse_callback       & pre_collapse,
        const igl::decimate_post_collapse_callback      & post_collapse,
        Eigen::MatrixXd & U,
        Eigen::MatrixXi & G,
        Eigen::VectorXi & J,
        Eigen::VectorXi & I) {

    //check if all edges are manifold
    if(!igl::is_edge_manifold(F)) return false;

    //copy input V and F to preserve constant
    Eigen::MatrixXd V_ = V;
    Eigen::MatrixXi F_ = F;

    //#F*3 list, mapping directed edges to unique edge, so uE(EMAP(f+#F*c)) is the unique edge opposite F(f,c)
    Eigen::VectorXi EMAP;

    //#uE by 2 list of undirected edge indices into V
    Eigen::MatrixXi uE;

    //#uE by 2 list of edge flaps
    Eigen::MatrixXi EF;

    //#uE by 2 list of edge flap corners
    Eigen::MatrixXi EI;

    //get edge flaps and flap corners
    igl::edge_flaps(F_,uE,EMAP,EF,EI);

    //define a min-queue, use a tuple to store cost, edge index and the time stamp of insertion
    typedef std::tuple<double,int,int> queue_element;
    std::priority_queue<queue_element, std::vector<queue_element>, std::greater<queue_element>> Q;

    //list of "time" of last time pushed into Q
    Eigen::VectorXi EQ = Eigen::VectorXi::Zero(uE.rows());

    //store the new points' position which need to be created after an edge is collapsed
    Eigen::MatrixXd C(uE.rows(),V_.cols());

    //calculate the cost and "new position" of all edges
    for(int e = 0; e < uE.rows(); e++) {
        double cost = 0.0;
        Eigen::RowVectorXd p(1,3);
        cost_and_placement(e,V_,F_,uE,EMAP,EF,EI,cost,p);
        C.row(e) = p;
        Q.emplace(cost,e,0);
    }

    int prev_e = -1;
    bool clean_finish = false;

    while(true) {

        int e,e1,e2,f1,f2;

        if(collapse_one_edge(
                cost_and_placement, pre_collapse, post_collapse,
                V_,F_,uE,EMAP,EF,EI,Q,EQ,C,e,e1,e2,f1,f2)) {

            if(stopping_condition(V_,F_,uE,EMAP,EF,EI,Q,EQ,C,e,e1,e2,f1,f2)) {
                clean_finish = true;
                break;
            }
        }else
        {
            if(e == -1) {
                // no candidate edge
                break;
            }
            if(prev_e == e) {
                assert(false && "Edge collapse no progress... bad stopping condition?");
            }
        }
        prev_e = e;
    }


    Eigen::MatrixXi new_F(F_.rows(),3);
    J.resize(F_.rows());
    int counter = 0;
    for(int f = 0;f<F_.rows();f++) {
        if(F_(f,0) != -1 || F_(f,1) != -1 || F_(f,2) != -1)
        {
            new_F.row(counter) = F_.row(f);
            J(counter) = f;
            counter++;
        }
    }
    new_F.conservativeResize(counter, new_F.cols());
    J.conservativeResize(counter);

    Eigen::VectorXi temp;
    igl::remove_unreferenced(V,new_F,U,G,temp,I);
    return clean_finish;
}

void get_neighbors(const int & e,
                   const Eigen::MatrixXi & F,
                   const Eigen::MatrixXi & uE,
                   const Eigen::VectorXi & EMAP,
                   const Eigen::MatrixXi & EF,
                   const Eigen::MatrixXi & EI,
                   std::vector<int> & Nv,
                   std::vector<int> & Nf,
                   std::vector<int> & Nsv,
                   std::vector<int> & Nsf,
                   std::vector<int> & Ndv,
                   std::vector<int> & Ndf) {

    int v1 = uE(e, 0);
    int v2 = uE(e, 1);

    Nf.clear();
    Nv.clear();
    Nsf.clear();
    Nsv.clear();
    Ndf.clear();
    Ndv.clear();

    std::set<int> unique_faces;
    std::set<int> unique_vertices;
    std::set<int> unique_s_vertices;
    std::set<int> unique_d_vertices;
    std::queue<int> current_face;

    if(EF(e, 0) != -1) {
        current_face.push(EF(e, 0));
        unique_faces.insert(EF(e, 0));
    }
    if(EF(e, 1) != -1) {
        current_face.push(EF(e, 1));
        unique_faces.insert(EF(e, 1));
    }

    assert(!unique_faces.empty());

    while(!current_face.empty()) {

        int f = current_face.front();
        current_face.pop();

        for(int i = 0; i < 3; i++) {
            int new_e = EMAP(f + F.rows() * i);

            if(uE(new_e, 0) == v1 || uE(new_e, 1) == v1 || uE(new_e, 0) == v2 || uE(new_e, 1) == v2) {

                if(EF(new_e, 0) != -1 && unique_faces.find(EF(new_e, 0)) == unique_faces.end()) {
                    current_face.push(EF(new_e, 0));
                    unique_faces.insert(EF(new_e, 0));
                }

                if(EF(new_e, 1) != -1 && unique_faces.find(EF(new_e, 1)) == unique_faces.end()) {
                    current_face.push(EF(new_e, 1));
                    unique_faces.insert(EF(new_e, 1));
                }
            }
        }
    }

    for(const auto & f : unique_faces) {
        Nf.push_back(f);
        if(F(f, 0) == uE(e,0) || F(f, 1) == uE(e,0) || F(f, 2) == uE(e,0)) {
            Ndf.push_back(f);
            unique_d_vertices.insert(F(f, 0));
            unique_d_vertices.insert(F(f, 1));
            unique_d_vertices.insert(F(f, 2));
        }
        if(F(f, 0) == uE(e,1) || F(f, 1) == uE(e,1) || F(f, 2) == uE(e,1)) {
            Nsf.push_back(f);
            unique_s_vertices.insert(F(f, 0));
            unique_s_vertices.insert(F(f, 1));
            unique_s_vertices.insert(F(f, 2));
        }
        unique_vertices.insert(F(f, 0));
        unique_vertices.insert(F(f, 1));
        unique_vertices.insert(F(f, 2));
    }
    for(const auto & vertex : unique_s_vertices) {
        if(vertex != v2) Nsv.push_back(vertex);
    }

    for(const auto & vertex : unique_d_vertices) {
        if(vertex != v1) Ndv.push_back(vertex);
    }

    for(const auto & vertex : unique_vertices) {
        Nv.push_back(vertex);
    }



}

bool collapse_one_edge(
        const igl::decimate_cost_and_placement_callback & cost_and_placement,
        const igl::decimate_pre_collapse_callback       & pre_collapse,
        const igl::decimate_post_collapse_callback      & post_collapse,
        Eigen::MatrixXd & V,
        Eigen::MatrixXi & F,
        Eigen::MatrixXi & E,
        Eigen::VectorXi & EMAP,
        Eigen::MatrixXi & EF,
        Eigen::MatrixXi & EI,
        igl::min_heap< std::tuple<double,int,int> > & Q,
        Eigen::VectorXi & EQ,
        Eigen::MatrixXd & C,
        int & e,
        int & e1,
        int & e2,
        int & f1,
        int & f2) {

    while(true) {

        //there is no edge in the min-heap
        if(Q.empty()) {
            e = -1;
            return false;
        }

        //get the top tuple of the heap and pop it
        auto p = Q.top();
        Q.pop();

        //if the cost is infinity, do not collapse the edge
        if(std::get<0>(p) == std::numeric_limits<double>::infinity()) {
            e = -1;
            return false;
        }

        //get the index of edge
        e = std::get<1>(p);

        //get the insert time of tuple and check if matches timestamp
        if(std::get<2>(p) == EQ(e)) {
            break;
        }

        //if timestamp is -1, the edge has been collapsed; if timestamp is bigger than insert time, the edge has been updated
        assert(std::get<2>(p)  < EQ(e) || EQ(e) == -1);

        //continue to find another edge
    }

    //collect all neighbor faces and vertices of the edge e, classified by two points
    std::vector<int> Nf, Nv, Nsf, Ndf, Nsv, Ndv;
    get_neighbors(e, F, E, EMAP, EF, EI, Nv, Nf, Nsv, Nsf, Ndv, Ndf);


    bool collapsed;
    if(pre_collapse(V,F,E,EMAP,EF,EI,Q,EQ,C,e)) {
        collapsed = collapse_by_neighbor(
                e,C.row(e),
                Nsf,Nsv,Ndf, Ndv,
                V,F,E,EMAP,EF,EI,e1,e2,f1,f2);

    }else {
        collapsed = false;
    }

    post_collapse(V,F,E,EMAP,EF,EI,Q,EQ,C,e,e1,e2,f1,f2,collapsed);

    if(collapsed)
    {
        //mark collapsed edges' time stamp by -1 in EQ
        EQ(e) = -1;
        if(e1 >= 0) EQ(e1) = -1;
        if(e2 >= 0) EQ(e2) = -1;

        //update cost and position of all edges which have been updated
        //use set to store unique edges
        std::set<int> unique_edges;
        for(auto & n : Nf) {

            //if the triangle is not collapsed
            if(F(n,0) != -1 &&
               F(n,1) != -1 &&
               F(n,2) != -1) {

                //for all three edges of n
                for(int i = 0; i < 3; i++)
                {
                    // get edge id
                    const int ei = EMAP(i * F.rows()+n);


                    unique_edges.insert(ei);
                }
            }
        }

        for(auto & ei : unique_edges)
        {
            // compute cost and position
            double cost;
            Eigen::RowVectorXd place;
            cost_and_placement(ei,V,F,E,EMAP,EF,EI,cost,place);

            // Increment timestamp to store the version number of edges
            EQ(ei)++;

            //push updated edges in queue and update position matrix
            Q.emplace(cost,ei,EQ(ei));
            C.row(ei) = place;
        }

    }else {

        //if the edge is not collapsed, we do not want it be further considered, so update its cost as infinity
        EQ(e)++;
        Q.emplace(std::numeric_limits<double>::infinity(),e,EQ(e));
    }

    return collapsed;
}

bool collapse_by_neighbor(
        const int & e,
        const Eigen::RowVectorXd & p,
        const std::vector<int> & Nsf,
        std::vector<int> & Nsv,
        const std::vector<int> & Ndf,
        std::vector<int> & Ndv,
        Eigen::MatrixXd & V,
        Eigen::MatrixXi & F,
        Eigen::MatrixXi & E,
        Eigen::VectorXi & EMAP,
        Eigen::MatrixXi & EF,
        Eigen::MatrixXi & EI,
        int & a_e1,
        int & a_e2,
        int & a_f1,
        int & a_f2) {

    const int s = E(e,0)>E(e,1) ? E(e, 1) : E(e, 0);
    const int d = E(e,0)>E(e,1) ? E(e, 0) : E(e, 1);
    assert(s<d);


    if(!igl::edge_collapse_is_valid(Nsv,Ndv)) {
        return false;
    }


    //collect all neighbor faces of d
    const std::vector<int> & d_neighbor = (E(e,0) < E(e,1) ? Nsf : Ndf);

    // move source and destination to placement
    V.row(s) = p;
    V.row(d) = p;

    //set default value of collapsed edges and faces, if e is on the boundary, there is only one face collapsed
    a_e1 = -1;a_f1 = -1;a_e2 = -1;a_f2 = -1;

    // update edge info
    // for each flap
    const int m = (int)F.rows();
    for(int side = 0;side<2;side++)
    {

        const int f = EF(e,side);
        const int v = EI(e,side);

        if(f != -1) {
            int es, ed;
            for(int i = 0; i < 3; i++) {
                if(F(f, i) == d) es = EMAP(f + F.rows() * i);
                if(F(f, i) == s) ed = EMAP(f + F.rows() * i);
            }
            assert(E(ed, 0) == d || E(ed, 1) == d);
            assert(E(es, 0) == s || E(es, 1) == s);

            // face adjacent to f on ed, also incident on d
            int fd, vd;
            if(EF(ed, 0) == f) {
                fd = EF(ed, 1);
                vd = EI(ed, 1);
            } else {
                fd = EF(ed, 0);
                vd = EI(ed, 0);
            }
            assert(fd!=f);

            // collapse edge ed
            E(ed,0) = -1;
            E(ed,1) = -1;
            EF(ed,0) = -1;
            EF(ed,1) = -1;
            EI(ed,0) = -1;
            EI(ed,1) = -1;

            //collapse face f
            F(f,0) = -1;
            F(f,1) = -1;
            F(f,2) = -1;

            // map fd's edge on ed to es
            if(fd != -1)
                EMAP(fd + m * vd) = es;

            // side opposite f2, the face adjacent to f on es, also incident on s
            const int opp2 = (EF(es, 0) == f ? 0 : 1);
            assert(EF(es, opp2) == f);
            EF(es, opp2) = fd;
            EI(es, opp2) = vd;

            // remap es from d to s
            E(es, 0) = E(es, 0) == d ? s : E(es, 0);
            E(es, 1) = E(es, 1) == d ? s : E(es, 1);

            //update the collapsed edges and faces
            if(side==0) {
                a_e1 = ed;
                a_f1 = f;
            }else {
                a_e2 = ed;
                a_f2 = f;
            }
        }
    }

    //reindex all triangles and edges related to the collapsed vertex

    //use set to make sure we only process each edge once
    std::set<int> unique_edge;

    //for all faces adjacent to d
    for(auto f : d_neighbor) {
        for(int i = 0; i < 3; i++) {

            //update F
            if(F(f, i) == d)
                F(f,i) = s;

            const int edge = EMAP(f + F.rows() * i);

            //collect edges
            if(E(edge, 0) == d || E(edge, 1) == d)
                unique_edge.insert(edge);
        }
    }

    //update E
    for(auto edge : unique_edge) {
        if(E(edge, 0) == d) E(edge, 0) = s;
        if(E(edge, 1) == d) E(edge, 1) = s;
    }

    // Finally, remove the candidate edge e
    E(e,0) = -1;
    E(e,1) = -1;
    EF(e,0) = -1;
    EF(e,1) = -1;
    EI(e,0) = -1;
    EI(e,1) = -1;

    return true;
}

