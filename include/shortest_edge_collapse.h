#ifndef SHORTEST_EDGE_COLLAPSE_H
#define SHORTEST_EDGE_COLLAPSE_H
#include <Eigen/Core>
  /// Decimate (simplify) a triangle mesh in nD using the shortest edge collapse
  /// method.
  ///
  /// @param[in] V  #V by dim list of vertex positions.
  /// @param[in] F  #F by 3 list of triangle indices into V
  /// @param[in] max_m  desired number of output faces
  /// @param[out] U  #U by dim list of output vertex posistions (can be same ref as V)
  /// @param[out] G  #G by 3 list of output face indices into U (can be same ref as F)
  /// @param[out] J  #G list of indices into F of birth face
  /// @param[out] I  #U list of indices into V of birth vertices
  /// @return true if m was reached (otherwise #G > m)
  ///
  bool shortest_edge_collapse(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const int max_m,
    Eigen::MatrixXd & U,
    Eigen::MatrixXi & G,
    Eigen::VectorXi & J,
    Eigen::VectorXi & I);

#endif
