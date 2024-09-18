#include "poisson_surface_reconstruction.h"
#include "fd_interpolate.h"
#include "fd_grad.h"
#include <igl/copyleft/marching_cubes.h>
#include <algorithm>

void poisson_surface_reconstruction(
    const Eigen::MatrixXd & P,
    const Eigen::MatrixXd & N,
    Eigen::MatrixXd & V,
    Eigen::MatrixXi & F)
{
  ////////////////////////////////////////////////////////////////////////////
  // Construct FD grid, CONGRATULATIONS! You get this for free!
  ////////////////////////////////////////////////////////////////////////////
  // number of input points
  const int n = P.rows();
  // Grid dimensions
  int nx, ny, nz;
  // Maximum extent (side length of bounding box) of points
  double max_extent =
    (P.colwise().maxCoeff()-P.colwise().minCoeff()).maxCoeff();
  // padding: number of cells beyond bounding box of input points
  const double pad = 8;
  // choose grid spacing (h) so that shortest side gets 30+2*pad samples
  double h  = max_extent/double(30+2*pad);
  // Place bottom-left-front corner of grid at minimum of points minus padding
  Eigen::RowVector3d corner = P.colwise().minCoeff().array()-pad*h;
  // Grid dimensions should be at least 3 
  nx = std::max((P.col(0).maxCoeff()-P.col(0).minCoeff()+(2.*pad)*h)/h,3.);
  ny = std::max((P.col(1).maxCoeff()-P.col(1).minCoeff()+(2.*pad)*h)/h,3.);
  nz = std::max((P.col(2).maxCoeff()-P.col(2).minCoeff()+(2.*pad)*h)/h,3.);
  // Compute positions of grid nodes
  Eigen::MatrixXd x(nx*ny*nz, 3);
  for(int i = 0; i < nx; i++) 
  {
    for(int j = 0; j < ny; j++)
    {
      for(int k = 0; k < nz; k++)
      {
         // Convert subscript to index
         const auto ind = i + nx*(j + k * ny);
         x.row(ind) = corner + h*Eigen::RowVector3d(i,j,k);
      }
    }
  }
  Eigen::VectorXd g = Eigen::VectorXd::Zero(nx*ny*nz);

  ////////////////////////////////////////////////////////////////////////////

  //using fd_grad to construct a gradient matrix for a finite-difference grid
  Eigen::SparseMatrix<double> G;
  fd_grad(nx, ny, nz, h, G);

  //construct sparse trilinear interpolation matrices of three directions
  Eigen::SparseMatrix<double> W;
  fd_interpolate(nx, ny, nz, h, corner, P, W);

  //construct sparse trilinear interpolation matrices of three directions
  Eigen::SparseMatrix<double> Wx(n, (nx - 1) * ny * nz);
  Eigen::SparseMatrix<double> Wy(n, nx * (ny - 1) * nz);
  Eigen::SparseMatrix<double> Wz(n, nx * ny * (nz - 1));

/*
  //extract non-zero value in W to triplets
  std::vector<Eigen::Triplet<double>> tripletList;
  for(int i = 0; i < W.outerSize(); i++)
      for(typename Eigen::SparseMatrix<double>::InnerIterator it(W,i); it; ++it)
          tripletList.emplace_back(it.row(),it.col(),it.value());

  //set three sparse trilinear interpolation matrices by triplets
  Wx.setFromTriplets(tripletList.begin(), tripletList.end());
  Wy.setFromTriplets(tripletList.begin(), tripletList.end());
  Wz.setFromTriplets(tripletList.begin(), tripletList.end());
*/

  Eigen::RowVector3d x_corner;
  x_corner << corner(0) + h / 2, corner(1), corner(2);
  Eigen::RowVector3d y_corner;
  y_corner << corner(0), corner(1) + h / 2, corner(2);
  Eigen::RowVector3d z_corner;
  z_corner << corner(0), corner(1), corner(2) + h / 2;

  fd_interpolate(nx - 1, ny, nz, h, x_corner, P, Wx);
  fd_interpolate(nx, ny - 1, nz, h, y_corner, P, Wy);
  fd_interpolate(nx, ny, nz - 1, h, z_corner, P, Wz);



  //distribute the given normals N onto the staggered grid values in v
  Eigen::VectorXd vx = Wx.transpose() * N.col(0);
  Eigen::VectorXd vy = Wy.transpose() * N.col(1);
  Eigen::VectorXd vz = Wz.transpose() * N.col(2);

  //concatenate v together
  Eigen::VectorXd v(vx.rows() + vy.rows() + vz.rows());
  v << vx, vy, vz;

  //use Eigen::BiCGSTAB to solve g
  Eigen::SparseMatrix<double> GTG = G.transpose() * G;
  Eigen::VectorXd GTv = G.transpose() * v;
  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> bicg;
  Eigen::VectorXd solution = bicg.compute(GTG).solve(GTv);
  g = solution;

  //calculate sigma and pre-shift g
  double sigma = (1.0 / n) * Eigen::VectorXd::Ones(n).transpose() * W * g;
  Eigen::VectorXd sigmaVector(nx * ny * nz);
  sigmaVector.setConstant(sigma);
  g = g - sigmaVector;

  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Run black box algorithm to compute mesh from implicit function: this
  // function always extracts g=0, so "pre-shift" your g values by -sigma
  ////////////////////////////////////////////////////////////////////////////
  igl::copyleft::marching_cubes(g, x, nx, ny, nz, V, F);
}
