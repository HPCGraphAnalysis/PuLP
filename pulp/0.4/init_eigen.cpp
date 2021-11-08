#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <iostream>

using namespace Spectra;

#include "graph.h"

int init_eigen(graph* g, int num_parts, int* parts)
{
  int num_verts = g->num_verts;
  int num_edges = g->num_edges;
  
  Eigen::SparseMatrix<int> A(num_verts, num_verts);
  Eigen::tripletList.reserve(num_edges+num_verts);
  typedef Eigen::Triplet<double> T;
  
  for (int i = 0; i < num_verts; ++i) {
    int vert = i;
    int degree = out_degree(g, i);
    int* outs = out_vertices(g, i);
    for (int j = 0; j < degree; ++j) {
      int out = outs[j];
      tripletList.push_back(T(vert, out, -1));
    }
    tripletList.push_back(T(vert, vert, degree));
  }
  
  A.set

    // Construct matrix operation object using the wrapper class SparseGenMatProd
    SparseGenMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    GenEigsSolver<SparseGenMatProd<double>> eigs(op, 3, 6);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute(SortRule::LargestMagn);

    // Retrieve results
    Eigen::VectorXcd evalues;
    if(eigs.info() == CompInfo::Successful)
        evalues = eigs.eigenvalues();

    std::cout << "Eigenvalues found:\n" << evalues << std::endl;

    return 0;
}
