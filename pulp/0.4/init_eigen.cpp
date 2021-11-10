#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <iostream>

#include "graph.h"
#include "util.h"

int init_eigen(graph* g, int num_parts, int* parts)
{
  printf("Begin init_eigen()\n");
  double elt2 = omp_get_wtime();

  printf("Constructing matrix\n");
  double elt = elt2;

  int num_verts = g->num_verts;
  int num_edges = g->num_edges;

  printf("n: %d, m: %d\n", num_verts, num_edges);
  
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(num_edges+num_verts);
  Eigen::SparseMatrix<double> A(num_verts, num_verts);
  
  for (int i = 0; i < num_verts; ++i) {
    int vert = i;
    int degree = out_degree(g, i);
    int* outs = out_vertices(g, i);
    int* weights = out_vertices(g, i);
    for (int j = 0; j < degree; ++j) {
      int out = outs[j];
      if (vert < out) {
        tripletList.push_back(T(vert, out, -1.0*weights[j]));
        tripletList.push_back(T(out, vert, -1.0*weights[j]));
      }
    }
    tripletList.push_back(T(vert, vert, (double)degree));
  }
  
  A.setFromTriplets(tripletList.begin(), tripletList.end());
  printf("\tdone: %lf\n", omp_get_wtime() - elt);
  printf("Solving system\n");
  elt = omp_get_wtime();

  Spectra::SparseSymMatProd<double> op(A);

  int num_eigenvalues = (int)log2((double)num_parts) + 1;
  Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(op, num_eigenvalues, num_eigenvalues*2);

  eigs.init();
  int nconv = eigs.compute(Spectra::SortRule::LargestAlge);

  Eigen::VectorXd evalues;
  Eigen::MatrixXd evectors;
  if (eigs.info() == Spectra::CompInfo::Successful) {
    evalues = eigs.eigenvalues();
    evectors = eigs.eigenvectors();
  }
  else
    printf("FAILURE\n");

  printf("\tdone: %lf\n", omp_get_wtime() - elt);
  printf("Determining initial partition\n");
  elt = omp_get_wtime();

  //std::cout << "Eigs: " << evalues << std::endl;
  //std::cout << "Eigs: " << eigs.eigenvectors() << std::endl;



  double** eigenvectors = new double*[num_eigenvalues-1];
  for (int i = 0; i < num_eigenvalues-1; ++i)
    eigenvectors[i] = new double[num_verts];

  for (int e = 0; e < num_eigenvalues-1; ++e) {
#pragma omp parallel for
    for (int i = 0; i < num_verts; ++i) {
      eigenvectors[e][i] = evectors(i, e+1);
    }
  }

  // k means
  double** centroids = new double*[num_parts];
  double** centroid_sums = new double*[num_parts];
  int* centroid_counts = new int[num_parts];
  for (int i = 0; i < num_parts; ++i) {
    centroids[i] = new double[num_eigenvalues-1];
    centroid_sums[i] = new double[num_eigenvalues-1];
    centroid_counts[i] = 0;
    int rand_index = rand() % num_verts;
    for (int j = 0; j < num_eigenvalues-1; ++j) {
      centroids[i][j] = eigenvectors[j][rand_index];
      centroid_sums[i][j] = 0.0;
    }
  }

#pragma omp parallel for
  for (int i = 0; i < num_verts; ++i)
    parts[i] = -1;

  int max_size = g->vert_weights_sum / num_parts;
  int changes = 1;
  int iter = 0;
  while (changes && iter < 10) {
    ++iter;
    changes = 0;

#pragma omp parallel for reduction(+:changes)
    for (int i = 0; i < num_verts; ++i) {
      double min_distance = 9999999999999;
      int min_k = -1;

      for (int k = 0; k < num_parts; ++k) {
        double distance = 0.0;
        for (int j = 0; j < num_eigenvalues-1; ++j) {
          distance += pow(centroids[k][j] - eigenvectors[j][i], 2.0);
        }
        //if (centroid_counts[k] > max_size)
        distance *= (double)centroid_counts[k] / (double)max_size;
        //printf("%d %d %lf\n", i, k, distance);

        //distance = pow(distance, 2.0);

        if (distance < min_distance) {
          min_distance = distance;
          min_k = k;
        }
      }

      if (min_k != parts[i]) {
        parts[i] = min_k;
        ++changes;
      }
  #pragma omp atomic
      centroid_counts[parts[i]] += g->vert_weights[i];

      for (int j = 0; j < num_eigenvalues-1; ++j) {
        centroid_sums[min_k][j] += eigenvectors[j][i] * (double)g->vert_weights[i];
      }
    }
    printf("Changes: %d\n", changes);

    for (int i = 0; i < num_parts; ++i) {
      printf("Centroid %d (%d): ", i, centroid_counts[i]);
      for (int j = 0; j < num_eigenvalues-1; ++j) {
        centroids[i][j] = centroid_sums[i][j] / (double)centroid_counts[i];
        centroid_sums[i][j] = 0.0;
        printf("%lf ", centroids[i][j]);
      }
      printf("\n");
      centroid_counts[i] = 0;
    } 
  }


//   double* avgs = new double[num_eigenvalues-1];
//   for (int e = 0; e < num_eigenvalues-1; ++e) {
//     double sum = 0.0;
// #pragma omp parallel for reduction(+:sum)
//       for (int i = 0; i < num_verts; ++i) {
//         sum += eigenvectors[e][i];
//     }
//     avgs[e] = sum / (double)num_verts;
//   }

// #pragma omp parallel for
//   for (int i = 0; i < num_verts; ++i) {
//     if (eigenvectors[0][i] < avgs[0])
//       if (eigenvectors[1][i] < avgs[1])
//         parts[i] = 0;
//       else
//         parts[i] = 1;
//     else
//       if (eigenvectors[1][i] < avgs[1])
//         parts[i] = 2;
//       else
//         parts[i] = 3;
//   }




//   quicksort(eigenvector, 0, num_verts-1);
//   double median = eigenvector[num_verts/2];
//   printf("Median: %lf\n", median);

// #pragma omp parallel for
//   for (int i = 0; i < num_verts; ++i)
//     if (evectors(i,1) < median)
//       parts[i] = 0;
//     else if (evectors(i,1) > median)
//       parts[i] = 1;
//     else if (rand() & 1)
//       parts[i] = 1;
//     else
//       parts[i] = 0;

  printf("\tdone: %lf\n", omp_get_wtime() - elt);
  printf("Done init_eigen(): %lf (s)\n", omp_get_wtime() - elt2);
  
#if OUTPUT_STEP
  char step[] = "InitEigen";
  evaluate_quality_step(step, g, num_parts, parts);
#endif

  return 0;
}
