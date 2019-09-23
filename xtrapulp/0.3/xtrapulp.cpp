/*
//@HEADER
// *****************************************************************************
//
//  XtraPuLP: Xtreme-Scale Graph Partitioning using Label Propagation
//              Copyright (2016) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota   (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//                      Kamesh Madduri    (madduri@cse.psu.edu)
//
// *****************************************************************************
//@HEADER
*/

#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <cstring>
#include <sys/time.h>
#include <time.h>

#include "xtrapulp.h"

#include "comms.h"
#include "pulp_data.h"
#include "dist_graph.h"
#include "pulp_init.h"
#include "pulp_w.h"
#include "pulp_mm.h"

int procid, nprocs;
int seed;
bool verbose, debug, verify;
float X,Y;

extern "C" int xtrapulp_run(
  dist_graph_t* g, pulp_part_control_t* ppc,
  int* parts, int num_parts)
{
  mpi_data_t comm;
  pulp_data_t pulp;
  queue_data_t q;
  init_comm_data(&comm);
  init_pulp_data(g, &pulp, num_parts);
  init_queue_data(g, &q);
  if (ppc->do_repart)
    memcpy(pulp.local_parts, parts, g->n_local*sizeof(int32_t));   

  if (g->num_weights == 0)
    xtrapulp(g, ppc, &comm, &pulp, &q);
  else
    xtrapulp_weighted(g, ppc, &comm, &pulp, &q);

  memcpy(parts, pulp.local_parts, g->n_local*sizeof(int32_t));  
  clear_comm_data(&comm);
  clear_pulp_data(&pulp);
  clear_queue_data(&q);

  return 0;
}


extern "C" int xtrapulp(dist_graph_t* g, pulp_part_control_t* ppc,
          mpi_data_t* comm, pulp_data_t* pulp, queue_data_t* q)
{
  double vert_balance = ppc->vert_balance;
  double edge_balance = ppc->edge_balance;
  double do_label_prop = ppc->do_lp_init;
  double do_nonrandom_init = ppc->do_bfs_init;
  verbose = ppc->verbose_output;
  debug = false;
  bool do_vert_balance = true;
  bool do_edge_balance = ppc->do_edge_balance;
  bool do_maxcut_balance = ppc->do_maxcut_balance;
  bool do_repart = ppc->do_repart;
  int balance_outer_iter = 1;
  int label_prop_iter = 3;
  int outer_iter = 3;
  int balance_iter = 5;
  int refine_iter = 10;
  int num_parts = (int)pulp->num_parts;
  seed = ppc->pulp_seed;

  X = 1.0;
  Y = 0.25;
  // Tighten up allowable exchange for small graphs, 
  //  and just use block initialization
  if (g->n/(long unsigned)nprocs < 100) {
    X = 2.0;
    Y = 1.0;
    do_label_prop = false;
    do_nonrandom_init = false;
  }

  double elt, elt2, elt3;
  elt = omp_get_wtime();
  if (do_label_prop)
  {
    elt2 = omp_get_wtime();
    if (procid == 0 && verbose) 
      printf("\tDoing lp init stage with %d parts\n", num_parts);

    pulp_init_label_prop(g, comm, q, pulp, label_prop_iter);

    elt2 = omp_get_wtime() - elt2;
    if (procid == 0 && verbose) 
      printf("done: %9.6lf(s)\n", elt2);
  }
  else if (do_nonrandom_init)
  {
    elt2 = omp_get_wtime();
    if (procid == 0 && verbose) 
      printf("\tDoing bfs init stage with %d parts\n", num_parts);

    pulp_init_bfs_max(g, comm, q, pulp);

    elt2 = omp_get_wtime() - elt2;
    if (procid == 0 && verbose) 
      printf("done: %9.6lf(s)\n", elt2);
  }
  else if (!do_repart)
  {
    elt2 = omp_get_wtime();
    if (procid == 0 && verbose) 
      printf("\tDoing block init stage with %d parts\n", num_parts);

    pulp_init_block(g, comm, q, pulp);

    elt2 = omp_get_wtime() - elt2;
    if (procid == 0 && verbose) 
      printf("done: %9.6lf(s)\n", elt2);
  }

  if (procid == 0 && verbose) 
    printf("\tBeginning vertex (and edge) refinement\n");
  for (int boi = 0; boi < balance_outer_iter; ++boi)
  {
    elt2 = omp_get_wtime();
    if (do_vert_balance)
    {
      elt3 = omp_get_wtime(); 
      if (procid == 0 && verbose) 
        printf("\t\tDoing vert balance and refinement stage\n");

      pulp_v(g, comm, q, pulp, 
        outer_iter, balance_iter, refine_iter, 
        vert_balance, edge_balance);

      elt3 = omp_get_wtime() - elt3;
      if (procid == 0 && verbose) 
        printf("done: %9.6lf(s)\n", elt3);
    }

    if (do_edge_balance && !do_maxcut_balance)
    {
      elt3 = omp_get_wtime();
      if (procid == 0 && verbose) 
        printf("\t\tDoing edge balance and refinement stage\n");

      pulp_ve(g, comm, q, pulp, 
        outer_iter, balance_iter, refine_iter, 
        vert_balance, edge_balance);

      elt3 = omp_get_wtime() - elt3;
      if (procid == 0 && verbose) 
        printf("done: %9.6lf(s)\n", elt3);
    }      
    else if (do_edge_balance && do_maxcut_balance)
    {
      elt3 = omp_get_wtime();
      if (procid == 0 && verbose) 
        printf("\t\tDoing maxcut balance and refinement stage\n");

      pulp_vec(g, comm, q, pulp, 
        outer_iter, balance_iter, refine_iter, 
        vert_balance, edge_balance);

      elt3 = omp_get_wtime() - elt3;
      if (procid == 0 && verbose) 
        printf("done: %9.6lfs\n", elt3);
    }

    elt2 = omp_get_wtime() - elt2;
    if (procid == 0 && verbose) 
      printf("\tFinished outer loop iter %d: %9.6lf(s)\n", (boi+1), elt2);
  }
  elt = omp_get_wtime() - elt;
  if (procid == 0 && verbose) 
    printf("Partitioning finished: %9.6lf(s)\n", elt);

  return 0;
}


extern "C" int xtrapulp_weighted(
  dist_graph_t* g, pulp_part_control_t* ppc,
  mpi_data_t* comm, pulp_data_t* pulp, queue_data_t* q)
{
  double do_label_prop = ppc->do_lp_init;
  double do_nonrandom_init = ppc->do_bfs_init;
  verbose = ppc->verbose_output;
  debug = false;
  bool do_repart = ppc->do_repart;
  int label_prop_iter = 3;
  int outer_iter = 3*g->num_weights;
  int balance_iter = 5;
  int refine_iter = 10;
  int num_parts = (int)pulp->num_parts;
  seed = ppc->pulp_seed;

  X = 1.25;
  Y = 1.0;
  // Tighten up allowable exchange for small graphs, 
  //  and just use block initialization
  if (g->n/(long unsigned)nprocs < 5000) {
    X = 2.0;
    Y = 1.25;
    do_label_prop = false;
    do_nonrandom_init = false;
  }

  double elt, elt2;
  elt = omp_get_wtime();
  if (do_label_prop)
  {
    elt2 = omp_get_wtime();
    if (procid == 0 && verbose) 
      printf("\tDoing (weighted) lp init stage with %d parts\n", num_parts);

    pulp_init_label_prop_weighted(g, comm, q, pulp, label_prop_iter);

    elt2 = omp_get_wtime() - elt2;
    if (procid == 0 && verbose) 
      printf("done: %9.6lf(s)\n", elt2);
  }
  else if (do_nonrandom_init)
  {
    elt2 = omp_get_wtime();
    if (procid == 0 && verbose) 
      printf("\tDoing bfs init stage with %d parts\n", num_parts);

    pulp_init_bfs_max(g, comm, q, pulp);

    elt2 = omp_get_wtime() - elt2;
    if (procid == 0 && verbose) 
      printf("done: %9.6lf(s)\n", elt2);
  }
  else if (!do_repart)
  {
    elt2 = omp_get_wtime();
    if (procid == 0 && verbose) 
      printf("\tDoing block init stage with %d parts\n", num_parts);

    pulp_init_block(g, comm, q, pulp);

    elt2 = omp_get_wtime() - elt2;
    if (procid == 0 && verbose) 
      printf("done: %9.6lf(s)\n", elt2);
  }

  elt2 = omp_get_wtime(); 
  if (procid == 0 && verbose) 
    printf("\t\tDoing (weighted) vert balance and refinement stage\n");

  pulp_w(g, comm, q, pulp, 
    outer_iter, balance_iter, refine_iter, 
    ppc->constraints);

  elt2 = omp_get_wtime() - elt2;
  if (procid == 0 && verbose) 
    printf("done: %9.6lf(s)\n", elt2);

  elt = omp_get_wtime() - elt;
  if (procid == 0 && verbose) 
    printf("Partitioning finished: %9.6lf(s)\n", elt);

  return 0;
}


extern "C" int create_xtrapulp_dist_graph(
  dist_graph_t* g, 
  unsigned long n_global, unsigned long m_global, 
  unsigned long n_local, unsigned long m_local,
  unsigned long* local_adjs, unsigned long* local_offsets, 
  unsigned long* global_ids, unsigned long* vert_dist,
  int num_weights, int* vertex_weights, int* edge_weights)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (nprocs > 1)
  {
    create_graph(g, (uint64_t)n_global, (uint64_t)m_global, 
                 (uint64_t)n_local, (uint64_t)m_local,
                 (uint64_t*)local_offsets, (uint64_t*)local_adjs, 
                 (uint64_t*)global_ids,
                 (int32_t*)vertex_weights, (int32_t*)edge_weights);
    relabel_edges(g, vert_dist);
  }
  else
  {
    create_graph_serial(g, (uint64_t)n_global, (uint64_t)m_global, 
                 (uint64_t)n_local, (uint64_t)m_local,
                 (uint64_t*)local_offsets, (uint64_t*)local_adjs,
                 (int32_t*)vertex_weights, (int32_t*)edge_weights);
  }

  if (num_weights == 0)
    get_ghost_degrees(g);

  return 0;
}

