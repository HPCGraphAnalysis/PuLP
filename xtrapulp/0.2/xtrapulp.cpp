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
#include <iostream>

#include "xtrapulp.h"

#include "comms.h"
#include "pulp_data.h"
#include "dist_graph.h"
#include "pulp_init.h"
#include "pulp_v.h"
#include "pulp_vec.h"

int procid, nprocs;
int seed;
bool verbose, debug, verify;
float X,Y;

//TODO: Integrate vertex_weights_num to dist_graph_t* and mvtxwgt_method to pulp_part_control_t?
extern "C" int xtrapulp_run(dist_graph_t* g, pulp_part_control_t* ppc,
          int* parts, int num_parts, int mvtxwgt_method)
{
  //setup
  mpi_data_t comm;
  pulp_data_t pulp;
  queue_data_t q;

  init_comm_data(&comm);
  init_pulp_data(g, &pulp, num_parts);
  init_queue_data(g, &q);

  if (ppc->do_repart)
    memcpy(pulp.local_parts, parts, g->n_local*sizeof(int32_t));

  //The meat
  xtrapulp(g, ppc, &comm, &pulp, &q, mvtxwgt_method);

  //cleanup
  memcpy(parts, pulp.local_parts, g->n_local*sizeof(int32_t));
  clear_comm_data(&comm);
  clear_pulp_data(&pulp);
  clear_queue_data(&q);

  return 0;
}

extern "C" int xtrapulp(dist_graph_t* g, pulp_part_control_t* ppc,
          mpi_data_t* comm, pulp_data_t* pulp, queue_data_t* q, int mvtxwgt_method)
{
  double vert_balance = ppc->vert_balance;
  //double vert_balance_lower = 0.25;
  double edge_balance = ppc->edge_balance;
  double do_label_prop = ppc->do_lp_init;
  double do_nonrandom_init = ppc->do_bfs_init;
  verbose = ppc->verbose_output;
  debug = false;
  bool do_vert_balance = true;
  //bool do_edge_balance = ppc->do_edge_balance;
  //bool do_maxcut_balance = ppc->do_maxcut_balance;
  bool do_repart = ppc->do_repart;
  bool has_vert_weights = (g->vertex_weights != NULL);
  bool has_edge_weights = (g->edge_weights != NULL);
  int balance_outer_iter = 1;
  int label_prop_iter = 3;
  int vert_outer_iter = 3;
  int vert_balance_iter = 5;
  int vert_refine_iter = 10;
  //int edge_outer_iter = 3;
  //int edge_balance_iter = 5;
  //int edge_refine_iter = 10;
  int num_parts = (int)pulp->num_parts;
  seed = ppc->pulp_seed;

  X = 1.0;
  Y = 0.25;
  // Tighten up allowable exchange for small graphs,
  //  and just use block initialization
  if (g->n/(long unsigned)nprocs < 100) {
    Y = 1.0;
    X = 2.0;
    do_label_prop = false;
    do_nonrandom_init = false;
  }

  double elt, elt2, elt3;
  elt = timer();

  //graph labeling initialization; Algorithm 2
  if (do_label_prop && (has_vert_weights || has_edge_weights))
  {
    if (procid == 0 && verbose) printf("\tDoing (weighted) lp init stage with %d parts\n", num_parts);
    elt2 = timer();
    pulp_init_label_prop_weighted(g, comm, q, pulp, label_prop_iter);
    elt2 = timer() - elt2;
    if (procid == 0 && verbose) printf("done: %9.6lf(s)\n", elt2);
  }
  /*
  else if (do_label_prop)
  {
    if (procid == 0 && verbose) printf("\tDoing lp init stage with %d parts\n", num_parts);
    elt2 = timer();
    pulp_init_label_prop(g, comm, q, pulp, label_prop_iter);
    elt2 = timer() - elt2;
    if (procid == 0 && verbose) printf("done: %9.6lf(s)\n", elt2);
  }*/
  else if (do_nonrandom_init)
  {
    if (procid == 0 && verbose) printf("\tDoing bfs init stage with %d parts\n", num_parts);
    elt2 = timer();
    pulp_init_bfs_max(g, comm, q, pulp);
    elt2 = timer() - elt2;
    if (procid == 0 && verbose) printf("done: %9.6lf(s)\n", elt2);
  }
  else if (!do_repart)
  {
    if (procid == 0 && verbose) printf("\tDoing block init stage with %d parts\n", num_parts);
    elt2 = timer();
    pulp_init_block(g, comm, q, pulp);
    elt2 = timer() - elt2;
    if (procid == 0 && verbose) printf("done: %9.6lf(s)\n", elt2);
  }

  //Vertex Balancing + Refinement; Algorithm 4
  if (procid == 0 && verbose) printf("\tBeginning vertex (and edge) refinement\n");
  for (int boi = 0; boi < balance_outer_iter; ++boi)
  {
    elt2 = timer();
    //if we want to balance vertices given vertex weights or edge weights, then run pulp_v_weighted
    if (do_vert_balance && (has_vert_weights || has_edge_weights))
    {
      if (procid == 0 && verbose) printf("\t\tDoing (weighted) vert balance and refinement stage\n");
      elt3 = timer();

      if((mvtxwgt_method == 1) && (g->vertex_weights_num > 1))
      {
        for(uint32_t wc = 0; wc < g->vertex_weights_num; ++wc)
        {
          std::cout << "Starting pulp_v_weighted" << std::endl;
          pulp_v_weighted(g, comm, q, pulp, vert_outer_iter, vert_balance_iter, vert_refine_iter, vert_balance, edge_balance, wc);
          std::cout << "Completed pulp_v_weighted" << std::endl;
        }
        //std::cout << "Starting pulp_vec_weighted" << std::endl;
        //pulp_vec_weighted(g, comm, q, pulp, vert_outer_iter, vert_balance_iter, vert_refine_iter, vert_balance, edge_balance);
        //std::cout << "Completed pulp_vec_weighted" << std::endl;
      }
      else
      {
        pulp_v_weighted(g, comm, q, pulp, vert_outer_iter, vert_balance_iter, vert_refine_iter, vert_balance, edge_balance, 0);
      }
      elt3 = timer() - elt3;
      if (procid == 0 && verbose) printf("done: %9.6lf(s)\n", elt3);
    }

    elt2 = timer() - elt2;
    if (procid == 0 && verbose) printf("\tFinished outer loop iter %d: %9.6lf(s)\n", (boi+1), elt2);
  }
  elt = timer() - elt;
  if (procid == 0 && verbose) printf("Partitioning finished: %9.6lf(s)\n", elt);

  return 0;
}

extern "C" int create_xtrapulp_dist_graph(dist_graph_t* g,
          unsigned long n_global, unsigned long m_global,
          unsigned long n_local, unsigned long m_local,
          unsigned long* local_adjs, unsigned long* local_offsets,
          unsigned long* global_ids, unsigned long* vert_dist,
          int* vertex_weights, int* edge_weights, double * vertex_weights_sum,
          unsigned long vertex_weights_num, int norm_option, int multiweight_option)
{
  //In xtrapulp.h, the default parameters are: vertex_weights_num = 1, norm_option = 2, and multiweight_option = 0
  if(multiweight_option == 0)
  {
    //converts multiple weights per vertex into a single weight per vertex
    vertex_weights = norm_weights(n_local, vertex_weights, vertex_weights_num, norm_option);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (nprocs > 1)
  {
    create_graph(g, (uint64_t)n_global, (uint64_t)m_global,
                 (uint64_t)n_local, (uint64_t)m_local,
                 (uint64_t*)local_offsets, (uint64_t*)local_adjs,
                 (uint64_t*)global_ids,
                 (int32_t*)vertex_weights, (int32_t*)edge_weights, 
                 (uint32_t*) vertex_weights_sum, (uint64_t) vertex_weights_num);
    relabel_edges(g, vert_dist);
  }
  else
  {
    create_graph_serial(g, (uint64_t)n_global, (uint64_t)m_global,
                 (uint64_t)n_local, (uint64_t)m_local,
                 (uint64_t*)local_offsets, (uint64_t*)local_adjs,
                 (int32_t*)vertex_weights, (int32_t*)edge_weights, 
                 (uint32_t*) vertex_weights_sum, (uint64_t) vertex_weights_num);
  }

  get_ghost_degrees(g);
  //get_ghost_weights(g);

  return 0;
}

// normalizes multiple weights to a scalar and single weight based on the argument norm_option
// If norm_option = 1, then 1-norm; 2, then 2 -norm; otherwise, inf-norm

extern "C" int * norm_weights(unsigned long n_local, int * vertex_weights, unsigned long vertex_weights_num, int norm_option)
{
	int * norm_vertex_weights = new int[n_local];
	
	for (int vtx_idx = 0; vtx_idx < (int) n_local; ++vtx_idx)
	{
		//unsigned long long is used since the norm-2 calculations involve very huge numbers
		unsigned long long result = 0;

		// vertex_weights is a 1-D array containing all vertex weights first ordered by the vertex each belongs to, second by order of weight components
		// vertex_weights[vtx_idx * vertex_weights_num + wc] corresponds to the vertex weight for vertex vtx_idx, weight component wc
		for (uint32_t wc = 0; wc < vertex_weights_num; ++wc)
		{
			if (norm_option == 1) result += vertex_weights[vtx_idx * vertex_weights_num + wc];
			else if (norm_option == 2) result += (unsigned long long) (vertex_weights[vtx_idx * vertex_weights_num + wc]) ^ 2;
			else if (vertex_weights[vtx_idx * vertex_weights_num + wc] > result) result = vertex_weights[vtx_idx * vertex_weights_num + wc];
		}
		if (norm_option == 2) result = sqrt(result);

		norm_vertex_weights[vtx_idx] = result;
	}

	return norm_vertex_weights;
}

