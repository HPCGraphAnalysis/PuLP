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
#ifndef __XTRAPULP_H__
#define __XTRAPULP_H__

#include <stdint.h>

//typedef int64_t pulp_int;
//typedef double pulp_real;

struct mpi_data_t;
struct pulp_data_t;
struct queue_data_t;
struct fast_map;

typedef struct {
  double vert_balance;
  double edge_balance;

  bool do_lp_init;
  bool do_bfs_init;
  bool do_repart;
  bool do_edge_balance;
  bool do_maxcut_balance;

  bool verbose_output;

  int pulp_seed;
} pulp_part_control_t;


struct dist_graph_t {
  uint64_t n;
  uint64_t m;
  uint64_t m_local;

  uint64_t n_local;
  uint64_t n_offset;
  uint64_t n_ghost;
  uint64_t n_total;

  uint64_t max_degree_vert;
  uint64_t max_degree;

  uint64_t* out_edges;
  uint64_t* out_degree_list;
  uint64_t* ghost_degrees;

  int32_t* vertex_weights;
  int32_t* edge_weights;
  uint64_t vertex_weights_sum;

  uint64_t* local_unmap;
  uint64_t* ghost_unmap;
  uint64_t* ghost_tasks;
  fast_map* map;
} ;
#define out_degree(g, n) (g->out_degree_list[n+1] - g->out_degree_list[n])
#define out_vertices(g, n) &g->out_edges[g->out_degree_list[n]]
#define out_weights(g, n) &g->edge_weights[g->out_degree_list[n]]


extern "C" int xtrapulp_run(dist_graph_t* g, pulp_part_control_t* ppc, 
          int* parts, int num_parts);

extern "C" int xtrapulp(dist_graph_t* g, pulp_part_control_t* ppc,
          mpi_data_t* comm, pulp_data_t* pulp, queue_data_t* q);

extern "C" int create_xtrapulp_dist_graph(dist_graph_t* g, 
          unsigned long n_global, unsigned long m_global, 
          unsigned long n_local, unsigned long m_local,
          unsigned long* local_adjs, unsigned long* local_offsets,  
          unsigned long* global_ids, unsigned long* vert_dist,
          int* vertex_weights, int* edge_weights);

double timer();

#endif
