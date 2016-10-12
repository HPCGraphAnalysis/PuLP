/*
//@HEADER
// *****************************************************************************
//
// PuLP: Multi-Objective Multi-Constraint Partitioning Using Label Propagation
//              Copyright (2014) Sandia Corporation
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
//
// *****************************************************************************
//@HEADER
*/
#ifndef __pulp_h__
#define __pulp_h__



#define VERBOSE 0
#define DEBUG 0
#define OUTPUT_STEP 0
#define OUTPUT_TIME 1
#define WRITE_OUTPUT 1
#define DO_EVAL 0
#define THREAD_QUEUE_SIZE 2048
#define QUEUE_MULTIPLIER 2

//typedef int32_t pulp_part_t;
//typedef int32_t pulp_vert_t;

typedef struct {
  int n;
  long m;
  int* out_array;
  long* out_degree_list;
  int* vertex_weights;
  int* edge_weights;
  long vertex_weights_sum;
} pulp_graph_t;
#define out_degree(g, n) (g.out_degree_list[n+1] - g.out_degree_list[n])
#define out_vertices(g, n) &g.out_array[g.out_degree_list[n]]
#define out_weights(g, n) &g.edge_weights[g.out_degree_list[n]]


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


extern "C" int pulp_run(pulp_graph_t* g, pulp_part_control_t* ppc, 
          int* parts, int num_parts);

double timer();

void evaluate_quality(pulp_graph_t& g, int num_parts, int* parts);


#endif
