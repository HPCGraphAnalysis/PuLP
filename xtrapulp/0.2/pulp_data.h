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

#ifndef _PULP_DATA_H
#define _PULP_DATA_H

#include "xtrapulp.h"

struct pulp_data_t {

  double avg_size;
  double avg_edge_size;
  double avg_cut_size;
  double max_v;
  double max_e;
  double max_c;
  double running_max_v;
  double running_max_e;
  double running_max_c;
  double weight_exponent_e;
  double weight_exponent_c;

  int32_t num_parts;
  int32_t* local_parts;

  int64_t cut_size;
  int64_t cut_size_change;
  int64_t max_cut;
  int64_t* part_sizes;
  int64_t* part_edge_sizes;
  int64_t* part_cut_sizes;
  int64_t* part_size_changes;
  int64_t* part_edge_size_changes;
  int64_t* part_cut_size_changes;
};

struct thread_pulp_t {
  double* part_counts;
  double* part_weights;
  double* part_edge_weights;
  double* part_cut_weights;
};

void init_thread_pulp(thread_pulp_t* tp, pulp_data_t* pulp);

void clear_thread_pulp(thread_pulp_t* tp);

void init_pulp_data(dist_graph_t* g, pulp_data_t* pulp, int32_t num_parts);

void update_pulp_data(dist_graph_t* g, pulp_data_t* pulp);

void update_pulp_data_weighted(dist_graph_t* g, pulp_data_t* pulp);

void clear_pulp_data(pulp_data_t* pulp);

#endif
