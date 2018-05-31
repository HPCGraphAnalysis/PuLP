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
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "pulp_data.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;

void init_thread_pulp(thread_pulp_t* tp, pulp_data_t* pulp)
{  
  //if (debug) printf("Task %d init_thread_pulp() start\n", procid); 

  tp->part_counts = (double*)malloc(pulp->num_parts*sizeof(double));
  tp->part_weights = (double*)malloc(pulp->num_parts*sizeof(double));
  tp->part_edge_weights = (double*)malloc(pulp->num_parts*sizeof(double));
  tp->part_cut_weights = (double*)malloc(pulp->num_parts*sizeof(double));
 
  //if (debug) printf("Task %d init_thread_pulp() success\n", procid);
}

void clear_thread_pulp(thread_pulp_t* tp)
{
  //if (debug) printf("Task %d clear_thread_pulp() start\n", procid); 

  free(tp->part_counts);
  free(tp->part_weights);
  free(tp->part_edge_weights);
  free(tp->part_cut_weights);

  //if (debug) printf("Task %d clear_thread_pulp() success\n", procid);
}

void init_pulp_data(dist_graph_t* g, pulp_data_t* pulp, int32_t num_parts)
{
  if (debug) printf("Task %d init_pulp_data() start\n", procid); 

  pulp->num_parts = num_parts;
  if (g->edge_weights == NULL && g->vertex_weights == NULL)
    pulp->avg_size = (double)g->n / (double)pulp->num_parts;
  else
    pulp->avg_size = (double)g->vertex_weights_sum / (double)pulp->num_parts;
  pulp->avg_edge_size = (double)g->m*2 / (double)pulp->num_parts;
  pulp->avg_cut_size = 0.0;
  pulp->max_v = 0.0;
  pulp->max_e = 1.0;
  pulp->max_c = 1.0;
  pulp->weight_exponent_e = 1.0;
  pulp->weight_exponent_c = 1.0;
  pulp->running_max_v = 1.0;
  pulp->running_max_e = 1.0;
  pulp->running_max_c = 1.0;

  pulp->local_parts = (int32_t*)malloc(g->n_total*sizeof(int32_t));
  pulp->part_sizes = (int64_t*)malloc(pulp->num_parts*sizeof(int64_t));
  pulp->part_edge_sizes = (int64_t*)malloc(pulp->num_parts*sizeof(int64_t));
  pulp->part_cut_sizes = (int64_t*)malloc(pulp->num_parts*sizeof(int64_t));
  pulp->part_size_changes = (int64_t*)malloc(pulp->num_parts*sizeof(int64_t));
  pulp->part_edge_size_changes = (int64_t*)malloc(pulp->num_parts*sizeof(int64_t));
  pulp->part_cut_size_changes = (int64_t*)malloc(pulp->num_parts*sizeof(int64_t));
  if (pulp->local_parts == NULL || 
      pulp->part_sizes == NULL || pulp->part_edge_sizes == NULL ||
      pulp->part_cut_sizes == NULL || 
      pulp->part_size_changes == NULL || pulp->part_edge_size_changes == NULL ||
      pulp->part_cut_size_changes == NULL)
    throw_err("init_pulp_data(), unable to allocate resources", procid);

  pulp->cut_size = 0;
  pulp->max_cut = 0;
  pulp->cut_size_change = 0;
  for (int32_t p = 0; p < pulp->num_parts; ++p)
    pulp->part_sizes[p] = 0;
  for (int32_t p = 0; p < pulp->num_parts; ++p)
    pulp->part_edge_sizes[p] = 0;
  for (int32_t p = 0; p < pulp->num_parts; ++p)
    pulp->part_cut_sizes[p] = 0;
  for (int32_t p = 0; p < pulp->num_parts; ++p)
    pulp->part_size_changes[p] = 0;
  for (int32_t p = 0; p < pulp->num_parts; ++p)
    pulp->part_edge_size_changes[p] = 0;
  for (int32_t p = 0; p < pulp->num_parts; ++p)
    pulp->part_cut_size_changes[p] = 0;
  
  if (debug) printf("Task %d init_pulp_data() success\n", procid);
}

void update_pulp_data(dist_graph_t* g, pulp_data_t* pulp)
{
  for (int32_t p = 0; p < pulp->num_parts; ++p)
  {
    pulp->part_sizes[p] = 0;
    pulp->part_edge_sizes[p] = 0;
    pulp->part_cut_sizes[p] = 0;
    pulp->part_size_changes[p] = 0;
    pulp->part_edge_size_changes[p] = 0;
    pulp->part_cut_size_changes[p] = 0;
  }
  pulp->cut_size = 0;

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert_index = i;
    int32_t part = pulp->local_parts[vert_index];
    ++pulp->part_sizes[part];

    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    pulp->part_edge_sizes[part] += (int64_t)out_degree;
    for (uint64_t j = 0; j < out_degree; ++j)
    {
      uint64_t out_index = outs[j];
      int32_t part_out = pulp->local_parts[out_index];
      if (part_out != part)
      {
        ++pulp->part_cut_sizes[part];
        ++pulp->cut_size;
      }
    }
  }


  MPI_Allreduce(MPI_IN_PLACE, pulp->part_sizes, pulp->num_parts, 
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_edge_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_cut_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pulp->cut_size, 1,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  pulp->cut_size /= 2;
  pulp->avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;

  pulp->max_v = 0;
  pulp->max_e = 0;
  pulp->max_c = 0;
  pulp->max_cut = 0;
  for (int32_t p = 0; p < pulp->num_parts; ++p)
  {
    if ((double)pulp->part_sizes[p] / pulp->avg_size > pulp->max_v)
      pulp->max_v = (double)pulp->part_sizes[p] / pulp->avg_size;        
    if ((double)pulp->part_edge_sizes[p] / pulp->avg_edge_size > pulp->max_e)
      pulp->max_e = (double)pulp->part_edge_sizes[p] / pulp->avg_edge_size;
    if ((double)pulp->part_cut_sizes[p] / pulp->avg_cut_size > pulp->max_c)
      pulp->max_c = (double)pulp->part_cut_sizes[p] / pulp->avg_cut_size;
    if (pulp->part_cut_sizes[p] > pulp->max_cut)
      pulp->max_cut = pulp->part_cut_sizes[p];
  }
}

void update_pulp_data_weighted(dist_graph_t* g, pulp_data_t* pulp)
{
  bool has_vwgts = (g->vertex_weights != NULL);
  bool has_ewgts = (g->edge_weights != NULL);

  for (int32_t p = 0; p < pulp->num_parts; ++p)
  {
    pulp->part_sizes[p] = 0;
    pulp->part_edge_sizes[p] = 0;
    pulp->part_cut_sizes[p] = 0;
    pulp->part_size_changes[p] = 0;
    pulp->part_edge_size_changes[p] = 0;
    pulp->part_cut_size_changes[p] = 0;
  }
  pulp->cut_size = 0;

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert_index = i;
    int32_t part = pulp->local_parts[vert_index];
    if (has_vwgts) 
      pulp->part_sizes[part] += g->vertex_weights[vert_index];
    else 
      ++pulp->part_sizes[part];

    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    int32_t* weights = out_weights(g, vert_index);
    pulp->part_edge_sizes[part] += (int64_t)out_degree;
    for (uint64_t j = 0; j < out_degree; ++j)
    {
      uint64_t out_index = outs[j];
      int32_t part_out = pulp->local_parts[out_index];
      if (part_out != part)
      {
        if (has_ewgts)
        {
          pulp->part_cut_sizes[part] += weights[j];
          pulp->cut_size += weights[j];
        }
        else
        {
          ++pulp->part_cut_sizes[part];
          ++pulp->cut_size;
        }
      }
    }
  }


  MPI_Allreduce(MPI_IN_PLACE, pulp->part_sizes, pulp->num_parts, 
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_edge_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_cut_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pulp->cut_size, 1,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  pulp->cut_size /= 2;
  pulp->avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;

  pulp->max_v = 0;
  pulp->max_e = 0;
  pulp->max_c = 0;
  pulp->max_cut = 0;
  for (int32_t p = 0; p < pulp->num_parts; ++p)
  {
    if ((double)pulp->part_sizes[p] / pulp->avg_size > pulp->max_v)
      pulp->max_v = (double)pulp->part_sizes[p] / pulp->avg_size;        
    if ((double)pulp->part_edge_sizes[p] / pulp->avg_edge_size > pulp->max_e)
      pulp->max_e = (double)pulp->part_edge_sizes[p] / pulp->avg_edge_size;
    if ((double)pulp->part_cut_sizes[p] / pulp->avg_cut_size > pulp->max_c)
      pulp->max_c = (double)pulp->part_cut_sizes[p] / pulp->avg_cut_size;
    if (pulp->part_cut_sizes[p] > pulp->max_cut)
      pulp->max_cut = pulp->part_cut_sizes[p];
  }

}



void clear_pulp_data(pulp_data_t* pulp)
{
  if (debug) printf("Task %d clear_pulp_data() start\n", procid); 

  free(pulp->local_parts);
  free(pulp->part_sizes);
  free(pulp->part_edge_sizes);
  free(pulp->part_cut_sizes);
  free(pulp->part_size_changes);
  free(pulp->part_edge_size_changes);
  free(pulp->part_cut_size_changes);

  if (debug) printf("Task %d clear_pulp_data() success\n", procid); 
}
