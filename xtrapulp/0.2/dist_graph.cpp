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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "xtrapulp.h"
#include "fast_map.h"
#include "dist_graph.h"
#include "comms.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;

int create_graph(graph_gen_data_t *ggi, dist_graph_t *g)
{  
  if (debug) { printf("Task %d create_graph() start\n", procid); }

  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = ggi->n;
  g->n_local = ggi->n_local;
  g->n_offset = ggi->n_offset;
  g->m = ggi->m;
  g->m_local = ggi->m_local_edges;
  g->vertex_weights = NULL;
  g->edge_weights = NULL;
  g->map = (struct fast_map*)malloc(sizeof(struct fast_map));

  uint64_t* out_edges = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  uint64_t* out_degree_list = (uint64_t*)malloc((g->n_local+1)*sizeof(uint64_t));
  uint64_t* temp_counts = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (out_edges == NULL || out_degree_list == NULL || temp_counts == NULL)
    throw_err("create_graph(), unable to allocate graph edge storage", procid);

#pragma omp parallel
{
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local+1; ++i)
    out_degree_list[i] = 0;
#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = 0;
}

  for (uint64_t i = 0; i < g->m_local*2; i+=2)
    ++temp_counts[ggi->gen_edges[i] - g->n_offset];
  for (uint64_t i = 0; i < g->n_local; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, out_degree_list, g->n_local*sizeof(uint64_t));


  for (uint64_t i = 0; i < g->m_local*2; i+=2)
    out_edges[temp_counts[ggi->gen_edges[i] - g->n_offset]++] = ggi->gen_edges[i+1];
  
  free(ggi->gen_edges);
  free(temp_counts);
  g->out_edges = out_edges;
  g->out_degree_list = out_degree_list;

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (g->local_unmap == NULL)
    throw_err("create_graph(), unable to allocate unmap", procid);

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = i + g->n_offset;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d create_graph() %9.6f (s)\n", procid, elt);
  }

  if (debug) { printf("Task %d create_graph() success\n", procid); }
  return 0;
}

int create_graph_serial(graph_gen_data_t *ggi, dist_graph_t *g)
{
  if (debug) { printf("Task %d create_graph_serial() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = ggi->n;
  g->n_local = ggi->n_local;
  g->n_offset = 0;
  g->m = ggi->m;
  g->m_local = ggi->m_local_edges;
  g->n_ghost = 0;
  g->n_total = g->n_local;
  g->vertex_weights = NULL;
  g->edge_weights = NULL;
  g->map = (struct fast_map*)malloc(sizeof(struct fast_map));

  uint64_t* out_edges = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  uint64_t* out_degree_list = (uint64_t*)malloc((g->n_local+1)*sizeof(uint64_t));
  uint64_t* temp_counts = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (out_edges == NULL || out_degree_list == NULL || temp_counts == NULL)
  throw_err("create_graph_serial(), unable to allocate out edge storage\n", procid);

#pragma omp parallel
{
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local+1; ++i)
    out_degree_list[i] = 0;
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = 0;
}

  for (uint64_t i = 0; i < g->m_local*2; i+=2)
    ++temp_counts[ggi->gen_edges[i] - g->n_offset];
  for (uint64_t i = 0; i < g->n_local; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, out_degree_list, g->n_local*sizeof(uint64_t));
  for (uint64_t i = 0; i < g->m_local*2; i+=2)
    out_edges[temp_counts[ggi->gen_edges[i] - g->n_offset]++] = ggi->gen_edges[i+1];

  free(ggi->gen_edges);
  free(temp_counts);
  g->out_edges = out_edges;
  g->out_degree_list = out_degree_list;

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));  
  if (g->local_unmap == NULL)
    throw_err("create_graph_serial(), unable to allocate unmap\n", procid);

  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = i + g->n_offset;

  //int64_t total_edges = g->m_local_in + g->m_local_out;
  init_map_nohash(g->map, g->n);
  for (uint64_t i = 0; i < g->n_local; ++i)
    set_value_uq(g->map, i, i);


  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d create_graph_serial() %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d create_graph_serial() success\n", procid); }
  return 0;
}

int create_graph(dist_graph_t* g, 
          uint64_t n_global, uint64_t m_global, 
          uint64_t n_local, uint64_t m_local,
          uint64_t* local_offsets, uint64_t* local_adjs, 
          uint64_t* global_ids,
          int32_t* vertex_weights, int32_t* edge_weights)
{ 
  if (debug) { printf("Task %d create_graph() start\n", procid); }

  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = n_global;
  g->n_local = n_local;
  g->m = m_global;
  g->m_local = m_local;
  g->vertex_weights = NULL;
  g->edge_weights = NULL;
  g->map = (struct fast_map*)malloc(sizeof(struct fast_map));

  g->out_edges = local_adjs;
  g->out_degree_list = local_offsets;
  if (vertex_weights != NULL) 
  {
    g->vertex_weights = vertex_weights;
    g->vertex_weights_sum = 0;
    for (uint64_t i = 0; i < g->n_local; ++i)
      g->vertex_weights_sum += g->vertex_weights[i];
    MPI_Allreduce(MPI_IN_PLACE, &g->vertex_weights_sum, 1, 
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  }
  else g->vertex_weights = NULL;
  if (edge_weights != NULL) g->edge_weights = edge_weights;
  else g->edge_weights = NULL;

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (g->local_unmap == NULL)
    throw_err("create_graph(), unable to allocate unmap", procid);

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = global_ids[i];

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d create_graph() %9.6f (s)\n", procid, elt);
  }

  if (debug) { printf("Task %d create_graph() success\n", procid); }
  return 0;
}


int create_graph_serial(dist_graph_t* g, 
          uint64_t n_global, uint64_t m_global, 
          uint64_t n_local, uint64_t m_local,
          uint64_t* local_offsets, uint64_t* local_adjs,
          int32_t* vertex_weights, int32_t* edge_weights)
{
  if (debug) { printf("Task %d create_graph_serial() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = n_global; printf("N global %lu %lu\n", n_global, g->n);
  g->n_local = n_local;
  g->n_offset = 0;
  g->n_ghost = 0;
  g->m = m_global;
  g->m_local = m_local;
  g->n_total = g->n_local;
  g->vertex_weights = NULL;
  g->edge_weights = NULL;
  g->map = (struct fast_map*)malloc(sizeof(struct fast_map));

  g->out_edges = local_adjs;
  g->out_degree_list = local_offsets;
  if (vertex_weights != NULL) 
  {
    g->vertex_weights = vertex_weights;
    g->vertex_weights_sum = 0;
    for (uint64_t i = 0; i < g->n; ++i)
      g->vertex_weights_sum += g->vertex_weights[i];
  }
  else g->vertex_weights = NULL;
  if (edge_weights != NULL) g->edge_weights = edge_weights;
  else g->edge_weights = NULL; 

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));  
  if (g->local_unmap == NULL)
    throw_err("create_graph_serial(), unable to allocate unmap\n", procid);

  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = i + g->n_offset;

  init_map_nohash(g->map, g->n);
  for (uint64_t i = 0; i < g->n_local; ++i)
    set_value_uq(g->map, i, i);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d create_graph_serial() %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d create_graph_serial() success\n", procid); }
  return 0;
}


int clear_graph(dist_graph_t *g)
{
  if (debug) { printf("Task %d clear_graph() start\n", procid); }

  free(g->out_edges);
  free(g->out_degree_list);
  free(g->ghost_degrees);
  free(g->local_unmap);
  free(g->ghost_unmap);
  free(g->ghost_tasks);
  clear_map(g->map);
  free(g->map);

  if (g->vertex_weights != NULL) free(g->vertex_weights);
  if (g->edge_weights != NULL) free(g->edge_weights);

  if (debug) { printf("Task %d clear_graph() success\n", procid); }
  return 0;
} 


int relabel_edges(dist_graph_t *g)
{
  relabel_edges(g, NULL);
  return 0;
}


int relabel_edges(dist_graph_t *g, uint64_t* vert_dist)
{
  if (debug) { printf("Task %d relabel_edges() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t cur_label = g->n_local;
  uint64_t total_edges = g->m_local + g->n_local;

  if (total_edges*2 < g->n)
    init_map_nohash(g->map, g->n);
  else
    init_map(g->map, total_edges*2);

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert = g->local_unmap[i];
    set_value(g->map, vert, i);
  }

  for (uint64_t i = 0; i < g->m_local; ++i)
  {
    uint64_t out = g->out_edges[i];
    uint64_t val = get_value(g->map, out);
    if (val == NULL_KEY)
    {
      set_value_uq(g->map, out, cur_label);
      g->out_edges[i] = cur_label++;
    }
    else        
      g->out_edges[i] = val;
  }

  g->n_ghost = g->map->num_unique;
  g->n_total = g->n_ghost + g->n_local;

  if (debug)
    printf("Task %d, n_ghost %lu\n", procid, g->n_ghost);

  g->ghost_unmap = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  g->ghost_tasks = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  if (g->ghost_unmap == NULL || g->ghost_tasks == NULL)
    throw_err("relabel_edges(), unable to allocate ghost unmaps", procid);

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_ghost; ++i)
  {
    uint64_t cur_index = get_value(g->map, g->map->unique_keys[i]);

    cur_index -= g->n_local;
    g->ghost_unmap[cur_index] = g->map->unique_keys[i];
  }

  if (vert_dist == NULL)
  {
    uint64_t n_per_rank = g->n / (uint64_t)nprocs + 1;  

#pragma omp parallel for
    for (uint64_t i = 0; i < g->n_ghost; ++i)
      g->ghost_tasks[i] = g->ghost_unmap[i] / n_per_rank;
  }
  else
  {
 #pragma omp parallel for
    for (uint64_t i = 0; i < g->n_ghost; ++i)
    {   
      uint64_t global_id = g->ghost_unmap[i];
      int32_t rank = highest_less_than(vert_dist, global_id);
      g->ghost_tasks[i] = rank;
    }
  }

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf(" Task %d relabel_edges() %9.6f (s)\n", procid, elt); 
  }

  if (debug) { printf("Task %d relabel_edges() success\n", procid); }
  return 0;
}





int get_max_degree_vert(dist_graph_t *g)
{ 
  if (debug) { printf("Task %d get_max_degree_vert() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t my_max_degree = 0;
  uint64_t my_max_vert = -1;
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t this_degree = out_degree(g, i);
    if (this_degree > my_max_degree)
    {
      my_max_degree = this_degree;
      my_max_vert = g->local_unmap[i];
    }
  }

  uint64_t max_degree;
  uint64_t max_vert;

  MPI_Allreduce(&my_max_degree, &max_degree, 1, MPI_UINT64_T,
                MPI_MAX, MPI_COMM_WORLD);
  if (my_max_degree == max_degree)
    max_vert = my_max_vert;
  else
    max_vert = NULL_KEY;
  MPI_Allreduce(MPI_IN_PLACE, &max_vert, 1, MPI_UINT64_T,
                MPI_MIN, MPI_COMM_WORLD);

  g->max_degree_vert = max_vert;
  g->max_degree = max_degree;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d, max_degree %lu, max_vert %lu, %f (s)\n", 
           procid, max_degree, max_vert, elt);
  }

  if (debug) { printf("Task %d get_max_degree_vert() success\n", procid); }
  return 0;
}


int get_ghost_degrees(dist_graph_t* g)
{
  mpi_data_t comm;
  queue_data_t q;
  init_comm_data(&comm);
  init_queue_data(g, &q);

  get_ghost_degrees(g, &comm, &q);

  clear_comm_data(&comm);
  clear_queue_data(&q);

  return 0;
}


int get_ghost_degrees(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q)
{
  if (debug) { printf("Task %d get_ghost_degrees() start\n", procid); }

  g->ghost_degrees = (uint64_t*)malloc(g->n_ghost*(sizeof(uint64_t)));
  if (g->ghost_degrees == NULL)
    throw_err("get_ghost_degrees(), unable to allocate ghost degrees\n", procid);

  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

#pragma omp parallel 
{
  thread_queue_t tq;
  thread_comm_t tc;
  init_thread_queue(&tq);
  init_thread_comm(&tc);

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    update_sendcounts_thread(g, &tc, i);

  for (int32_t i = 0; i < nprocs; ++i)
  {
#pragma omp atomic
    comm->sendcounts_temp[i] += tc.sendcounts_thread[i];

    tc.sendcounts_thread[i] = 0;
  }

#pragma omp barrier

#pragma omp single
{
  init_sendbuf_vid_data(comm);    
}

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    update_vid_data_queues(g, &tc, comm, i, 
                           (out_degree(g, i)));

  empty_vid_data(&tc, comm);
#pragma omp barrier

#pragma omp single
{
  exchange_vert_data(g, comm, q);
} // end single

#pragma omp for
  for (uint64_t i = 0; i < comm->total_recv; ++i)
  {
    uint64_t index = get_value(g->map, comm->recvbuf_vert[i]);
    assert(index >= g->n_local);
    assert(index < g->n_total);
    g->ghost_degrees[index - g->n_local] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
}

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
} // end parallel


  if (debug) { printf("Task %d get_ghost_degrees() success\n", procid); }

  return 0;
}

