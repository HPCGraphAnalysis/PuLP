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
#include <assert.h>
#include <math.h>

#include "pulp_init.h"
#include "util.h"
#include "comms.h"
#include "pulp_util.h"
#include "pulp_data.h"

extern int procid, nprocs;
extern int seed;
extern bool verbose, debug, verify;

void pulp_init_rand(
  dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, pulp_data_t* pulp) 
{
  if (debug) { printf("Task %d pulp_rand_init() start\n", procid); }

  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

#pragma omp parallel 
{
  thread_queue_t tq;
  thread_comm_t tc;
  init_thread_queue(&tq);
  init_thread_comm(&tc);

  xs1024star_t xs;
  xs1024star_seed((uint64_t)seed + omp_get_thread_num(), &xs);

#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
    pulp->local_parts[i] = xs1024star_next(&xs) % pulp->num_parts;

#pragma omp for
  for (uint64_t i = g->n_local; i < g->n_total; ++i)
    pulp->local_parts[i] = -1;

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
    update_vid_data_queues(g, &tc, comm, i, pulp->local_parts[i]);

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
    pulp->local_parts[index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
}

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
} // end parallel

  //part_eval(g, pulp);

  if (debug) { printf("Task %d pulp_rand_init() success\n", procid); }
}

void pulp_init_block(
  dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, pulp_data_t* pulp) 
{
  if (debug) { printf("Task %d pulp_block_init() start\n", procid); }

  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  uint64_t num_per_part = g->n / (uint64_t)pulp->num_parts + 1;

#pragma omp parallel 
{
  thread_queue_t tq;
  thread_comm_t tc;
  init_thread_queue(&tq);
  init_thread_comm(&tc);

#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t gid = g->local_unmap[i];
    int32_t part_assignment = (int32_t)(gid / num_per_part);
    pulp->local_parts[i] = part_assignment;
    assert(part_assignment < pulp->num_parts);
  }

#pragma omp for
  for (uint64_t i = g->n_local; i < g->n_total; ++i)
    pulp->local_parts[i] = -1;

#pragma omp for schedule(guided)  nowait 
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
    update_vid_data_queues(g, &tc, comm, i, pulp->local_parts[i]);

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
    pulp->local_parts[index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
}

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);

} // end parallel

  if (debug) { printf("Task %d pulp_block_init() success\n", procid); }
}


void pulp_init_bfs_pull(
  dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, pulp_data_t* pulp)
{

#pragma omp parallel
{
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    pulp->local_parts[i] = -1;
}

  uint64_t* roots = (uint64_t*)malloc(pulp->num_parts*sizeof(uint64_t));
  if (procid == 0)
  {    
    xs1024star_t xs;
    xs1024star_seed((uint64_t)seed, &xs);
    
    for (int32_t i = 0; i < pulp->num_parts; ++i)
      roots[i] = xs1024star_next(&xs) % g->n;

    quicksort_inc(roots, 0, (int64_t)pulp->num_parts-1);   

    for (int32_t i = 1; i < pulp->num_parts; ++i)
      if (roots[i] <= roots[i-1])
        roots[i] = roots[i-1] + 1;

    if (debug) {
      printf("BFS Init Roots ");
      for (int32_t i = 0; i < pulp->num_parts; ++i)
        printf("%lu ", roots[i]);
      printf("\n");
    }
  }

  MPI_Bcast(roots, pulp->num_parts, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  for (int32_t i = 0; i < pulp->num_parts; ++i)
  {
    uint64_t root = roots[i];
    uint64_t root_index = get_value(g->map, root);

    if (root_index != NULL_KEY)
    {
      if (debug) printf("Task %d initialize root %lu index %lu as %d\n", procid, root, root_index, i);

      pulp->local_parts[root_index] = i;
    }
  }
  free(roots);


  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts[i] = 0;

  comm->global_queue_size = 1;
  uint64_t temp_send_size = 0;
  uint64_t not_initialized = 0;
#pragma omp parallel default(shared)
{
  thread_queue_t tq;
  thread_comm_t tc;
  init_thread_queue(&tq);
  init_thread_comm(&tc);

  while (comm->global_queue_size)
  {
#pragma omp for schedule(guided) nowait
    for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    {
      if (pulp->local_parts[vert_index] >= 0)
        continue;

      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        int32_t part_out = pulp->local_parts[out_index];
        if (part_out >= 0)
        {
          pulp->local_parts[vert_index] = part_out;
          break;
        }
      }

      if (pulp->local_parts[vert_index] >= 0)
      {
        add_vid_to_send(&tq, q, vert_index);
        add_vid_to_queue(&tq, q, vert_index);
      }
    }  

    empty_send(&tq, q);
    empty_queue(&tq, q);
#pragma omp barrier

    for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->send_size; ++i)
    {
      uint64_t vert_index = q->queue_send[i];
      update_sendcounts_thread(g, &tc, vert_index);
    }

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
    for (uint64_t i = 0; i < q->send_size; ++i)
    {
      uint64_t vert_index = q->queue_send[i];
      update_vid_data_queues(g, &tc, comm,
                             vert_index, pulp->local_parts[vert_index]);
    }

    empty_vid_data(&tc, comm);
#pragma omp barrier

#pragma omp single
{
    temp_send_size = q->send_size;
    exchange_vert_data(g, comm, q);
} // end single


#pragma omp for
    for (uint64_t i = 0; i < comm->total_recv; ++i)
    {
      uint64_t index = get_value(g->map, comm->recvbuf_vert[i]);
      pulp->local_parts[index] = comm->recvbuf_data[i];
    }

#pragma omp single
{   
    clear_recvbuf_vid_data(comm);

    if (debug) printf("Task %d send_size %lu global_size %li\n", 
      procid, temp_send_size, comm->global_queue_size);
}

  } // end while

  xs1024star_t xs;
  xs1024star_seed((uint64_t)seed + omp_get_thread_num(), &xs);

#pragma omp for reduction(+:not_initialized)
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    if (pulp->local_parts[i] < 0)
    {
      pulp->local_parts[i] = 
        (int32_t)(xs1024star_next(&xs) % (uint64_t)pulp->num_parts);
      add_vid_to_send(&tq, q, i);
      add_vid_to_queue(&tq, q, i);
      ++not_initialized;
    }
  }

  empty_send(&tq, q);
  empty_queue(&tq, q);
#pragma omp barrier

  for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < q->send_size; ++i)
  {
    uint64_t vert_index = q->queue_send[i];
    update_sendcounts_thread(g, &tc, vert_index);
  }

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
  for (uint64_t i = 0; i < q->send_size; ++i)
  {
    uint64_t vert_index = q->queue_send[i];
    update_vid_data_queues(g, &tc, comm,
                           vert_index, pulp->local_parts[vert_index]);
  }

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
    pulp->local_parts[index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
}

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
} // end parallel 

  if (debug) printf("Task %d pulp_init_bfs() success, not initialized %lu\n", procid, not_initialized);

  //part_eval(g, pulp);

}


//#define MAX_IMBALANCE 4

void pulp_init_bfs_max(
  dist_graph_t* g, mpi_data_t* comm, queue_data_t* q, pulp_data_t* pulp)
{
  int64_t max_part_size = int64_t( (double)(g->n / pulp->num_parts) * sqrt((double)pulp->num_parts) );

#pragma omp parallel
{
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    pulp->local_parts[i] = -1;
}

  uint64_t* roots = (uint64_t*)malloc(pulp->num_parts*sizeof(uint64_t));
  if (procid == 0)
  {    
    xs1024star_t xs;
    xs1024star_seed((uint64_t)seed, &xs);
    
    for (int32_t i = 0; i < pulp->num_parts; ++i)
      roots[i] = (uint64_t)(xs1024star_next(&xs) % g->n);

    quicksort_inc(roots, 0, (int64_t)pulp->num_parts-1);   

    for (int32_t i = 1; i < pulp->num_parts; ++i)
      if (roots[i] <= roots[i-1])
        roots[i] = roots[i-1] + 1;

    if (debug) {
      printf("BFS Init Roots ");
      for (int32_t i = 0; i < pulp->num_parts; ++i)
        printf("%lu ", roots[i]);
      printf("\n");
    }
  }

  MPI_Bcast(roots, pulp->num_parts, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  for (int32_t i = 0; i < pulp->num_parts; ++i)
  {
    pulp->part_sizes[i] = 1;
    uint64_t root = roots[i];
    uint64_t root_index = get_value(g->map, root);

    if (root_index != NULL_KEY)
    {
      if (debug) printf("Task %d initialize root %lu index %lu as %d\n", procid, root, root_index, i);

      pulp->local_parts[root_index] = i;
    }
  }
  free(roots);

  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  comm->global_queue_size = 1;
  uint64_t temp_send_size = 0;
  uint64_t not_initialized = 0;
#pragma omp parallel default(shared)
{
  thread_queue_t tq;
  thread_comm_t tc;
  init_thread_queue(&tq);
  init_thread_comm(&tc);

  while (comm->global_queue_size)
  {
#pragma omp for schedule(guided) nowait
    for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
    {
      if (pulp->local_parts[vert_index] >= 0)
        continue;

      int32_t new_part = -1;
      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        int32_t part_out = pulp->local_parts[out_index];
        if (part_out >= 0 && 
            pulp->part_sizes[part_out] + pulp->part_size_changes[part_out] < max_part_size)
        {
          pulp->local_parts[vert_index] = part_out;
          new_part = part_out;
          break;
        }
      }

      if (new_part >= 0)
      {
        add_vid_to_send(&tq, q, vert_index);
        add_vid_to_queue(&tq, q, vert_index);

    #pragma omp atomic
        ++pulp->part_size_changes[new_part];
      }
    }  

    empty_send(&tq, q);
    empty_queue(&tq, q);
#pragma omp barrier

    for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->send_size; ++i)
    {
      uint64_t vert_index = q->queue_send[i];
      update_sendcounts_thread(g, &tc, vert_index);
    }

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
    for (uint64_t i = 0; i < q->send_size; ++i)
    {
      uint64_t vert_index = q->queue_send[i];
      update_vid_data_queues(g, &tc, comm,
                             vert_index, pulp->local_parts[vert_index]);
    }

    empty_vid_data(&tc, comm);
#pragma omp barrier

#pragma omp single
{
    temp_send_size = q->send_size;
    exchange_vert_data(g, comm, q);
} // end single


#pragma omp for
    for (uint64_t i = 0; i < comm->total_recv; ++i)
    {
      uint64_t index = get_value(g->map, comm->recvbuf_vert[i]);
      pulp->local_parts[index] = comm->recvbuf_data[i];
    }

#pragma omp single
{   
    clear_recvbuf_vid_data(comm);

    MPI_Allreduce(MPI_IN_PLACE, pulp->part_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      pulp->part_sizes[p] += pulp->part_size_changes[p];
      pulp->part_size_changes[p] = 0;
    }
   
    if (debug) printf("Task %d send_size %lu global_size %lu\n", 
      procid, temp_send_size, comm->global_queue_size);
}

  } // end while
  
  xs1024star_t xs;
  xs1024star_seed((uint64_t)seed + omp_get_thread_num(), &xs);

#pragma omp for reduction(+:not_initialized)
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    if (pulp->local_parts[i] < 0)
    {
      pulp->local_parts[i] = 
        (int32_t)(xs1024star_next(&xs) % (uint64_t)pulp->num_parts);
      add_vid_to_send(&tq, q, i);
      add_vid_to_queue(&tq, q, i);
      ++not_initialized;
    }
  }

  empty_send(&tq, q);
  empty_queue(&tq, q);
#pragma omp barrier

  for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < q->send_size; ++i)
  {
    uint64_t vert_index = q->queue_send[i];
    update_sendcounts_thread(g, &tc, vert_index);
  }

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
  for (uint64_t i = 0; i < q->send_size; ++i)
  {
    uint64_t vert_index = q->queue_send[i];
    update_vid_data_queues(g, &tc, comm,
                           vert_index, pulp->local_parts[vert_index]);
  }

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
    pulp->local_parts[index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
}

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
} // end parallel 

  if (debug) printf("Task %d pulp_init_bfs() success, not initialized %lu\n", procid, not_initialized);

  //part_eval(g, pulp);
  //update_pulp_data(g, pulp);

}


#define MIN_SIZE 0.25

void pulp_init_label_prop_weighted(dist_graph_t* g, 
  mpi_data_t* comm, queue_data_t* q, pulp_data_t* pulp,
  uint64_t lp_num_iter) 
{
  if (debug) { printf("Task %d pulp_init_label_prop_weighted() start\n", procid); }

  bool has_vwgts = (g->vertex_weights != NULL);
  bool has_ewgts = (g->edge_weights != NULL);

  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  double min_size = pulp->avg_size * MIN_SIZE;
  double multiplier = (double) nprocs;

#pragma omp parallel 
{
  thread_queue_t tq;
  thread_comm_t tc;
  thread_pulp_t tp;
  init_thread_queue(&tq);
  init_thread_comm(&tc);
  init_thread_pulp(&tp, pulp);

  xs1024star_t xs;
  xs1024star_seed((uint64_t)(seed + omp_get_thread_num()), &xs);

#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
    pulp->local_parts[i] = 
      (int32_t)(xs1024star_next(&xs) % (uint64_t)pulp->num_parts);

#pragma omp for
  for (uint64_t i = g->n_local; i < g->n_total; ++i)
    pulp->local_parts[i] = -1;

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
    update_vid_data_queues(g, &tc, comm, i, pulp->local_parts[i]);

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
    pulp->local_parts[index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
  update_pulp_data_weighted(g, pulp);
}

for (uint64_t cur_iter = 0; cur_iter < lp_num_iter; ++cur_iter)
{

#pragma omp for schedule(guided)  nowait
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
  {
    int32_t part = pulp->local_parts[vert_index];
    int32_t vert_weight = 1;
    if (has_vwgts) vert_weight = g->vertex_weights[vert_index];

    for (int32_t p = 0; p < pulp->num_parts; ++p)
      tp.part_counts[p] = 0.0;

    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    int32_t* weights = out_weights(g, vert_index);
    for (uint64_t j = 0; j < out_degree; ++j)
    {
      uint64_t out_index = outs[j];
      int32_t part_out = pulp->local_parts[out_index];
      double weight_out = 1.0;
      if (has_ewgts) weight_out = (double)weights[j];
      tp.part_counts[part_out] += weight_out;
    }

    int32_t max_part = part;
    double max_val = -1.0;
    uint64_t num_max = 0;
    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      if (tp.part_counts[p] == max_val)
      {
        tp.part_counts[num_max++] = (double)p;
      }
      else if (tp.part_counts[p] > max_val)
      {
        max_val = tp.part_counts[p];
        max_part = p;
        num_max = 0;
        tp.part_counts[num_max++] = (double)p;
      }
    }      

    if (num_max > 1)
      max_part = 
        (int32_t)tp.part_counts[(xs1024star_next(&xs) % num_max)];

    if (max_part != part)
    {
      int64_t new_size = (int64_t)pulp->avg_size;

      pulp->part_size_changes[part] - (int64_t)vert_weight > 0 ?
        new_size = pulp->part_sizes[part] + pulp->part_size_changes[part] - vert_weight :
        new_size = (int64_t)((double)pulp->part_sizes[part] + multiplier * pulp->part_size_changes[part] - vert_weight);

      if (new_size > (int64_t)min_size)
      {
    #pragma omp atomic
        pulp->part_size_changes[part] -= vert_weight;
    #pragma omp atomic
        pulp->part_size_changes[max_part] += vert_weight;

        pulp->local_parts[vert_index] = max_part;
        add_vid_to_send(&tq, q, vert_index);
      }
    }
  }  

  empty_send(&tq, q);
#pragma omp barrier

  for (int32_t i = 0; i < nprocs; ++i)
    tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < q->send_size; ++i)
  {
    uint64_t vert_index = q->queue_send[i];
    update_sendcounts_thread(g, &tc, vert_index);
  }

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
  for (uint64_t i = 0; i < q->send_size; ++i)
  {
    uint64_t vert_index = q->queue_send[i];
    update_vid_data_queues(g, &tc, comm,
                           vert_index, pulp->local_parts[vert_index]);
  }

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
    pulp->local_parts[index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);

  MPI_Allreduce(MPI_IN_PLACE, pulp->part_size_changes, pulp->num_parts, 
  MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  for (int32_t p = 0; p < pulp->num_parts; ++p)
  {
    pulp->part_sizes[p] += pulp->part_size_changes[p];
    pulp->part_size_changes[p] = 0;
  }
}


} // end for iter loop

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
} // end parallel

  //update_pulp_data_weighted(g, pulp);

  if (debug) { printf("Task %d pulp_init_label_prop_weighted() success\n", procid); }
}




void pulp_init_label_prop(dist_graph_t* g, 
  mpi_data_t* comm, queue_data_t* q, pulp_data_t* pulp,
  uint64_t lp_num_iter) 
{
  if (debug) { printf("Task %d pulp_init_label_prop() start\n", procid); }

  q->send_size = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  double min_size = pulp->avg_size * MIN_SIZE;
  double multiplier = (double) nprocs;

#pragma omp parallel 
{
  thread_queue_t tq;
  thread_comm_t tc;
  thread_pulp_t tp;
  init_thread_queue(&tq);
  init_thread_comm(&tc);
  init_thread_pulp(&tp, pulp);

  xs1024star_t xs;
  xs1024star_seed((uint64_t)(seed + omp_get_thread_num()), &xs);

#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
    pulp->local_parts[i] = 
      (int32_t)(xs1024star_next(&xs) % (uint64_t)pulp->num_parts);

#pragma omp for
  for (uint64_t i = g->n_local; i < g->n_total; ++i)
    pulp->local_parts[i] = -1;

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
    update_vid_data_queues(g, &tc, comm, i, pulp->local_parts[i]);

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
    pulp->local_parts[index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);
  update_pulp_data(g, pulp);
}

for (uint64_t cur_iter = 0; cur_iter < lp_num_iter; ++cur_iter)
{

#pragma omp for schedule(guided) nowait
  for (uint64_t vert_index = 0; vert_index < g->n_local; ++vert_index)
  {
    int32_t part = pulp->local_parts[vert_index];

    for (int32_t p = 0; p < pulp->num_parts; ++p)
      tp.part_counts[p] = 0.0;

    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    for (uint64_t j = 0; j < out_degree; ++j)
    {
      uint64_t out_index = outs[j];
      int32_t part_out = pulp->local_parts[out_index];
      tp.part_counts[part_out] += 1.0;
    }

    int32_t max_part = part;
    double max_val = -1.0;
    uint64_t num_max = 0;
    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      if (tp.part_counts[p] == max_val)
      {
        tp.part_counts[num_max++] = (double)p;
      }
      else if (tp.part_counts[p] > max_val)
      {
        max_val = tp.part_counts[p];
        max_part = p;
        num_max = 0;
        tp.part_counts[num_max++] = (double)p;
      }
    }      

    if (num_max > 1)
      max_part = 
        (int32_t)tp.part_counts[(xs1024star_next(&xs) % num_max)];

    if (max_part != part)
    {
      int64_t new_size = (int64_t)pulp->avg_size;

      pulp->part_size_changes[part] - 1 > 0 ?
        new_size = pulp->part_sizes[part] + pulp->part_size_changes[part] - 1 :
        new_size = (int64_t)((double)pulp->part_sizes[part] + multiplier * pulp->part_size_changes[part] - 1);

      if (new_size > (int64_t)min_size)
      {
    #pragma omp atomic
        --pulp->part_size_changes[part];
    #pragma omp atomic
        ++pulp->part_size_changes[max_part];

        pulp->local_parts[vert_index] = max_part;
        add_vid_to_send(&tq, q, vert_index);
      }
    }
  }  

  empty_send(&tq, q);
#pragma omp barrier

  for (int32_t i = 0; i < nprocs; ++i)
    tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < q->send_size; ++i)
  {
    uint64_t vert_index = q->queue_send[i];
    update_sendcounts_thread(g, &tc, vert_index);
  }

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
  for (uint64_t i = 0; i < q->send_size; ++i)
  {
    uint64_t vert_index = q->queue_send[i];
    update_vid_data_queues(g, &tc, comm,
                           vert_index, pulp->local_parts[vert_index]);
  }

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
    pulp->local_parts[index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_recvbuf_vid_data(comm);

  MPI_Allreduce(MPI_IN_PLACE, pulp->part_size_changes, pulp->num_parts, 
  MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  for (int32_t p = 0; p < pulp->num_parts; ++p)
  {
    pulp->part_sizes[p] += pulp->part_size_changes[p];
    pulp->part_size_changes[p] = 0;
  }
}


} // end for iter loop

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
} // end parallel

  //update_pulp_data(g, pulp);

  if (debug) { printf("Task %d pulp_init_label_prop() success\n", procid); }
}
