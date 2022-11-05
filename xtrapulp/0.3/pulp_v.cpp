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

#include "xtrapulp.h"
#include "util.h"
#include "comms.h"
#include "pulp_data.h"
#include "pulp_util.h"
#include "pulp_v.h"

extern int procid, nprocs;
extern int seed;
extern bool verbose, debug, verify;
extern float X,Y;

int pulp_v(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
            pulp_data_t *pulp,            
            uint64_t outer_iter, 
            uint64_t balance_iter, uint64_t refine_iter, 
            double vert_balance, double edge_balance)
{ 
  if (debug) { printf("Task %d pulp_v() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  q->queue_size = 0;
  q->next_size = 0;
  q->send_size = 0;

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  double tot_iter = 
    (double)(outer_iter*(refine_iter+balance_iter));
  double cur_iter = 0.0;
  double multiplier = (double)nprocs*( (X - Y)*(cur_iter/tot_iter) + Y );
  //double running_bal = pulp->max_v;
  //double running_cut = (double)pulp->cut_size;
  //uint64_t num_tries = 0;


  uint64_t num_swapped_1 = 0;
  uint64_t num_swapped_2 = 0;
  comm->global_queue_size = 1;
#pragma omp parallel default(shared)
{
  thread_queue_t tq;
  thread_comm_t tc;
  thread_pulp_t tp;
  init_thread_queue(&tq);
  init_thread_comm(&tc);
  init_thread_pulp(&tp, pulp);
  xs1024star_t xs;
  xs1024star_seed((uint64_t)(seed + omp_get_thread_num()), &xs);

for (uint64_t cur_outer_iter = 0; cur_outer_iter < outer_iter; ++cur_outer_iter)
{

#pragma omp single
{
  if (procid == 0 && debug) {
    printf("EVAL pulp_v bal ------------------------------\n");
    part_eval(g, pulp);
  }
  update_pulp_data(g, pulp);
  num_swapped_1 = 0;
}

  for (uint64_t cur_bal_iter = 0; cur_bal_iter < balance_iter; ++cur_bal_iter)
  {
    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      tp.part_vert_weights[p] = 
          vert_balance * pulp->avg_vert_size / 
          ((double)pulp->part_vert_sizes[p] + multiplier*(double)pulp->part_vert_size_changes[p]) - 1.0;
      if (tp.part_vert_weights[p] < 0.0)
        tp.part_vert_weights[p] = 0.0;
    }

#pragma omp for schedule(guided) reduction(+:num_swapped_1) nowait
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
        if (out_index >= g->n_local)
        {
          tp.part_counts[part_out] += g->ghost_degrees[out_index - g->n_local];
        }
        else 
        { 
          tp.part_counts[part_out] += out_degree(g, out_index);
        }
      }

      int32_t max_part = part;
      double max_val = 0.0;
      uint64_t num_max = 0;
      for (int32_t p = 0; p < pulp->num_parts; ++p)
      {
        if (tp.part_vert_weights[p] > 0.0)
          tp.part_counts[p] *= tp.part_vert_weights[p];
        else
          tp.part_counts[p] = 0.0;

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
        ++num_swapped_1;
    #pragma omp atomic
        --pulp->part_vert_size_changes[part];
    #pragma omp atomic
        ++pulp->part_vert_size_changes[max_part];
        
        tp.part_vert_weights[part] = 
          vert_balance * pulp->avg_vert_size / 
          ((double)pulp->part_vert_sizes[part] + multiplier*(double)pulp->part_vert_size_changes[part]) - 1.0;
        tp.part_vert_weights[max_part] = 
          vert_balance * pulp->avg_vert_size / 
          ((double)pulp->part_vert_sizes[max_part] + multiplier*(double)pulp->part_vert_size_changes[max_part]) - 1.0;
        
        if (tp.part_vert_weights[part] < 0.0)
          tp.part_vert_weights[part] = 0.0;
        if (tp.part_vert_weights[max_part] < 0.0)
          tp.part_vert_weights[max_part] = 0.0;

        pulp->local_parts[vert_index] = max_part;
        add_vid_to_send(&tq, q, vert_index);
        //add_vid_to_queue(&tq, q, vert_index);
      }
    }  

    empty_send(&tq, q);
    //empty_queue(&tq, q);
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

    MPI_Allreduce(MPI_IN_PLACE, pulp->part_vert_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      pulp->part_vert_sizes[p] += pulp->part_vert_size_changes[p];
      pulp->part_vert_size_changes[p] = 0;
    }

    cur_iter += 1.0;
    //multiplier = (double)pulp->num_parts*(1.0-(double)cur_iter/(double)tot_iter)+1.0*pulp->num_parts*(double)cur_iter/((double)tot_iter*2.0);
    multiplier = (double)nprocs*( (X - Y)*(cur_iter/tot_iter) + Y );

    //int32_t* tmp = pulp->local_parts;
    //pulp->local_parts = pulp->local_parts_next;
    //pulp->local_parts_next = tmp;
    if (debug) printf("Task %d num_swapped_1 %lu \n", procid, num_swapped_1);
    num_swapped_1 = 0;
}

  }// end balance loop


#pragma omp single
{  
  if (procid == 0 && debug) {
    printf("EVAL pulp_v ref ------------------------------\n");
    part_eval(g, pulp);
  }
  update_pulp_data(g, pulp);
  num_swapped_2 = 0;
}


  for (uint64_t cur_ref_iter = 0; cur_ref_iter < refine_iter; ++cur_ref_iter)
  {

#pragma omp for schedule(guided) reduction(+:num_swapped_2) nowait
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
      double max_val = 0.0;
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
        int64_t new_size = (int64_t)pulp->avg_vert_size;

        pulp->part_vert_size_changes[max_part] + 1 < 0 ? 
          new_size = pulp->part_vert_sizes[max_part] + pulp->part_vert_size_changes[max_part] + 1 :
          new_size = (int64_t)((double)pulp->part_vert_sizes[max_part] + multiplier*(double)pulp->part_vert_size_changes[max_part] + 1.0);

        if (new_size < (int64_t)(pulp->avg_vert_size*vert_balance))
        {
          ++num_swapped_2;
      #pragma omp atomic
          --pulp->part_vert_size_changes[part];
      #pragma omp atomic
          ++pulp->part_vert_size_changes[max_part];

          pulp->local_parts[vert_index] = max_part;
          add_vid_to_send(&tq, q, vert_index);
          //add_vid_to_queue(&tq, q, vert_index);
        }
      }
    }  

    empty_send(&tq, q);
    //empty_queue(&tq, q);
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

    MPI_Allreduce(MPI_IN_PLACE, pulp->part_vert_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      pulp->part_vert_sizes[p] += pulp->part_vert_size_changes[p];
      pulp->part_vert_size_changes[p] = 0;
    }

    cur_iter += 1.0;
    //multiplier = (double)pulp->num_parts*(1.0-(double)cur_iter/(double)tot_iter)+1.0*pulp->num_parts*(double)cur_iter/((double)tot_iter*2.0);
    multiplier = (double)nprocs*( (X - Y)*(cur_iter/tot_iter) + Y );

    if (debug) printf("Task %d num_swapped_2 %lu \n", procid, num_swapped_2);
    num_swapped_2 = 0;
}

  } // end refine iter

  /*if (cur_outer_iter + 1 == outer_iter)
  {
#pragma omp single
{
    update_pulp_data(g, pulp);

    if ( ( (pulp->max_v > vert_balance*BAL_CUTOFF &&
            pulp->max_v*BAL_CHANGE < running_bal) ||
            (double)pulp->cut_size < running_cut*CUT_CHANGE ) &&
          num_tries < 3)
    {
      outer_iter += 1;
      tot_iter += (double)(balance_iter + refine_iter);
      ++num_tries;
      running_cut = (double)pulp->cut_size;
      running_bal = pulp->max_v;
      multiplier = (double)nprocs*( (X - Y)*(cur_iter/tot_iter) + Y );
    }
}
  }*/

} // end outer loop

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
  clear_thread_pulp(&tp);
} // end parallel

  //part_eval(g, pulp);
  //update_pulp_data(g, pulp);
  
  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d, pulp_v() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d pulp_v() success\n", procid); }

  return 0;
}
