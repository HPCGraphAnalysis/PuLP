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

#include "xtrapulp.h"
#include "util.h"
#include "comms.h"
#include "pulp_data.h"
#include "pulp_util.h"
#include "pulp_vec.h"

//#define X 1.0
//#define Y 0.25
#define CUT_CHANGE 0.95
#define BAL_CHANGE 0.95
#define BAL_CUTOFF 1.05

extern int procid, nprocs;
extern int seed;
extern bool verbose, debug, verify;
extern float X,Y;

int pulp_vec(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
            pulp_data_t *pulp,            
            uint64_t outer_iter, 
            uint64_t balance_iter, uint64_t refine_iter, 
            double vert_balance, double edge_balance)
{ 
  if (debug) { printf("Task %d pulp_vec() start\n", procid); }
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

  pulp->avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;
  pulp->max_v = 0.0;
  pulp->max_e = 0.0;
  pulp->max_c = 0.0;
  pulp->max_cut = 0;
  for (int p = 0; p < pulp->num_parts; ++p)
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
  if (pulp->max_e < edge_balance)
  {
    pulp->max_e = edge_balance;
    pulp->weight_exponent_e = 1.0;
    pulp->weight_exponent_c *= pulp->max_c;
  }
  else
  {
    pulp->weight_exponent_e *= pulp->max_e / edge_balance;
    pulp->weight_exponent_c = 1.0;
  }

  double tot_iter = 
    (double)(outer_iter*(refine_iter+balance_iter));
  double cur_iter = 0.0;    
  double multiplier = (double)nprocs*( (X - Y)*(cur_iter/tot_iter) + Y );
  //double running_bal = pulp->max_e;
  //double running_cut = (double)pulp->cut_size;
  //uint64_t num_tries = 3;

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
  //if (procid == 0) printf("EVAL ec VEC EB ------------------------------\n");
  //part_eval(g, pulp);
  update_pulp_data(g, pulp);
  num_swapped_1 = 0;
}

  for (uint64_t cur_bal_iter = 0; cur_bal_iter < balance_iter; ++cur_bal_iter)
  {

    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      tp.part_weights[p] = vert_balance * pulp->avg_size / (double)pulp->part_sizes[p] - 1.0;
      tp.part_edge_weights[p] = pulp->max_e * pulp->avg_edge_size / (double)pulp->part_edge_sizes[p] - 1.0;
      tp.part_cut_weights[p] = pulp->max_c * pulp->avg_cut_size / (double)pulp->part_cut_sizes[p] - 1.0;
      if (tp.part_weights[p] < 0.0)
        tp.part_weights[p] = 0.0;
      if (tp.part_edge_weights[p] < 0.0)
        tp.part_edge_weights[p] = 0.0;
      if (tp.part_cut_weights[p] < 0.0)
        tp.part_cut_weights[p] = 0.0;
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
        tp.part_counts[part_out] += 1.0;
      }

      int32_t max_part = part;
      double max_val = 0.0;
      uint64_t num_max = 0;
      int64_t max_count = 0;
      int64_t part_count = (int64_t)tp.part_counts[part];
      for (int32_t p = 0; p < pulp->num_parts; ++p)
      {
        int64_t count_init = (int64_t)tp.part_counts[p];
        if (tp.part_weights[p] > 0.0 && tp.part_edge_weights[p] > 0.0 && tp.part_cut_weights[p] > 0.0)
          tp.part_counts[p] *= (tp.part_edge_weights[p]*pulp->weight_exponent_e * tp.part_cut_weights[p]*pulp->weight_exponent_c);
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
          max_count = count_init;
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
        int64_t diff_part = 2*part_count - (int64_t)out_degree;
        int64_t diff_max_part = (int64_t)(out_degree) - 2*max_count;
        int64_t diff_cut = part_count - max_count;  

    #pragma omp atomic
        pulp->cut_size_change += diff_cut;
    #pragma omp atomic
        pulp->part_cut_size_changes[part] += diff_part;
    #pragma omp atomic
        pulp->part_cut_size_changes[max_part] += diff_max_part;
    #pragma omp atomic
        --pulp->part_size_changes[part];
    #pragma omp atomic
        ++pulp->part_size_changes[max_part];
    #pragma omp atomic
        pulp->part_edge_size_changes[part] -= (int64_t)out_degree;
    #pragma omp atomic
        pulp->part_edge_size_changes[max_part] += (int64_t)out_degree;
        
        tp.part_weights[part] = 
          vert_balance * pulp->avg_size / 
          ((double)pulp->part_sizes[part] + multiplier*(double)pulp->part_size_changes[part]) - 1.0;
        tp.part_weights[max_part] = 
          vert_balance * pulp->avg_size / 
          ((double)pulp->part_sizes[max_part] + multiplier*(double)pulp->part_size_changes[max_part]) - 1.0;

        tp.part_edge_weights[part] = 
          pulp->max_e * pulp->avg_edge_size / 
          ((double)pulp->part_edge_sizes[part] + multiplier*(double)pulp->part_edge_size_changes[part]) - 1.0;
        tp.part_edge_weights[max_part] = 
          pulp->max_e * pulp->avg_edge_size / 
          ((double)pulp->part_edge_sizes[max_part] + multiplier*(double)pulp->part_edge_size_changes[max_part]) - 1.0;

        double avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;
        tp.part_cut_weights[part] = 
          pulp->max_c * avg_cut_size / 
          ((double)pulp->part_cut_sizes[part] + multiplier*(double)pulp->part_cut_size_changes[part]) - 1.0;  
        tp.part_cut_weights[max_part] = 
          pulp->max_c * avg_cut_size / 
          ((double)pulp->part_cut_sizes[max_part] + multiplier*(double)pulp->part_cut_size_changes[max_part]) - 1.0;  

        if (tp.part_weights[part] < 0.0)
          tp.part_weights[part] = 0.0;
        if (tp.part_weights[max_part] < 0.0)
          tp.part_weights[max_part] = 0.0;

        if (tp.part_edge_weights[part] < 0.0)
          tp.part_edge_weights[part] = 0.0;
        if (tp.part_edge_weights[max_part] < 0.0)
          tp.part_edge_weights[max_part] = 0.0;

        if (tp.part_cut_weights[part] < 0.0)
          tp.part_cut_weights[part] = 0.0;
        if (tp.part_cut_weights[max_part] < 0.0)
          tp.part_cut_weights[max_part] = 0.0;

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

    MPI_Allreduce(MPI_IN_PLACE, pulp->part_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pulp->part_edge_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pulp->part_cut_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &pulp->cut_size_change, 1, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    pulp->cut_size += pulp->cut_size_change;
    pulp->cut_size_change = 0;
    pulp->avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;

    pulp->max_v = 0.0;
    pulp->max_e = 0.0;
    pulp->max_c = 0.0;
    pulp->max_cut = 0;

    //if (procid == 0)
    //for (int32_t i = 0; i < pulp->num_parts; ++i)
    //  printf("p: %d, v: %li, e: %li, cut: %li\n",
    //  i, pulp->part_sizes[i], pulp->part_edge_sizes[i], pulp->part_cut_sizes[i]);
    //part_eval(g, pulp);

    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      //if (procid == 0)
      //  printf("p: %d, vc: %li, ec: %li, cc: %li\n", p, 
      //    pulp->part_size_changes[p], pulp->part_edge_size_changes[p],
      //    pulp->part_cut_size_changes[p]);

      pulp->part_sizes[p] += pulp->part_size_changes[p];
      pulp->part_edge_sizes[p] += pulp->part_edge_size_changes[p];
      pulp->part_cut_sizes[p] += pulp->part_cut_size_changes[p];
      pulp->part_size_changes[p] = 0;
      pulp->part_edge_size_changes[p] = 0;
      pulp->part_cut_size_changes[p] = 0;

      if ((double)pulp->part_sizes[p] / pulp->avg_size > pulp->max_v)
        pulp->max_v = (double)pulp->part_sizes[p] / pulp->avg_size;        
      if ((double)pulp->part_edge_sizes[p] / pulp->avg_edge_size > pulp->max_e)
        pulp->max_e = (double)pulp->part_edge_sizes[p] / pulp->avg_edge_size;
      if ((double)pulp->part_cut_sizes[p] / pulp->avg_cut_size > pulp->max_c)
        pulp->max_c = (double)pulp->part_cut_sizes[p] / pulp->avg_cut_size;
      if (pulp->part_cut_sizes[p] > pulp->max_cut)
        pulp->max_cut = pulp->part_cut_sizes[p];
    }
    if (pulp->max_e < edge_balance)
    {
      pulp->max_e = edge_balance;
      pulp->weight_exponent_e = 1.0;
      pulp->weight_exponent_c *= pulp->max_c;
    }
    else
    {
      pulp->weight_exponent_e *= pulp->max_e / edge_balance;
      pulp->weight_exponent_c = 1.0;
    }

    cur_iter += 1.0;
    //multiplier = (double)pulp->num_parts*(1.0-(double)cur_iter/(double)tot_iter)+1.0*pulp->num_parts*(double)cur_iter/((double)tot_iter*2.0);
    multiplier = (double)nprocs*( (X - Y)*(cur_iter/tot_iter) + Y );

    //int32_t* tmp = pulp->local_parts;
    //pulp->local_parts = pulp->local_parts_next;
    //pulp->local_parts_next = tmp;
    if (debug) printf("Task %d num_swapped_1 %lu\n", procid, num_swapped_1);
    num_swapped_1 = 0;
    //if (procid == 0)
    //  printf("&&& XtraPuLP, %2.3lf, %2.3lf, %2.3lf, %li, %li\n",
    //    pulp->max_v, pulp->max_e, pulp->max_c, pulp->cut_size, pulp->max_cut);
    //part_eval(g, pulp);
    //printf("\n\n\n\n");
}
  }// end balance loop


#pragma omp single
{  
  //if (procid == 0) printf("EVAL ec VEC ER ------------------------------\n");
  //part_eval(g, pulp);
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
      int64_t max_count = 0;
      int64_t part_count = (int64_t)tp.part_counts[part];
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
          max_count = (int64_t)max_val;
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
        int64_t new_edge_size = (int64_t)pulp->avg_edge_size;
        double avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;
        int64_t new_cut_size = (int64_t)avg_cut_size;
        int64_t new_max_cut_size = (int64_t)avg_cut_size;

        pulp->part_size_changes[max_part] + 1 < 0 ? 
          new_size = pulp->part_sizes[max_part] + pulp->part_size_changes[max_part] + 1 :
          new_size = (int64_t)((double)pulp->part_sizes[max_part] + multiplier*(double)pulp->part_size_changes[max_part] + 1.0);

        pulp->part_edge_size_changes[max_part] + out_degree < 0 ?
          new_edge_size = pulp->part_edge_sizes[max_part] + pulp->part_edge_size_changes[max_part] + out_degree :
          new_edge_size = (int64_t)((double)pulp->part_edge_sizes[max_part] + multiplier*(double)pulp->part_edge_size_changes[max_part] + (double)(out_degree));

        pulp->part_cut_size_changes[part] < 0 ?
          new_cut_size = pulp->part_cut_sizes[part] + pulp->part_cut_size_changes[part] + 2*part_count - out_degree :
          new_cut_size = (int64_t)((double)pulp->part_cut_sizes[part] + multiplier*(double)pulp->part_cut_size_changes[part] + 2.0*(double)part_count - (double)(out_degree));
             
        pulp->part_cut_size_changes[max_part] < 0 ?
          new_cut_size = pulp->part_cut_sizes[max_part] + pulp->part_cut_size_changes[max_part] + out_degree - 2*max_count :
          new_cut_size = (int64_t)((double)pulp->part_cut_sizes[max_part] + multiplier*(double)pulp->part_cut_size_changes[max_part] + (double)(out_degree) - 2.0*(double)max_count);


        if (new_size < (int64_t)(pulp->avg_size*vert_balance) &&
          new_edge_size < (int64_t)(pulp->avg_edge_size*pulp->max_e) &&
          new_cut_size < (int64_t)(avg_cut_size*pulp->max_c) &&
          new_max_cut_size < (int64_t)(avg_cut_size*pulp->max_c) )
        {
          ++num_swapped_2;
          int64_t diff_part = 2*part_count - (int64_t)out_degree;
          int64_t diff_max_part = (int64_t)out_degree+ - 2*max_count;
          int64_t diff_cut = part_count - max_count;  

      #pragma omp atomic
          pulp->cut_size_change += diff_cut;
      #pragma omp atomic
          pulp->part_cut_size_changes[part] += diff_part;
      #pragma omp atomic
          pulp->part_cut_size_changes[max_part] += diff_max_part;
      #pragma omp atomic
          --pulp->part_size_changes[part];
      #pragma omp atomic
          ++pulp->part_size_changes[max_part];
      #pragma omp atomic
          pulp->part_edge_size_changes[part] -= (int64_t)out_degree;
      #pragma omp atomic
          pulp->part_edge_size_changes[max_part] += (int64_t)out_degree;     

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
      //pulp->local_parts_next[index] = comm->recvbuf_data[i];
    }

#pragma omp single
{
    clear_recvbuf_vid_data(comm);

    MPI_Allreduce(MPI_IN_PLACE, pulp->part_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pulp->part_edge_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pulp->part_cut_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &pulp->cut_size_change, 1, 
        MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    pulp->cut_size += pulp->cut_size_change;
    pulp->cut_size_change = 0;
    pulp->avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;

    /*pulp->max_v = 0.0;
    pulp->max_e = 0.0;
    pulp->max_c = 0.0;*/
    pulp->max_cut = 0;
    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      pulp->part_sizes[p] += pulp->part_size_changes[p];
      pulp->part_edge_sizes[p] += pulp->part_edge_size_changes[p];
      pulp->part_cut_sizes[p] += pulp->part_cut_size_changes[p];
      pulp->part_size_changes[p] = 0;
      pulp->part_edge_size_changes[p] = 0;
      pulp->part_cut_size_changes[p] = 0;

      /*if ((double)pulp->part_sizes[p] / pulp->avg_size > pulp->max_v)
        pulp->max_v = (double)pulp->part_sizes[p] / pulp->avg_size;
      if ((double)pulp->part_edge_sizes[p] / pulp->avg_edge_size > pulp->max_e)
        pulp->max_e = (double)pulp->part_edge_sizes[p] / pulp->avg_edge_size;
      if ((double)pulp->part_cut_sizes[p] / pulp->avg_cut_size > pulp->max_c)
        pulp->max_c = (double)pulp->part_cut_sizes[p] / pulp->avg_cut_size;*/
      if (pulp->part_cut_sizes[p] > pulp->max_cut)
        pulp->max_cut = pulp->part_cut_sizes[p];
    }
    /*if (pulp->max_e < edge_balance)
    {
      pulp->max_e = edge_balance;
      pulp->weight_exponent_e = 1.0;
      pulp->weight_exponent_c *= pulp->max_c;
    }
    else
    {
      pulp->weight_exponent_e *= pulp->max_e / edge_balance;
      pulp->weight_exponent_c = 1.0;
    }*/

    cur_iter += 1.0;
    //multiplier = (double)pulp->num_parts*(1.0-(double)cur_iter/(double)tot_iter)+1.0*pulp->num_parts*(double)cur_iter/((double)tot_iter*2.0);
    multiplier = (double)nprocs*( (X - Y)*(cur_iter/tot_iter) + Y );
    //printf("mult %9.6lf\n", multiplier);

    //int32_t* tmp = pulp->local_parts;
    //pulp->local_parts = pulp->local_parts_next;
    //pulp->local_parts_next = tmp;
    if (debug) printf("Task %d num_swapped_2 %lu \n", procid, num_swapped_2);
    num_swapped_2 = 0;
    //printf("%d &&& XtraPuLP, %2.3lf, %2.3lf, %2.3lf, %li, %li\n", procid,
    //    pulp->max_v, pulp->max_e, pulp->max_c, pulp->cut_size, pulp->max_cut);
    //part_eval(g, pulp);
}

  } // end refine iter

  /*if (cur_outer_iter + 1 == outer_iter)
  {
#pragma omp single
{
    update_pulp_data(g, pulp);

    if ( ( (pulp->max_e > edge_balance*BAL_CUTOFF &&
            pulp->max_e*BAL_CHANGE < running_bal) ||
            (double)pulp->cut_size < running_cut*CUT_CHANGE ) &&
          num_tries < 3)
    {
      outer_iter += 1;
      tot_iter += (double)(balance_iter + refine_iter);
      ++num_tries;
      running_cut = (double)pulp->cut_size;
      running_bal = pulp->max_e;
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
    printf("Task %d, pulp_vec() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d pulp_vec() success\n", procid); }

  return 0;
}




int pulp_vec_weighted(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
            pulp_data_t *pulp,            
            uint64_t outer_iter, 
            uint64_t balance_iter, uint64_t refine_iter, 
            double vert_balance, double edge_balance)
{ 
  if (debug) { printf("Task %d pulp_vec_weighted() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  bool has_vwgts = (g->vertex_weights != NULL);
  bool has_ewgts = (g->edge_weights != NULL);

  q->queue_size = 0;
  q->next_size = 0;
  q->send_size = 0;

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  pulp->avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;
  pulp->max_v = 0.0;
  pulp->max_e = 0.0;
  pulp->max_c = 0.0;
  pulp->max_cut = 0;
  for (int p = 0; p < pulp->num_parts; ++p)
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
  if (pulp->max_e < edge_balance)
  {
    pulp->max_e = edge_balance;
    pulp->weight_exponent_e = 1.0;
    pulp->weight_exponent_c *= pulp->max_c;
  }
  else
  {
    pulp->weight_exponent_e *= pulp->max_e / edge_balance;
    pulp->weight_exponent_c = 1.0;
  }

  double tot_iter = 
    (double)(outer_iter*(refine_iter+balance_iter));
  double cur_iter = 0.0;    
  double multiplier = (double)nprocs*( (X - Y)*(cur_iter/tot_iter) + Y );
  //double running_bal = pulp->max_e;
  //double running_cut = (double)pulp->cut_size;
  //uint64_t num_tries = 3;

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
  //if (procid == 0) printf("EVAL ec VEC EB ------------------------------\n");
  //part_eval_weighted(g, pulp);
  update_pulp_data_weighted(g, pulp);
  num_swapped_1 = 0;
}

  for (uint64_t cur_bal_iter = 0; cur_bal_iter < balance_iter; ++cur_bal_iter)
  {

    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      tp.part_weights[p] = vert_balance * pulp->avg_size / (double)pulp->part_sizes[p] - 1.0;
      tp.part_edge_weights[p] = pulp->max_e * pulp->avg_edge_size / (double)pulp->part_edge_sizes[p] - 1.0;
      tp.part_cut_weights[p] = pulp->max_c * pulp->avg_cut_size / (double)pulp->part_cut_sizes[p] - 1.0;
      if (tp.part_weights[p] < 0.0)
        tp.part_weights[p] = 0.0;
      if (tp.part_edge_weights[p] < 0.0)
        tp.part_edge_weights[p] = 0.0;
      if (tp.part_cut_weights[p] < 0.0)
        tp.part_cut_weights[p] = 0.0;
    }

#pragma omp for schedule(guided) reduction(+:num_swapped_1) nowait
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
      int64_t weights_sum = 0;
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        int32_t part_out = pulp->local_parts[out_index];
        double weight_out = 1.0;
        if (has_ewgts) weight_out = (double)weights[j];
        tp.part_counts[part_out] += weight_out;
        weights_sum += weight_out;
      }

      int32_t max_part = part;
      double max_val = 0.0;
      uint64_t num_max = 0;
      int64_t max_count = 0;
      int64_t part_count = (int64_t)tp.part_counts[part];
      for (int32_t p = 0; p < pulp->num_parts; ++p)
      {
        int64_t count_init = (int64_t)tp.part_counts[p];
        if (tp.part_weights[p] > 0.0 && tp.part_edge_weights[p] > 0.0 && tp.part_cut_weights[p] > 0.0)
          tp.part_counts[p] *= (tp.part_edge_weights[p]*pulp->weight_exponent_e * tp.part_cut_weights[p]*pulp->weight_exponent_c);
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
          max_count = count_init;
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
        int64_t diff_part = 2*part_count - weights_sum;
        int64_t diff_max_part = weights_sum - 2*max_count;
        int64_t diff_cut = part_count - max_count;  

    #pragma omp atomic
        pulp->cut_size_change += diff_cut;
    #pragma omp atomic
        pulp->part_cut_size_changes[part] += diff_part;
    #pragma omp atomic
        pulp->part_cut_size_changes[max_part] += diff_max_part;
    #pragma omp atomic
        pulp->part_size_changes[part] -= vert_weight;
    #pragma omp atomic
        pulp->part_size_changes[max_part] += vert_weight;
    #pragma omp atomic
        pulp->part_edge_size_changes[part] -= (int64_t)out_degree;
    #pragma omp atomic
        pulp->part_edge_size_changes[max_part] += (int64_t)out_degree;
        
        tp.part_weights[part] = 
          vert_balance * pulp->avg_size / 
          ((double)pulp->part_sizes[part] + multiplier*(double)pulp->part_size_changes[part]) - 1.0;
        tp.part_weights[max_part] = 
          vert_balance * pulp->avg_size / 
          ((double)pulp->part_sizes[max_part] + multiplier*(double)pulp->part_size_changes[max_part]) - 1.0;

        tp.part_edge_weights[part] = 
          pulp->max_e * pulp->avg_edge_size / 
          ((double)pulp->part_edge_sizes[part] + multiplier*(double)pulp->part_edge_size_changes[part]) - 1.0;
        tp.part_edge_weights[max_part] = 
          pulp->max_e * pulp->avg_edge_size / 
          ((double)pulp->part_edge_sizes[max_part] + multiplier*(double)pulp->part_edge_size_changes[max_part]) - 1.0;

        double avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;
        tp.part_cut_weights[part] = 
          pulp->max_c * avg_cut_size / 
          ((double)pulp->part_cut_sizes[part] + multiplier*(double)pulp->part_cut_size_changes[part]) - 1.0;  
        tp.part_cut_weights[max_part] = 
          pulp->max_c * avg_cut_size / 
          ((double)pulp->part_cut_sizes[max_part] + multiplier*(double)pulp->part_cut_size_changes[max_part]) - 1.0;  

        if (tp.part_weights[part] < 0.0)
          tp.part_weights[part] = 0.0;
        if (tp.part_weights[max_part] < 0.0)
          tp.part_weights[max_part] = 0.0;

        if (tp.part_edge_weights[part] < 0.0)
          tp.part_edge_weights[part] = 0.0;
        if (tp.part_edge_weights[max_part] < 0.0)
          tp.part_edge_weights[max_part] = 0.0;

        if (tp.part_cut_weights[part] < 0.0)
          tp.part_cut_weights[part] = 0.0;
        if (tp.part_cut_weights[max_part] < 0.0)
          tp.part_cut_weights[max_part] = 0.0;

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

    MPI_Allreduce(MPI_IN_PLACE, pulp->part_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pulp->part_edge_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pulp->part_cut_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &pulp->cut_size_change, 1, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    pulp->cut_size += pulp->cut_size_change;
    pulp->cut_size_change = 0;
    pulp->avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;

    pulp->max_v = 0.0;
    pulp->max_e = 0.0;
    pulp->max_c = 0.0;
    pulp->max_cut = 0;

    //if (procid == 0)
    //for (int32_t i = 0; i < pulp->num_parts; ++i)
    //  printf("p: %d, v: %li, e: %li, cut: %li\n",
    //  i, pulp->part_sizes[i], pulp->part_edge_sizes[i], pulp->part_cut_sizes[i]);
    //part_eval(g, pulp);

    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      //if (procid == 0)
      //  printf("p: %d, vc: %li, ec: %li, cc: %li\n", p, 
      //    pulp->part_size_changes[p], pulp->part_edge_size_changes[p],
      //    pulp->part_cut_size_changes[p]);

      pulp->part_sizes[p] += pulp->part_size_changes[p];
      pulp->part_edge_sizes[p] += pulp->part_edge_size_changes[p];
      pulp->part_cut_sizes[p] += pulp->part_cut_size_changes[p];
      pulp->part_size_changes[p] = 0;
      pulp->part_edge_size_changes[p] = 0;
      pulp->part_cut_size_changes[p] = 0;

      if ((double)pulp->part_sizes[p] / pulp->avg_size > pulp->max_v)
        pulp->max_v = (double)pulp->part_sizes[p] / pulp->avg_size;        
      if ((double)pulp->part_edge_sizes[p] / pulp->avg_edge_size > pulp->max_e)
        pulp->max_e = (double)pulp->part_edge_sizes[p] / pulp->avg_edge_size;
      if ((double)pulp->part_cut_sizes[p] / pulp->avg_cut_size > pulp->max_c)
        pulp->max_c = (double)pulp->part_cut_sizes[p] / pulp->avg_cut_size;
      if (pulp->part_cut_sizes[p] > pulp->max_cut)
        pulp->max_cut = pulp->part_cut_sizes[p];
    }
    if (pulp->max_e < edge_balance)
    {
      pulp->max_e = edge_balance;
      pulp->weight_exponent_e = 1.0;
      pulp->weight_exponent_c *= pulp->max_c;
    }
    else
    {
      pulp->weight_exponent_e *= pulp->max_e / edge_balance;
      pulp->weight_exponent_c = 1.0;
    }

    cur_iter += 1.0;
    //multiplier = (double)pulp->num_parts*(1.0-(double)cur_iter/(double)tot_iter)+1.0*pulp->num_parts*(double)cur_iter/((double)tot_iter*2.0);
    multiplier = (double)nprocs*( (X - Y)*(cur_iter/tot_iter) + Y );

    //int32_t* tmp = pulp->local_parts;
    //pulp->local_parts = pulp->local_parts_next;
    //pulp->local_parts_next = tmp;
    if (debug) printf("Task %d num_swapped_1 %lu \n", procid, num_swapped_1);
    num_swapped_1 = 0;
    //if (procid == 0)
    //  printf("&&& XtraPuLP, %2.3lf, %2.3lf, %2.3lf, %li, %li\n",
    //    pulp->max_v, pulp->max_e, pulp->max_c, pulp->cut_size, pulp->max_cut);
    //part_eval(g, pulp);
    //printf("\n\n\n\n");
}
  }// end balance loop


#pragma omp single
{  
  //if (procid == 0) printf("EVAL ec VEC ER ------------------------------\n");
  //part_eval_weighted(g, pulp);
  update_pulp_data_weighted(g, pulp);
  num_swapped_2 = 0;
}


  for (uint64_t cur_ref_iter = 0; cur_ref_iter < refine_iter; ++cur_ref_iter)
  {

#pragma omp for schedule(guided) reduction(+:num_swapped_2) nowait
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
      int64_t weights_sum = 0;
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        int32_t part_out = pulp->local_parts[out_index];
        double weight_out = 1.0;
        if (has_ewgts) weight_out = (double)weights[j];
        tp.part_counts[part_out] += weight_out;
        weights_sum += weight_out;
      }

      int32_t max_part = part;
      double max_val = 0.0;
      uint64_t num_max = 0;
      int64_t max_count = 0;
      int64_t part_count = (int64_t)tp.part_counts[part];
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
          max_count = (int64_t)max_val;
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
        int64_t new_edge_size = (int64_t)pulp->avg_edge_size;
        double avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;
        int64_t new_cut_size = (int64_t)avg_cut_size;
        int64_t new_max_cut_size = (int64_t)avg_cut_size;

        int64_t diff_part = 2*part_count - (int64_t)weights_sum;
        int64_t diff_max_part = (int64_t)weights_sum - 2*max_count;
        int64_t diff_cut = part_count - max_count;  

        pulp->part_size_changes[max_part] + (int64_t)vert_weight < 0 ? 
          new_size = pulp->part_sizes[max_part] + pulp->part_size_changes[max_part] + (int64_t)vert_weight :
          new_size = (int64_t)((double)pulp->part_sizes[max_part] + multiplier*(double)pulp->part_size_changes[max_part] + (double)vert_weight);

        pulp->part_edge_size_changes[max_part] + out_degree < 0 ?
          new_edge_size = pulp->part_edge_sizes[max_part] + pulp->part_edge_size_changes[max_part] + out_degree :
          new_edge_size = (int64_t)((double)pulp->part_edge_sizes[max_part] + multiplier*(double)pulp->part_edge_size_changes[max_part] + (double)(out_degree));

        pulp->part_cut_size_changes[part] < 0 ?
          new_cut_size = pulp->part_cut_sizes[part] + pulp->part_cut_size_changes[part] + diff_part :
          new_cut_size = (int64_t)((double)pulp->part_cut_sizes[part] + multiplier*(double)pulp->part_cut_size_changes[part] + (double)diff_part);
             
        pulp->part_cut_size_changes[max_part] < 0 ?
          new_cut_size = pulp->part_cut_sizes[max_part] + pulp->part_cut_size_changes[max_part] + diff_max_part :
          new_cut_size = (int64_t)((double)pulp->part_cut_sizes[max_part] + multiplier*(double)pulp->part_cut_size_changes[max_part] + (double)diff_max_part);


        if (new_size < (int64_t)(pulp->avg_size*vert_balance) &&
          new_edge_size < (int64_t)(pulp->avg_edge_size*pulp->max_e) &&
          new_cut_size < (int64_t)(avg_cut_size*pulp->max_c) &&
          new_max_cut_size < (int64_t)(avg_cut_size*pulp->max_c) )
        {
          ++num_swapped_2;
      #pragma omp atomic
          pulp->cut_size_change += diff_cut;
      #pragma omp atomic
          pulp->part_cut_size_changes[part] += diff_part;
      #pragma omp atomic
          pulp->part_cut_size_changes[max_part] += diff_max_part;
      #pragma omp atomic
          pulp->part_size_changes[part] -= vert_weight;
      #pragma omp atomic
          pulp->part_size_changes[max_part] += vert_weight;
      #pragma omp atomic
          pulp->part_edge_size_changes[part] -= (int64_t)out_degree;
      #pragma omp atomic
          pulp->part_edge_size_changes[max_part] += (int64_t)out_degree;   

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
      //pulp->local_parts_next[index] = comm->recvbuf_data[i];
    }

#pragma omp single
{
    clear_recvbuf_vid_data(comm);

    MPI_Allreduce(MPI_IN_PLACE, pulp->part_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pulp->part_edge_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pulp->part_cut_size_changes, pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &pulp->cut_size_change, 1, 
        MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    pulp->cut_size += pulp->cut_size_change;
    pulp->cut_size_change = 0;
    pulp->avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;

    /*pulp->max_v = 0.0;
    pulp->max_e = 0.0;
    pulp->max_c = 0.0;*/
    pulp->max_cut = 0;
    for (int32_t p = 0; p < pulp->num_parts; ++p)
    {
      pulp->part_sizes[p] += pulp->part_size_changes[p];
      pulp->part_edge_sizes[p] += pulp->part_edge_size_changes[p];
      pulp->part_cut_sizes[p] += pulp->part_cut_size_changes[p];
      pulp->part_size_changes[p] = 0;
      pulp->part_edge_size_changes[p] = 0;
      pulp->part_cut_size_changes[p] = 0;

      /*if ((double)pulp->part_sizes[p] / pulp->avg_size > pulp->max_v)
        pulp->max_v = (double)pulp->part_sizes[p] / pulp->avg_size;
      if ((double)pulp->part_edge_sizes[p] / pulp->avg_edge_size > pulp->max_e)
        pulp->max_e = (double)pulp->part_edge_sizes[p] / pulp->avg_edge_size;
      if ((double)pulp->part_cut_sizes[p] / pulp->avg_cut_size > pulp->max_c)
        pulp->max_c = (double)pulp->part_cut_sizes[p] / pulp->avg_cut_size;*/
      if (pulp->part_cut_sizes[p] > pulp->max_cut)
        pulp->max_cut = pulp->part_cut_sizes[p];
    }
    /*if (pulp->max_e < edge_balance)
    {
      pulp->max_e = edge_balance;
      pulp->weight_exponent_e = 1.0;
      pulp->weight_exponent_c *= pulp->max_c;
    }
    else
    {
      pulp->weight_exponent_e *= pulp->max_e / edge_balance;
      pulp->weight_exponent_c = 1.0;
    }*/

    cur_iter += 1.0;
    //multiplier = (double)pulp->num_parts*(1.0-(double)cur_iter/(double)tot_iter)+1.0*pulp->num_parts*(double)cur_iter/((double)tot_iter*2.0);
    multiplier = (double)nprocs*( (X - Y)*(cur_iter/tot_iter) + Y );
    //printf("mult %9.6lf\n", multiplier);

    //int32_t* tmp = pulp->local_parts;
    //pulp->local_parts = pulp->local_parts_next;
    //pulp->local_parts_next = tmp;
    if (debug) printf("Task %d num_swapped_2 %lu \n", procid, num_swapped_2);
    num_swapped_2 = 0;
    //printf("%d &&& XtraPuLP, %2.3lf, %2.3lf, %2.3lf, %li, %li\n", procid,
    //    pulp->max_v, pulp->max_e, pulp->max_c, pulp->cut_size, pulp->max_cut);
    //part_eval(g, pulp);
}

  } // end refine iter

  /*if (cur_outer_iter + 1 == outer_iter)
  {
#pragma omp single
{
    update_pulp_data(g, pulp);

    if ( ( (pulp->max_e > edge_balance*BAL_CUTOFF &&
            pulp->max_e*BAL_CHANGE < running_bal) ||
            (double)pulp->cut_size < running_cut*CUT_CHANGE ) &&
          num_tries < 3)
    {
      outer_iter += 1;
      tot_iter += (double)(balance_iter + refine_iter);
      ++num_tries;
      running_cut = (double)pulp->cut_size;
      running_bal = pulp->max_e;
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
    printf("Task %d, pulp_vec_weighted() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d pulp_vec_weighted() success\n", procid); }

  return 0;
}
