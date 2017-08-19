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

#include "pulp_data.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;

//TODO: Better way to get the info vertex_weights_num?
//What Esco did: Changed allocation size of part_weights to take in multiple weights
void init_thread_pulp(thread_pulp_t* tp, pulp_data_t* pulp, int vertex_weights_num)
{  
  //if (debug) printf("Task %d init_thread_pulp() start\n", procid); 

  tp->part_counts = (double*)malloc(pulp->num_parts*sizeof(double));
  
  tp->part_weights = (double*)malloc(vertex_weights_num * pulp->num_parts * sizeof(double));
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

//What Esco did: modified initilization of pulp->avg_size, pulp->max_v, pulp->running_max_v for multiple weight components;
//Changed allocation size of part_sizes and part_size_changes to take in multiple weights
void init_pulp_data(dist_graph_t* g, pulp_data_t* pulp, int32_t num_parts)
{
  if (debug) printf("Task %d init_pulp_data() start\n", procid); 
  
  pulp->num_parts = num_parts;
  pulp->avg_size = new double[g->vertex_weights_num];
  pulp->max_v = new double[g->vertex_weights_num];
  pulp->running_max_v = new double[g->vertex_weights_num];
  

  for(uint32_t wc = 0; wc < g->vertex_weights_num; ++wc)
  {
    pulp->avg_size[wc] = 0;
  }

  if (g->edge_weights == NULL && g->vertex_weights == NULL)
    pulp->avg_size[0] = (double)g->n / (double)pulp->num_parts;
	else if (g->vertex_weights != NULL)
	{
		for (uint64_t wc = 0; wc < g->vertex_weights_num; ++wc)
		{
			pulp->avg_size[wc] = (double)g->vertex_weights_sum[wc] / (double)pulp->num_parts;
		}
	}
	else if (g->edge_weights != NULL)
	{
		//write something similar like vertex_weights above
	}

  pulp->avg_edge_size = (double)g->m*2 / (double)pulp->num_parts;
  pulp->avg_cut_size = 0.0;

	for (uint64_t wc = 0; wc < g->vertex_weights_num; ++wc)
	{
		pulp->max_v[wc] = 0.0;
	}

  pulp->max_e = 1.0;
  pulp->max_c = 1.0;
  pulp->weight_exponent_e = 1.0;
  pulp->weight_exponent_c = 1.0;

	for (uint64_t wc = 0; wc < g->vertex_weights_num; ++wc)
	{
		pulp->running_max_v[wc] = 0.0;
	}

  pulp->running_max_e = 1.0;
  pulp->running_max_c = 1.0;

  pulp->local_parts = (int32_t*)malloc(g->n_total*sizeof(int32_t));

  pulp->part_sizes = new int64_t[pulp->num_parts * g->vertex_weights_num];
  pulp->part_edge_sizes = (int64_t*)malloc(pulp->num_parts*sizeof(int64_t));

  pulp->part_cut_sizes = (int64_t*)malloc(pulp->num_parts*sizeof(int64_t));

  pulp->part_size_changes = new int64_t[pulp->num_parts * g->vertex_weights_num];

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
	{
		for (uint64_t wc = 0; wc < g->vertex_weights_num; ++wc)
		{
			pulp->part_sizes[p * g->vertex_weights_num + wc] = 0;
			pulp->part_size_changes[p * g->vertex_weights_num + wc] = 0;
		}
	}

  for (int32_t p = 0; p < pulp->num_parts; ++p)
    pulp->part_edge_sizes[p] = 0;
  for (int32_t p = 0; p < pulp->num_parts; ++p)
    pulp->part_cut_sizes[p] = 0;
  for (int32_t p = 0; p < pulp->num_parts; ++p)
    pulp->part_edge_size_changes[p] = 0;
  for (int32_t p = 0; p < pulp->num_parts; ++p)
    pulp->part_cut_size_changes[p] = 0;
  
  if (debug) printf("Task %d init_pulp_data() success\n", procid);
}


//What Esco did: modified several variables involving vertex_weights (pulp->part_sizes, pulp->part_size_changes,...)
//to iterate through each weight component. 
void update_pulp_data_weighted(dist_graph_t* g, pulp_data_t* pulp)
{
  bool has_vwgts = (g->vertex_weights != NULL);
  bool has_ewgts = (g->edge_weights != NULL);

  for (int32_t p = 0; p < pulp->num_parts; ++p)
  {
		for (uint64_t wc = 0; wc < g->vertex_weights_num; ++wc)
		{
			pulp->part_sizes[p * g->vertex_weights_num + wc] = 0;
			pulp->part_size_changes[p * g->vertex_weights_num + wc] = 0;
		}
  
    pulp->part_edge_sizes[p] = 0;
    pulp->part_cut_sizes[p] = 0;
    pulp->part_edge_size_changes[p] = 0;
    pulp->part_cut_size_changes[p] = 0;
  }

  pulp->cut_size = 0;

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert_index = i;
    int32_t part = pulp->local_parts[vert_index];
		if (has_vwgts)
		{
			for (uint64_t wc = 0; wc < g->vertex_weights_num; ++wc)
			{
        pulp->part_sizes[part * g->vertex_weights_num + wc] += g->vertex_weights[vert_index*(g->vertex_weights_num) + wc];
      }
		}
		else
		{
			//if graph has no vertex weights, then this is equilavent to g->vertex_weights_num = 1 where weights are 1
			//part_sizes[part] = # of vertices in the partition = sum of weights (all equal to 1) in the partition 
			++pulp->part_sizes[part];
    }
    
    //variable weights uses edge weights, not vertex weights. No modifications needed for vertex weights.
    
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

	//pulp->part_sizes is a 1-D array of each partition's multiple vertex weights, first ordered by partition, then weight component
	//Example: pulp->part_sizes = [p1_w1, p1_w2, p1_3, p2_w1, p2_w2, p2_w3, ... ]
  //pulp->num_parts * g->vertex_weights_num = total # of weights from all the parititons

  MPI_Allreduce(MPI_IN_PLACE, pulp->part_sizes, pulp->num_parts * g->vertex_weights_num, 
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_edge_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_cut_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pulp->cut_size, 1,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  pulp->cut_size /= 2;
  pulp->avg_cut_size = (double)pulp->cut_size / (double)pulp->num_parts;

	for (uint64_t wc = 0; wc < g->vertex_weights_num; ++wc)
	{
		pulp->max_v[wc] = 0;
	}

  pulp->max_e = 0;
  pulp->max_c = 0;
  pulp->max_cut = 0;

  for (int32_t p = 0; p < pulp->num_parts; ++p)
  {
		for (uint64_t wc = 0; wc < g->vertex_weights_num; ++wc)
		{
			if ((double)pulp->part_sizes[p * g->vertex_weights_num + wc] / pulp->avg_size[wc] > pulp->max_v[wc])
				pulp->max_v[wc] = (double) pulp->part_sizes[p * g->vertex_weights_num + wc] / pulp->avg_size[wc];
		}
   
    if ((double)pulp->part_edge_sizes[p] / pulp->avg_edge_size > pulp->max_e)
      pulp->max_e = (double)pulp->part_edge_sizes[p] / pulp->avg_edge_size;
    if ((double)pulp->part_cut_sizes[p] / pulp->avg_cut_size > pulp->max_c)
      pulp->max_c = (double)pulp->part_cut_sizes[p] / pulp->avg_cut_size;
    if (pulp->part_cut_sizes[p] > pulp->max_cut)
      pulp->max_cut = pulp->part_cut_sizes[p];
  }
}


//What Esco did: added in pulp->avg_size, max_v, and running_max_v since they're now memory allocated
void clear_pulp_data(pulp_data_t* pulp)
{
  if (debug) printf("Task %d clear_pulp_data() start\n", procid); 

	delete [] pulp->avg_size;
	delete [] pulp->max_v;
	delete [] pulp->running_max_v;

  free(pulp->local_parts);
  delete [] pulp->part_sizes;
  free(pulp->part_edge_sizes);
  free(pulp->part_cut_sizes);
  delete [] pulp->part_size_changes;
  free(pulp->part_edge_size_changes);
  free(pulp->part_cut_size_changes);

  if (debug) printf("Task %d clear_pulp_data() success\n", procid); 
}
