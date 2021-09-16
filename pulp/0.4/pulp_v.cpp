#include <omp.h>
#include <cstdlib>
#include <cstdio>

#include "pulp.h"
#include "pulp_v.h"
#include "graph.h"
#include "thread.h"

int part_balance(graph* g, int num_parts, int* parts, double imb_limit)
{
  double elt = omp_get_wtime();
  printf("Begin part_balance()\n");
  
  int num_verts = g->num_verts;
  
  long* part_sizes = new long[num_parts];
  for (int i = 0; i < num_parts; ++i)
    part_sizes[i] = 0;
  
  double avg_size = (double)g->vert_weights_sum  / (double)num_parts;
  int max_size = (int)(imb_limit * avg_size);
  double imbalance = (double)g->vert_weights_sum;
  double imbalance_prev = (double)g->vert_weights_sum*2.0;
  int edge_cut = g->edge_weights_sum;
  int edge_cut_prev = g->edge_weights_sum*2;
  
  int* queue = new int[num_verts*QUEUE_MULTIPLIER];
  int* queue_next = new int[num_verts*QUEUE_MULTIPLIER];
  bool* in_queue = new bool[num_verts];
  bool* in_queue_next = new bool[num_verts];
  int queue_size = num_verts;
  int queue_size_next = 0;
  
  int num_swapped = 0;
  int num_iter = 0;
  
#pragma omp parallel
{
  long* part_sizes_thread = new long[num_parts];
  for (int i = 0; i < num_parts; ++i) 
    part_sizes_thread[i] = 0;

#pragma omp for schedule(static) nowait
  for (int i = 0; i < num_verts; ++i)
    part_sizes_thread[parts[i]] += g->vert_weights[i];

  for (int i = 0; i < num_parts; ++i) 
#pragma omp atomic
    part_sizes[i] += part_sizes_thread[i];

  delete [] part_sizes_thread;
#pragma omp barrier

  double* part_counts = new double[num_parts];
  double* part_weights = new double[num_parts];

  int thread_queue[THREAD_QUEUE_SIZE];
  int thread_queue_size = 0;

  for (int p = 0; p < num_parts; ++p)
  {        
    part_weights[p] = imb_limit * avg_size / (double)part_sizes[p] - 1.0;
    if (part_weights[p] < 0.0)
      part_weights[p] = 0.0;
  }

#pragma omp for schedule(static) nowait
  for (int i = 0; i < num_verts; ++i)
    queue[i] = i;
#pragma omp for schedule(static)
  for (int i = 0; i < num_verts; ++i)
    in_queue_next[i] = false;

  while((imbalance < imbalance_prev*IMPROVEMENT_RATIO || 
        edge_cut < edge_cut_prev*IMPROVEMENT_RATIO) &&   
        num_iter < MAX_ITER)
  {
#pragma omp barrier
#pragma omp single
{
  edge_cut_prev = edge_cut;  
  edge_cut = 0;
}
    
  #pragma omp for schedule(guided) \
    reduction(+:num_swapped) reduction(+:edge_cut) nowait
    for (int i = 0; i < queue_size; ++i)
    {
      int v = queue[i];
      in_queue[v] = false;
      int part = parts[v];
      int v_weight = g->vert_weights[v];

      for (int p = 0; p < num_parts; ++p)
        part_counts[p] = 0.0;

      unsigned degree = out_degree(g, v);
      int* outs = out_vertices(g, v);
      int* weights = out_weights(g, v);
      for (unsigned j = 0; j < degree; ++j) {
        int out = outs[j];
        int part_out = parts[out];
        double weight_out = (double)weights[j];
        part_counts[part_out] += (double)out_degree(g, out)*weight_out;
      }
      
      int max_part = part;
      double max_val = 0.0;
      for (int p = 0; p < num_parts; ++p) {
        part_counts[p] *= part_weights[p];
        
        if (part_counts[p] > max_val) {
          max_val = part_counts[p];
          max_part = p;
        }
      }

      if (max_part != part) {        
        parts[v] = max_part;
        ++num_swapped;
    #pragma omp atomic
        part_sizes[max_part] += v_weight;
    #pragma omp atomic
        part_sizes[part] -= v_weight;
        
        part_weights[part] = (double)max_size / (double)part_sizes[part] - 1.0;
        part_weights[max_part] = (double)max_size / (double)part_sizes[max_part]  - 1.0;
        
        if (part_weights[part] < 0.0)
          part_weights[part] = 0.0;
        if (part_weights[max_part] < 0.0)
          part_weights[max_part] = 0.0;

        if (!in_queue_next[v]) {
          in_queue_next[v] = true;
          add_to_queue(thread_queue, thread_queue_size, 
                        queue_next, queue_size_next, v);
        }

        for (unsigned j = 0; j < degree; ++j) {
          int out = outs[j];
          if (!in_queue_next[out]) {
            in_queue_next[out] = true;
            add_to_queue(thread_queue, thread_queue_size, 
                          queue_next, queue_size_next, out);
          }
        }
      }
      
      for (unsigned j = 0; j < degree; ++j) {
        int out = outs[j];
        int part_out = parts[out];
        if (part_out != max_part)
          edge_cut += weights[j];
      }
    }

    empty_queue(thread_queue, thread_queue_size, 
                queue_next, queue_size_next);
  #pragma omp barrier

  #pragma omp single
    {
      ++num_iter;
    #if VERBOSE
      printf("%d %d\n", num_swapped, queue_size_next);
    #endif
      num_swapped = 0;
      
      int* temp = queue;
      queue = queue_next;
      queue_next = temp;
      bool* temp_b = in_queue;
      in_queue = in_queue_next;
      in_queue_next = temp_b;
      queue_size = queue_size_next;
      queue_size_next = 0;
      
      imbalance_prev = imbalance;
      imbalance = 0.0;
      for (int p = 0; p < num_parts; ++p) {
        if (part_sizes[p] / avg_size > imbalance)
          imbalance = part_sizes[p] / avg_size;
      }
      
#if OUTPUT_STEP
      char step[] = "Balance";
      evaluate_quality_step(step, g, num_parts, parts);
#endif
    } // end single
  } // end while
  delete [] part_counts;
  delete [] part_weights;
  
} // end parallel
  
  delete [] part_sizes;
  delete [] queue;
  delete [] queue_next;
  delete [] in_queue;
  delete [] in_queue_next;
  
  printf("Done part_balance(): %lf (s)\n", omp_get_wtime() - elt); 
  
  return 0;
}


int part_refine(graph* g, int num_parts, int* parts, double imb_limit)
{
  double elt = omp_get_wtime();
  printf("Begin part_refine()\n");
  
  int num_verts = g->num_verts;
  
  long* part_sizes = new long[num_parts];
  for (int i = 0; i < num_parts; ++i)
    part_sizes[i] = 0;
  
  double avg_size = (double)g->vert_weights_sum / (double)num_parts;
  int max_size = (int)(imb_limit * avg_size);
  double imbalance = (double)g->vert_weights_sum;
  double imbalance_prev = (double)g->vert_weights_sum*2.0;
  int edge_cut = g->edge_weights_sum;
  int edge_cut_prev = g->edge_weights_sum*2;
  
  int* queue = new int[num_verts*QUEUE_MULTIPLIER];
  int* queue_next = new int[num_verts*QUEUE_MULTIPLIER];
  bool* in_queue = new bool[num_verts];
  bool* in_queue_next = new bool[num_verts];
  int queue_size = num_verts;
  int queue_size_next = 0;
  
  int num_swapped = 0;
  int num_iter = 0;
  
#pragma omp parallel
{
  long* part_sizes_thread = new long[num_parts];
  for (int i = 0; i < num_parts; ++i) 
    part_sizes_thread[i] = 0;

#pragma omp for schedule(static) nowait
  for (int i = 0; i < num_verts; ++i)
    part_sizes_thread[parts[i]] += g->vert_weights[i];

  for (int i = 0; i < num_parts; ++i) 
#pragma omp atomic
    part_sizes[i] += part_sizes_thread[i];

  delete [] part_sizes_thread;
#pragma omp barrier

  double* part_counts = new double[num_parts];

  int thread_queue[THREAD_QUEUE_SIZE];
  int thread_queue_size = 0;

#pragma omp for schedule(static) nowait
  for (int i = 0; i < num_verts; ++i)
    queue[i] = i;
#pragma omp for schedule(static)
  for (int i = 0; i < num_verts; ++i)
    in_queue_next[i] = false;

  while((imbalance < imbalance_prev*IMPROVEMENT_RATIO || 
        edge_cut < (int)((double)edge_cut_prev*IMPROVEMENT_RATIO)) &&   
        num_iter < MAX_ITER)
  {
#pragma omp barrier
#pragma omp single
{
  edge_cut_prev = edge_cut;  
  edge_cut = 0;
}
    
  #pragma omp for schedule(guided) reduction(+:num_swapped) nowait
    for (int i = 0; i < queue_size; ++i)
    {
      int v = queue[i];
      in_queue[v] = false;
      int part = parts[v];
      int v_weight = g->vert_weights[v];

      for (int p = 0; p < num_parts; ++p)
        part_counts[p] = 0.0;

      unsigned degree = out_degree(g, v);
      int* outs = out_vertices(g, v);
      int* weights = out_weights(g, v);
      for (unsigned j = 0; j < degree; ++j) {
        int out = outs[j];
        int part_out = parts[out];
        double weight_out = (double)weights[j];
        part_counts[part_out] += weight_out;
      }
      
      int max_part = part;
      double max_val = 0.0;
      for (int p = 0; p < num_parts; ++p) {
        if (part_counts[p] > max_val) {
          max_val = part_counts[p];
          max_part = p;
        }
      }

      if (max_part != part && 
          (part_sizes[max_part] + v_weight) < max_size) {
        parts[v] = max_part;
        ++num_swapped;
    #pragma omp atomic
        part_sizes[max_part] += v_weight;
    #pragma omp atomic
        part_sizes[part] -= v_weight;

        if (!in_queue_next[v]) {
          in_queue_next[v] = true;
          add_to_queue(thread_queue, thread_queue_size, 
                        queue_next, queue_size_next, v);
        }

        for (unsigned j = 0; j < degree; ++j) {
          int out = outs[j];
          if (!in_queue_next[out]) {
            in_queue_next[out] = true;
            add_to_queue(thread_queue, thread_queue_size, 
                          queue_next, queue_size_next, out);
          }
        }
      }
      
      for (unsigned j = 0; j < degree; ++j) {
        int out = outs[j];
        int part_out = parts[out];
        if (part_out != max_part)
          edge_cut += weights[j];
      }
    }

    empty_queue(thread_queue, thread_queue_size, 
                queue_next, queue_size_next);
  #pragma omp barrier

  #pragma omp single
    {
      ++num_iter;
    #if VERBOSE
      printf("%d %d\n", num_swapped, queue_size_next);
    #endif
      num_swapped = 0;
      
      int* temp = queue;
      queue = queue_next;
      queue_next = temp;
      bool* temp_b = in_queue;
      in_queue = in_queue_next;
      in_queue_next = temp_b;
      queue_size = queue_size_next;
      queue_size_next = 0;
      
      imbalance_prev = imbalance;
      imbalance = 0.0;
      for (int p = 0; p < num_parts; ++p) {
        if (part_sizes[p] / avg_size > imbalance)
          imbalance = part_sizes[p] / avg_size;
      }
      
#if OUTPUT_STEP
      char step[] = "Refine";
      evaluate_quality_step(step, g, num_parts, parts);
#endif
    } // end single
  } // end while
  delete [] part_counts;
  
} // end parallel
  
  delete [] part_sizes;
  delete [] queue;
  delete [] queue_next;
  delete [] in_queue;
  delete [] in_queue_next;
  
  printf("Done part_refine(): %lf (s)\n", omp_get_wtime() - elt); 
  
  return 0;
}
