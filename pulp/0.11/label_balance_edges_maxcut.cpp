/*
//@HEADER
// *****************************************************************************
//
// PULP: Multi-Objective Multi-Constraint Partitioning Using Label Propagation
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
// Questions?  Contact  George M. Slota (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//
// *****************************************************************************
//@HEADER
*/

using namespace std;

/*
'########::'##:::'##:'##::: ##:'####:'########:'########:'########::
 ##.... ##:. ##:'##:: ###:: ##:. ##::... ##..:: ##.....:: ##.... ##:
 ##:::: ##::. ####::: ####: ##:: ##::::: ##:::: ##::::::: ##:::: ##:
 ##:::: ##:::. ##:::: ## ## ##:: ##::::: ##:::: ######::: ########::
 ##:::: ##:::: ##:::: ##. ####:: ##::::: ##:::: ##...:::: ##.. ##:::
 ##:::: ##:::: ##:::: ##:. ###:: ##::::: ##:::: ##::::::: ##::. ##::
 ########::::: ##:::: ##::. ##:'####:::: ##:::: ########: ##:::. ##:
........::::::..:::::..::::..::....:::::..:::::........::..:::::..::
*/
void label_balance_edges_maxcut(pulp_graph_t& g, int num_parts, int* parts,
  int edge_outer_iter, int edge_balance_iter, int edge_refine_iter,
  double vert_balance, double edge_balance)
{
  int num_verts = g.n;
  unsigned num_edges = g.m;
  unsigned cut_size = 0;
  int* part_sizes = new int[num_parts];
  unsigned* part_edge_sizes = new unsigned[num_parts];
  int* part_cut_sizes = new int[num_parts];
 
  for (int i = 0; i < num_parts; ++i)
    part_sizes[i] = 0;
  for (int i = 0; i < num_parts; ++i)
    part_edge_sizes[i] = 0;
  for (int i = 0; i < num_parts; ++i)
    part_cut_sizes[i] = 0;

  double avg_size = num_verts / num_parts;
  double avg_edge_size = num_edges / num_parts;
  int num_swapped_1 = 0;
  int num_swapped_2 = 0;
  bool swapped = true;
  double max_e = 0.0;
  double max_c = 0.0;
  double running_max_e = (double)num_edges;
  double weight_exponent_e = 1.0;
  double weight_exponent_c = 1.0;

  int* queue = new int[num_verts*QUEUE_MULTIPLIER];
  int* queue_next = new int[num_verts*QUEUE_MULTIPLIER];
  bool* in_queue = new bool[num_verts];
  bool* in_queue_next = new bool[num_verts];
  int queue_size;
  int next_size;
  int t = 0;
  int num_tries = 0;

#pragma omp parallel
{

  int* part_sizes_thread = new int[num_parts];
  unsigned* part_edge_sizes_thread = new unsigned[num_parts];
  int* part_cut_sizes_thread = new int[num_parts];
  unsigned cut_size_thread = 0;
  for (int i = 0; i < num_parts; ++i) 
    part_sizes_thread[i] = 0;
  for (int i = 0; i < num_parts; ++i) 
    part_edge_sizes_thread[i] = 0;
  for (int i = 0; i < num_parts; ++i) 
    part_cut_sizes_thread[i] = 0;
  
#pragma omp for schedule(guided) nowait
  for (int i = 0; i < num_verts; ++i)
  {
    int part = parts[i];
    int out_degree = out_degree(g, i);
    ++part_sizes_thread[part];
    part_edge_sizes_thread[part] += out_degree;

    int* outs = out_vertices(g, i);
    for (int j = 0; j < out_degree; ++j)
    {
      int out = outs[j];
      int out_part = parts[out];
      if (out_part != part)
      {
        ++part_cut_sizes_thread[part];
        ++cut_size_thread;
      }
    }
  }

  for (int i = 0; i < num_parts; ++i) 
#pragma omp atomic
    part_sizes[i] += part_sizes_thread[i];
  for (int i = 0; i < num_parts; ++i) 
#pragma omp atomic
    part_edge_sizes[i] += part_edge_sizes_thread[i];
  for (int i = 0; i < num_parts; ++i) 
#pragma omp atomic
    part_cut_sizes[i] += part_cut_sizes_thread[i];
#pragma omp atomic
  cut_size += cut_size_thread;

  delete [] part_sizes_thread;
  delete [] part_edge_sizes_thread;
  delete [] part_cut_sizes_thread;


  double avg_cut_size = (double)cut_size / (double)num_parts;
  double* part_counts = new double[num_parts];
  double* part_weights = new double[num_parts];
  double* part_edge_weights = new double[num_parts];
  double* part_cut_weights = new double[num_parts];

  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;
  int thread_start;


#pragma omp for schedule(static) nowait
  for (int i = 0; i < num_verts; ++i)
    in_queue_next[i] = false;

while(t < edge_outer_iter)
{

#pragma omp for schedule(static)
  for (int i = 0; i < num_verts; ++i)
    queue[i] = i;

#pragma omp single
{
  num_swapped_1 = 0;
  queue_size = num_verts;
  next_size = 0;
  swapped = true;    

  max_e = 0.0;
  max_c = 0.0;
  for (int p = 0; p < num_parts; ++p)
  {
    if ((double)part_edge_sizes[p] / avg_edge_size > max_e)
      max_e = (double)part_edge_sizes[p] / avg_edge_size;
    if ((double)part_cut_sizes[p] / avg_cut_size > max_c)
      max_c = (double)part_cut_sizes[p] / avg_cut_size;
  }
  if (max_e < edge_balance)
  {
    max_e = edge_balance;
    weight_exponent_e = 1.0;
    weight_exponent_c *= max_c;
  }
  else
  {
    weight_exponent_e *= max_e / edge_balance;
    weight_exponent_c = 1.0;
  }
}

  int num_iter = 0;
  while (/*swapped &&*/ num_iter < edge_balance_iter)
  {
    for (int p = 0; p < num_parts; ++p)
    {
      part_weights[p] = vert_balance * avg_size / (double)part_sizes[p] - 1.0;   
      part_edge_weights[p] = max_e * avg_edge_size / (double)part_edge_sizes[p] - 1.0;
      part_cut_weights[p] = max_c * avg_cut_size / (double)part_cut_sizes[p] - 1.0;
      if (part_weights[p] < 0.0)
        part_weights[p] = 0.0;
      if (part_edge_weights[p] < 0.0)
        part_edge_weights[p] = 0.0;
      if (part_cut_weights[p] < 0.0)
        part_cut_weights[p] = 0.0;
    }

#pragma omp for schedule(guided) reduction(+:num_swapped_1) nowait
    for (int i = 0; i < queue_size; ++i)
    {
      int v = queue[i];
      in_queue[v] = false;
      int part = parts[v];
      for (int p = 0; p < num_parts; ++p)
        part_counts[p] = 0.0;

      unsigned out_degree = out_degree(g, v);
      int* outs = out_vertices(g, v);
      for (unsigned j = 0; j < out_degree; ++j)
      {
        int out = outs[j];
        int part_out = parts[out];
        part_counts[part_out] += 1.0;
      }
      
      int max_part = part;
      double max_val = 0.0;
      int part_count = (int)part_counts[part];
      int max_count = 0;
      for (int p = 0; p < num_parts; ++p)
      {
        int count_init = (int)part_counts[p];
        if (part_weights[p] > 0.0 && part_edge_weights[p] > 0.0 && part_cut_weights[p] > 0.0)
          part_counts[p] *= (part_edge_weights[p]*weight_exponent_e * part_cut_weights[p]*weight_exponent_c);
        else
          part_counts[p] = 0.0;
        
        if (part_counts[p] > max_val)
        {
          max_val = part_counts[p];
          max_count = count_init;
          max_part = p;
        }
      }

      if (max_part != part)
      {
        parts[v] = max_part;
        ++num_swapped_1;
        int diff_part = 2*part_count - out_degree;
        int diff_max_part = out_degree - 2*max_count;
        int diff_cut = diff_part + diff_max_part;
    #pragma omp atomic
        cut_size += diff_cut;
    #pragma omp atomic
        part_cut_sizes[part] += diff_part;
    #pragma omp atomic
        part_cut_sizes[max_part] += diff_max_part;
    #pragma omp atomic
        --part_sizes[part];
    #pragma omp atomic
        ++part_sizes[max_part];
    #pragma omp atomic
        part_edge_sizes[part] -= out_degree;
    #pragma omp atomic
        part_edge_sizes[max_part] += out_degree;

        if (!in_queue_next[v])
        {
          in_queue_next[v] = true;
          thread_queue[thread_queue_size++] = v;

          if (thread_queue_size == THREAD_QUEUE_SIZE)
          {
#pragma omp atomic capture
            thread_start = next_size += thread_queue_size;
            
            thread_start -= thread_queue_size;
            for (int l = 0; l < thread_queue_size; ++l)
              queue_next[thread_start+l] = thread_queue[l];
            thread_queue_size = 0;
          }
        }
        for (int j = 0; j < out_degree; ++j)
        {
          if (!in_queue_next[outs[j]])
          {
            in_queue_next[outs[j]] = true;
            thread_queue[thread_queue_size++] = outs[j];

            if (thread_queue_size == THREAD_QUEUE_SIZE)
            {
#pragma omp atomic capture
              thread_start = next_size += thread_queue_size;
              
              thread_start -= thread_queue_size;
              for (int l = 0; l < thread_queue_size; ++l)
                queue_next[thread_start+l] = thread_queue[l];
              thread_queue_size = 0;
            }
          }
        }

        avg_cut_size = cut_size / num_parts;
        part_weights[part] = vert_balance * avg_size / (double)part_sizes[part] - 1.0;   
        part_edge_weights[part] = max_e * avg_edge_size / (double)part_edge_sizes[part] - 1.0;
        part_cut_weights[part] = max_c * avg_cut_size / (double)part_cut_sizes[part] - 1.0;

        part_weights[max_part] = vert_balance * avg_size / (double)part_sizes[max_part]  - 1.0;   
        part_edge_weights[max_part] = max_e * avg_edge_size / (double)part_edge_sizes[max_part] - 1.0;
        part_cut_weights[max_part] = max_c * avg_cut_size / (double)part_cut_sizes[max_part] - 1.0;

        if (part_weights[part] < 0.0)
          part_weights[part] = 0.0;
        if (part_edge_weights[part] < 0.0)
          part_edge_weights[part] = 0.0;
        if (part_cut_weights[part] < 0.0)
          part_cut_weights[part] = 0.0;

        if (part_weights[max_part] < 0.0)
          part_weights[max_part] = 0.0;
        if (part_edge_weights[max_part] < 0.0)
          part_edge_weights[max_part] = 0.0;
        if (part_cut_weights[max_part] < 0.0)
          part_cut_weights[max_part] = 0.0;
      }
    }

#pragma omp atomic capture
    thread_start = next_size += thread_queue_size;
    
    thread_start -= thread_queue_size;
    for (int l = 0; l < thread_queue_size; ++l)
      queue_next[thread_start+l] = thread_queue[l];
    thread_queue_size = 0;
    
#pragma omp barrier

    ++num_iter;
#pragma omp single
{
#if VERBOSE
    printf("%d -- V: %2.2lf   E: %2.2lf, %lf   C: %2.2lf, %lf\n", num_swapped_1, vert_balance, max_e, weight_exponent_e, max_c, weight_exponent_c);
#endif
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;
    bool* temp_b = in_queue;
    in_queue = in_queue_next;
    in_queue_next = temp_b;
    queue_size = next_size;
    next_size = 0;

    int p;
    int count;

    max_e = 0.0;
    max_c = 0.0;
    for (int p = 0; p < num_parts; ++p)
    {
      if ((double)part_edge_sizes[p] / avg_edge_size > max_e)
        max_e = (double)part_edge_sizes[p] / avg_edge_size;
      if ((double)part_cut_sizes[p] / avg_cut_size > max_c)
        max_c = (double)part_cut_sizes[p] / avg_cut_size;
    }
    if (max_e < edge_balance)
    {
      max_e = edge_balance;
      weight_exponent_e = 1.0;
      weight_exponent_c *= max_c;
    }
    else
    {
      weight_exponent_e *= max_e / edge_balance;
      weight_exponent_c = 1.0;
    }


    if (num_swapped_1)
      swapped = true;
    else
      swapped = false;

    num_swapped_1 = 0;

#if OUTPUT_STEP
  evaluate_quality_step(g, "EdgeCutBalance", parts, num_parts);
#endif
}
  } // end while

#pragma omp for schedule(static)
  for (int i = 0; i < num_verts; ++i)
    queue[i] = i;

#pragma omp single
{
  num_swapped_2 = 0;
  queue_size = num_verts;
  next_size = 0;
  swapped = true;
}

  num_iter = 0;
  while (/*swapped &&*/ num_iter < edge_refine_iter)
  {
    for (int p = 0; p < num_parts; ++p)
    {
      part_weights[p] = vert_balance * avg_size / (double)part_sizes[p] - 1.0;   
      part_edge_weights[p] = max_e * avg_edge_size / (double)part_edge_sizes[p] - 1.0;
      part_cut_weights[p] = max_c * avg_cut_size / (double)part_cut_sizes[p] - 1.0;
      if (part_weights[p] < 0.0)
        part_weights[p] = 0.0;
      if (part_edge_weights[p] < 0.0)
        part_edge_weights[p] = 0.0;
      if (part_cut_weights[p] < 0.0)
        part_cut_weights[p] = 0.0;
    }
    
#pragma omp for schedule(guided) reduction(+:num_swapped_2) nowait  
    for (int i = 0; i < queue_size; ++i)
    {
      int v = queue[i];
      in_queue[v] = false;
      for (int p = 0; p < num_parts; ++p)
        part_counts[p] = 0;

      int part = parts[v];
      unsigned out_degree = out_degree(g, v);
      int* outs = out_vertices(g, v);
      for (unsigned j = 0; j < out_degree; ++j)
      {
        int out = outs[j];
        int part_out = parts[out];
        part_counts[part_out]++;
      }

      int max_part = -1;
      int max_count = -1;
      int part_count = part_counts[part];
      for (int p = 0; p < num_parts; ++p)
        if (part_counts[p] > max_count)
        {
          max_count = part_counts[p];
          max_part = p;
        }

      if (max_part != part)
      {
        double new_max_imb = (double)(part_sizes[max_part] + 1) / avg_size;
        double new_max_edge_imb = (double)(part_edge_sizes[max_part] + out_degree) / avg_edge_size;
        double new_max_cut_imb = (double)(part_cut_sizes[max_part] + out_degree - 2*max_count) / avg_cut_size;
        double new_cut_imb = (double)(part_cut_sizes[part] + 2*part_count - out_degree) / avg_cut_size;
        if ( new_max_imb < vert_balance && 
          new_max_edge_imb < max_e && 
          new_max_cut_imb < max_c && new_cut_imb < max_c)
        {
          ++num_swapped_2;
          parts[v] = max_part;
          int diff_part = 2*part_count - out_degree;
          int diff_max_part = out_degree - 2*max_count;
          int diff_cut = diff_part + diff_max_part;
      #pragma omp atomic
          cut_size += diff_cut;
      #pragma omp atomic
          part_cut_sizes[part] += diff_part;
      #pragma omp atomic
          part_cut_sizes[max_part] += diff_max_part;
      #pragma omp atomic
          ++part_sizes[max_part];
      #pragma omp atomic
          --part_sizes[part];
      #pragma omp atomic
          part_edge_sizes[max_part] += out_degree;
      #pragma omp atomic
          part_edge_sizes[part] -= out_degree;

          avg_cut_size = cut_size / num_parts;

          if (!in_queue_next[v])
          {
            in_queue_next[v] = true;
            thread_queue[thread_queue_size++] = v;

            if (thread_queue_size == THREAD_QUEUE_SIZE)
            {
#pragma omp atomic capture
              thread_start = next_size += thread_queue_size;
              
              thread_start -= thread_queue_size;
              for (int l = 0; l < thread_queue_size; ++l)
                queue_next[thread_start+l] = thread_queue[l];
              thread_queue_size = 0;
            }
          }
          for (int j = 0; j < out_degree; ++j) 
          {
            if (!in_queue_next[outs[j]])
            {
              in_queue_next[outs[j]] = true;
              thread_queue[thread_queue_size++] = outs[j];

              if (thread_queue_size == THREAD_QUEUE_SIZE)
              {
#pragma omp atomic capture
                thread_start = next_size += thread_queue_size;
                
                thread_start -= thread_queue_size;
                for (int l = 0; l < thread_queue_size; ++l)
                  queue_next[thread_start+l] = thread_queue[l];
                thread_queue_size = 0;
              }
            }
          }
        }
      }
    }        

#pragma omp atomic capture
    thread_start = next_size += thread_queue_size;
    
    thread_start -= thread_queue_size;
    for (int l = 0; l < thread_queue_size; ++l)
      queue_next[thread_start+l] = thread_queue[l];
    thread_queue_size = 0;

#pragma omp barrier


    ++num_iter;
#pragma omp single
{
#if VERBOSE
    printf("%d -- V: %2.2lf  E: %2.2lf  C: %2.2lf\n", num_swapped_2, vert_balance, max_e, max_c);
#endif
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;
    bool* temp_b = in_queue;
    in_queue = in_queue_next;
    in_queue_next = temp_b;
    queue_size = next_size;
    next_size = 0;

    if (num_swapped_2)
      swapped = true;
    else
      swapped = false;

    num_swapped_2 = 0;  

    max_e = 0.0;
    max_c = 0.0;
    for (int p = 0; p < num_parts; ++p)
    {
      if ((double)part_edge_sizes[p] / avg_edge_size > max_e)
        max_e = (double)part_edge_sizes[p] / avg_edge_size;
      if ((double)part_cut_sizes[p] / avg_cut_size > max_c)
        max_c = (double)part_cut_sizes[p] / avg_cut_size;
    }
    if (max_e < edge_balance)
    {
      max_e = edge_balance;
      weight_exponent_e = 1.0;
      weight_exponent_c *= max_c;
    }
    else
    {
      weight_exponent_e *= max_e / edge_balance;
      weight_exponent_c = 1.0;
    }

#if OUTPUT_STEP
  evaluate_quality_step(g, "EdgeCutRefine", parts, num_parts);
#endif
}
  }

#pragma omp single
{
  if (max_e > edge_balance*1.01 && t == edge_outer_iter-1 && num_tries < 3)
  {
    --t;
    if (max_e < running_max_e*0.99)
    {
      running_max_e = max_e;
      printf("Edge balance missed, attempting further iterations: (%2.3lf)\n", max_e);
    }
    else
      ++num_tries;
  }
  else
    ++t;
}

} // end for

  delete [] part_counts;
  delete [] part_weights;
  delete [] part_edge_weights;
  delete [] part_cut_weights;

} // end par


  delete [] part_sizes;
  delete [] part_edge_sizes;
  delete [] part_cut_sizes;
  delete [] queue;
  delete [] queue_next;
  delete [] in_queue;
  delete [] in_queue_next;
}
