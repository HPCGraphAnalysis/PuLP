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
'########:::::'###::::'##::::::::::'##::::'##:'########:'########::'########:
 ##.... ##:::'## ##::: ##:::::::::: ##:::: ##: ##.....:: ##.... ##:... ##..::
 ##:::: ##::'##:. ##:: ##:::::::::: ##:::: ##: ##::::::: ##:::: ##:::: ##::::
 ########::'##:::. ##: ##:::::::::: ##:::: ##: ######::: ########::::: ##::::
 ##.... ##: #########: ##::::::::::. ##:: ##:: ##...:::: ##.. ##:::::: ##::::
 ##:::: ##: ##.... ##: ##:::::::::::. ## ##::: ##::::::: ##::. ##::::: ##::::
 ########:: ##:::: ##: ########::::::. ###:::: ########: ##:::. ##:::: ##::::
........:::..:::::..::........::::::::...:::::........::..:::::..:::::..:::::
*/
void label_balance_verts(pulp_graph_t& g, int num_parts, int* parts,
  int vert_outer_iter, int vert_balance_iter, int vert_refine_iter,
  double vert_balance)
{
  int num_verts = g.n;
  int* part_sizes = new int[num_parts];
 
  for (int i = 0; i < num_parts; ++i)
    part_sizes[i] = 0;

  double avg_size = num_verts / num_parts;
  int num_swapped_1 = 0;
  int num_swapped_2 = 0;
  double max_v;
  double running_max_v = (double)num_verts;

  int* queue = new int[num_verts*QUEUE_MULTIPLIER];
  int* queue_next = new int[num_verts*QUEUE_MULTIPLIER];
  bool* in_queue = new bool[num_verts];
  bool* in_queue_next = new bool[num_verts];
  int queue_size = num_verts;
  int next_size = 0;
  int t = 0;
  int num_tries = 0;

#pragma omp parallel
{
  int* part_sizes_thread = new int[num_parts];
  for (int i = 0; i < num_parts; ++i) 
    part_sizes_thread[i] = 0;

#pragma omp for schedule(static) nowait
  for (int i = 0; i < num_verts; ++i)
    ++part_sizes_thread[parts[i]];

  for (int i = 0; i < num_parts; ++i) 
#pragma omp atomic
    part_sizes[i] += part_sizes_thread[i];

  delete [] part_sizes_thread;


  double* part_counts = new double[num_parts];
  double* part_weights = new double[num_parts];

  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;
  int thread_start;

  for (int p = 0; p < num_parts; ++p)
  {        
    part_weights[p] = vert_balance * avg_size / (double)part_sizes[p] - 1.0;
    if (part_weights[p] < 0.0)
      part_weights[p] = 0.0;
  }

while(t < vert_outer_iter)
{

#pragma omp for schedule(static) nowait
  for (int i = 0; i < num_verts; ++i)
    queue[i] = i;
#pragma omp for schedule(static)
  for (int i = 0; i < num_verts; ++i)
    in_queue_next[i] = false;

#pragma omp single
{
  num_swapped_1 = 0;
  queue_size = num_verts;
  next_size = 0;
}

  int num_iter = 0;
  while (/*swapped &&*/ num_iter < vert_balance_iter)
  {
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
        part_counts[part_out] += out_degree(g, out);
        //part_counts[part_out] += 1.0;//out_degree(g, out);
      }
      
      int max_part = part;
      double max_val = 0.0;
      for (int p = 0; p < num_parts; ++p)
      {
        part_counts[p] *= part_weights[p];
        
        if (part_counts[p] > max_val)
        {
          max_val = part_counts[p];
          max_part = p;
        }
      }

      if (max_part != part)
      {
        parts[v] = max_part;
        ++num_swapped_1;
    #pragma omp atomic
        --part_sizes[part];
    #pragma omp atomic
        ++part_sizes[max_part];
        
        part_weights[part] = vert_balance * avg_size / (double)part_sizes[part] - 1.0;
        part_weights[max_part] = vert_balance * avg_size / (double)part_sizes[max_part]  - 1.0;   
        
        if (part_weights[part] < 0.0)
          part_weights[part] = 0.0;
        if (part_weights[max_part] < 0.0)
          part_weights[max_part] = 0.0;

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
        for (unsigned j = 0; j < out_degree; ++j)
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
    printf("%d\n", num_swapped_1);
#endif
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;
    bool* temp_b = in_queue;
    in_queue = in_queue_next;
    in_queue_next = temp_b;
    queue_size = next_size;
    next_size = 0;

    num_swapped_1 = 0;

#if OUTPUT_STEP
  evaluate_quality(g, num_parts, parts);
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
}

  num_iter = 0;
  while (/*swapped &&*/ num_iter < vert_refine_iter)
  {
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
      for (int p = 0; p < num_parts; ++p)
        if (part_counts[p] > max_count)
        {
          max_count = part_counts[p];
          max_part = p;
        }

      if (max_part != part)
      {
        double new_max_imb = (double)(part_sizes[max_part] + 1) / avg_size;
        if ( new_max_imb < vert_balance)
        {
          ++num_swapped_2;
          parts[v] = max_part;
      #pragma omp atomic
          ++part_sizes[max_part];
      #pragma omp atomic
          --part_sizes[part];

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
          for (unsigned j = 0; j < out_degree; ++j) 
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
    printf("%d\n", num_swapped_2);
#endif
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;
    bool* temp_b = in_queue;
    in_queue = in_queue_next;
    in_queue_next = temp_b;
    queue_size = next_size;
    next_size = 0;

    num_swapped_2 = 0;

    max_v = 0.0;
    for (int p = 0; p < num_parts; ++p)
    {
      if ((double)part_sizes[p] / avg_size > max_v)
        max_v = (double)part_sizes[p] / avg_size;
    }
#if OUTPUT_STEP
  evaluate_quality(g, num_parts, parts);
#endif
}
  } // end while

#pragma omp single
{
  if (max_v > vert_balance*1.01 && t == vert_outer_iter-1 && num_tries < 3)
  {
    --t;
    if (max_v < running_max_v)
    {
      running_max_v = max_v;
      printf("Vertex balance missed, attempting further iterations: (%2.3lf)\n", max_v);
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

} // end par


  delete [] part_sizes;
  delete [] queue;
  delete [] queue_next;
  delete [] in_queue;
  delete [] in_queue_next;
}





/*
'########:::::'###::::'##::::::::::'##::::'##:'########:'########::'########:
 ##.... ##:::'## ##::: ##:::::::::: ##:::: ##: ##.....:: ##.... ##:... ##..::
 ##:::: ##::'##:. ##:: ##:::::::::: ##:::: ##: ##::::::: ##:::: ##:::: ##::::
 ########::'##:::. ##: ##:::::::::: ##:::: ##: ######::: ########::::: ##::::
 ##.... ##: #########: ##::::::::::. ##:: ##:: ##...:::: ##.. ##:::::: ##::::
 ##:::: ##: ##.... ##: ##:::::::::::. ## ##::: ##::::::: ##::. ##::::: ##::::
 ########:: ##:::: ##: ########::::::. ###:::: ########: ##:::. ##:::: ##::::
........:::..:::::..::........::::::::...:::::........::..:::::..:::::..:::::
*/
void label_balance_verts_weighted(
  pulp_graph_t& g, int num_parts, int* parts,
  int vert_outer_iter, int vert_balance_iter, int vert_refine_iter,
  double vert_balance)
{
  int num_verts = g.n;
  long* part_sizes = new long[num_parts];

  bool has_vwgts = (g.vertex_weights != NULL);
  bool has_ewgts = (g.edge_weights != NULL);
  if (!has_vwgts) g.vertex_weights_sum = g.n;
 
  for (int i = 0; i < num_parts; ++i)
    part_sizes[i] = 0;

  double avg_size = (double)g.vertex_weights_sum / (double)num_parts;
  int num_swapped_1 = 0;
  int num_swapped_2 = 0;
  double max_v;
  double running_max_v = (double)num_verts;

  int* queue = new int[num_verts*QUEUE_MULTIPLIER];
  int* queue_next = new int[num_verts*QUEUE_MULTIPLIER];
  bool* in_queue = new bool[num_verts];
  bool* in_queue_next = new bool[num_verts];
  int queue_size = num_verts;
  int next_size = 0;
  int t = 0;
  int num_tries = 0;

#pragma omp parallel
{
  long* part_sizes_thread = new long[num_parts];
  for (int i = 0; i < num_parts; ++i) 
    part_sizes_thread[i] = 0;

#pragma omp for schedule(static) nowait
  for (int i = 0; i < num_verts; ++i)
    if (has_vwgts)
      part_sizes_thread[parts[i]] += g.vertex_weights[i];
    else
      ++part_sizes_thread[parts[i]];

  for (int i = 0; i < num_parts; ++i) 
#pragma omp atomic
    part_sizes[i] += part_sizes_thread[i];

  delete [] part_sizes_thread;
#pragma omp barrier

  double* part_counts = new double[num_parts];
  double* part_weights = new double[num_parts];

  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;
  int thread_start;

  for (int p = 0; p < num_parts; ++p)
  {        
    part_weights[p] = vert_balance * avg_size / (double)part_sizes[p] - 1.0;
    if (part_weights[p] < 0.0)
      part_weights[p] = 0.0;
  }

while(t < vert_outer_iter)
{

#pragma omp for schedule(static) nowait
  for (int i = 0; i < num_verts; ++i)
    queue[i] = i;
#pragma omp for schedule(static)
  for (int i = 0; i < num_verts; ++i)
    in_queue_next[i] = false;

#pragma omp single
{
  num_swapped_1 = 0;
  queue_size = num_verts;
  next_size = 0;
}  

  int num_iter = 0;
  while (/*swapped &&*/ num_iter < vert_balance_iter)
  {
#pragma omp for schedule(guided) reduction(+:num_swapped_1) nowait
    for (int i = 0; i < queue_size; ++i)
    {
      int v = queue[i];
      in_queue[v] = false;
      int part = parts[v];
      int v_weight = 1;
      if (has_vwgts) v_weight = g.vertex_weights[v];

      for (int p = 0; p < num_parts; ++p)
        part_counts[p] = 0.0;

      unsigned out_degree = out_degree(g, v);
      int* outs = out_vertices(g, v);
      int* weights = out_weights(g, v);
      for (unsigned j = 0; j < out_degree; ++j)
      {
        int out = outs[j];
        int part_out = parts[out];
        double weight_out = 1.0;
        if (has_ewgts) weight_out = (double)weights[j];
        part_counts[part_out] += (double)out_degree(g, out)*weight_out;
      }
      
      int max_part = part;
      double max_val = 0.0;
      for (int p = 0; p < num_parts; ++p)
      {
        part_counts[p] *= part_weights[p];
        
        if (part_counts[p] > max_val)
        {
          max_val = part_counts[p];
          max_part = p;
        }
      }

      if (max_part != part)
      {
        parts[v] = max_part;
        ++num_swapped_1;
    #pragma omp atomic
        part_sizes[max_part] += v_weight;
    #pragma omp atomic
        part_sizes[part] -= v_weight;
        
        part_weights[part] = vert_balance * avg_size / (double)part_sizes[part] - 1.0;
        part_weights[max_part] = vert_balance * avg_size / (double)part_sizes[max_part]  - 1.0;   
        
        if (part_weights[part] < 0.0)
          part_weights[part] = 0.0;
        if (part_weights[max_part] < 0.0)
          part_weights[max_part] = 0.0;

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
        for (unsigned j = 0; j < out_degree; ++j)
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
    printf("%d\n", num_swapped_1);
#endif
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;
    bool* temp_b = in_queue;
    in_queue = in_queue_next;
    in_queue_next = temp_b;
    queue_size = next_size;
    next_size = 0;

    num_swapped_1 = 0;

#if OUTPUT_STEP
  evaluate_quality(g, num_parts, parts);
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
}

  num_iter = 0;
  while (/*swapped &&*/ num_iter < vert_refine_iter)
  {
#pragma omp for schedule(guided) reduction(+:num_swapped_2) nowait  
    for (int i = 0; i < queue_size; ++i)
    {
      int v = queue[i];
      in_queue[v] = false;      
      int part = parts[v];
      int v_weight = 1;
      if (has_vwgts) v_weight = g.vertex_weights[v];

      for (int p = 0; p < num_parts; ++p)
        part_counts[p] = 0;

      unsigned out_degree = out_degree(g, v);
      int* outs = out_vertices(g, v);
      int* weights = out_weights(g, v);
      for (unsigned j = 0; j < out_degree; ++j)
      {
        int out = outs[j];
        int part_out = parts[out];
        int out_weight = 1;
        if (has_ewgts) out_weight = weights[j];
        part_counts[part_out] += out_weight;
      }

      int max_part = -1;
      int max_count = -1;
      for (int p = 0; p < num_parts; ++p)
        if (part_counts[p] > max_count)
        {
          max_count = part_counts[p];
          max_part = p;
        }

      if (max_part != part)
      {
        double new_max_imb = (double)(part_sizes[max_part] + v_weight) / avg_size;
        if (new_max_imb < vert_balance)
        {
          ++num_swapped_2;
          parts[v] = max_part;
      #pragma omp atomic
          part_sizes[max_part] += v_weight;
      #pragma omp atomic
          part_sizes[part] -= v_weight;

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
          for (unsigned j = 0; j < out_degree; ++j) 
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
    printf("%d\n", num_swapped_2);
#endif
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;
    bool* temp_b = in_queue;
    in_queue = in_queue_next;
    in_queue_next = temp_b;
    queue_size = next_size;
    next_size = 0;

    num_swapped_2 = 0;

    max_v = 0.0;
    for (int p = 0; p < num_parts; ++p)
    {
      if ((double)part_sizes[p] / avg_size > max_v)
        max_v = (double)part_sizes[p] / avg_size;
    }
#if OUTPUT_STEP
  evaluate_quality(g, num_parts, parts);
#endif
}
  } // end while

#pragma omp single
{
  if (max_v > vert_balance*1.01 && t == vert_outer_iter-1 && num_tries < 3)
  {
    --t;
    if (max_v < running_max_v)
    {
      running_max_v = max_v;
      printf("Vertex balance missed, attempting further iterations: (%2.3lf)\n", max_v);
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

} // end par


  delete [] part_sizes;
  delete [] queue;
  delete [] queue_next;
  delete [] in_queue;
  delete [] in_queue_next;
}

