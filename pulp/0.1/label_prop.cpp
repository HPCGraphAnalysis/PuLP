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

#include <vector>

using namespace std;


/*
'########::'########:::'#######::'########::
 ##.... ##: ##.... ##:'##.... ##: ##.... ##:
 ##:::: ##: ##:::: ##: ##:::: ##: ##:::: ##:
 ########:: ########:: ##:::: ##: ########::
 ##.....::: ##.. ##::: ##:::: ##: ##.....:::
 ##:::::::: ##::. ##:: ##:::: ##: ##::::::::
 ##:::::::: ##:::. ##:. #######:: ##::::::::
..:::::::::..:::::..:::.......:::..:::::::::
*/
int* label_prop(pulp_graph_t& g, int num_parts, int* parts,
  int label_prop_iter, double balance_vert_lower)
{
  int num_verts = g.n;

//#pragma omp parallel for
  for (int i = 0; i < num_verts; ++i)
    parts[i] = rand() % num_parts;
  int* part_sizes = new int[num_parts];
  for (int i = 0; i < num_parts; ++i)
    part_sizes[i] = 0;
  for (int i = 0; i < num_verts; ++i)
    ++part_sizes[parts[i]];

  int num_changes;
  bool colors_changed = true;
  int* queue = new int[num_verts*QUEUE_MULTIPLIER];
  int* queue_next = new int[num_verts*QUEUE_MULTIPLIER];
  bool* in_queue = new bool[num_verts];
  bool* in_queue_next = new bool[num_verts];
  int queue_size = num_verts;
  int next_size = 0;

  double avg_size = (double)num_verts / (double)num_parts;
  double min_size = avg_size * balance_vert_lower;

#pragma omp parallel
{
  int tid = omp_get_thread_num();

#pragma omp for schedule(static) nowait
  for (int i = 0; i < num_verts; ++i)
    queue[i] = i;
#pragma omp for schedule(static)
  for (int i = 0; i < num_verts; ++i)
    in_queue_next[i] = false;

  int* part_counts = new int[num_parts];
  vector<int> thread_queue;
  int thread_start;
  int num_iter = 0;

  while (/*colors_changed &&*/ num_iter < label_prop_iter)
  { 
    num_changes = 0;

#pragma omp for schedule(guided) reduction(+:num_changes)
    for (int i = 0; i < queue_size; ++i)
    {
      int v = queue[i];
      in_queue[v] = false;
      for (int j = 0; j < num_parts; ++j)
        part_counts[j] = 0;

      unsigned out_degree = out_degree(g, v);
      int* outs = out_vertices(g, v);
      for (unsigned j = 0; j < out_degree; ++j)
      {
        int out = outs[j];
        int part = parts[out];
        part_counts[part] += out_degree(g, out);
      }
      
      int part = parts[v];
      int max_count = -1;
      int max_part = -1;
      for (int j = 0; j < num_parts; ++j)
      {
        if (part_counts[j] > max_count)
        {
          max_count = part_counts[j];
          max_part = j;
        }
      }

      if (max_part != part && (part_sizes[part]-1) > (int)min_size)
      {
    #pragma omp atomic
        ++part_sizes[max_part];
    #pragma omp atomic
        --part_sizes[part];
        
        parts[v] = max_part;
        ++num_changes;

        if (!in_queue_next[v])
        {
          in_queue_next[v] = true;
          thread_queue.push_back(v);
        }
        for (int j = 0; j < out_degree; ++j)
          if (!in_queue_next[outs[j]])
          {
            in_queue_next[outs[j]] = true;
            thread_queue.push_back(outs[j]);
          }
      }
    }
    int thread_back = thread_queue.size();
    int* thread_data = thread_queue.data();

#pragma omp atomic capture
    thread_start = next_size += thread_back;

    thread_start -= thread_back;
    copy(thread_data, thread_data + thread_back, &queue_next[thread_start]);
    
#pragma omp barrier
    thread_queue.clear();
    
    ++num_iter;
#pragma omp single
{
#if VERBOSE
    printf("%d\n", next_size);
#endif
    if (num_changes)
      colors_changed = true;
    else 
      colors_changed = false;

    int* temp = queue;
    queue = queue_next;
    queue_next = temp;
    bool* temp_b = in_queue;
    in_queue = in_queue_next;
    in_queue_next = temp_b;

    queue_size = next_size;
    next_size = 0;

#if OUTPUT_STEP
  evaluate_quality_step(g, "LabelProp", parts, num_parts);
#endif
} // end single
  } // end while

  delete [] part_counts;
} // end par

  delete [] queue;
  delete [] queue_next;
  delete [] in_queue;
  delete [] in_queue_next;

  return parts;
}
