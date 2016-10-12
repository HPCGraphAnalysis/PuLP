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

extern int seed;


// multi-source bfs with random start points
int* init_nonrandom(pulp_graph_t& g, int num_parts, int* parts)
{
  int num_verts = g.n;
  int* queue = new int[num_verts*QUEUE_MULTIPLIER];
  int* queue_next = new int[num_verts*QUEUE_MULTIPLIER];
  int queue_size = num_parts;
  int next_size = 0;

#pragma omp parallel
{  
  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;
  int thread_start;

  xs1024star_t xs;
  xs1024star_seed((unsigned long)(seed + omp_get_thread_num()), &xs);

#pragma omp for
  for (int i = 0; i < num_verts; ++i)
    parts[i] = -1;

#pragma omp single
{
  for (int i = 0; i < num_parts; ++i)
  {
    int vert = (int)xs1024star_next(&xs) % num_verts;
    while (parts[vert] != -1) {vert = (int)xs1024star_next(&xs) % num_verts;}
    parts[vert] = i;
    queue[i] = vert;
  }
}

  while (queue_size > 0)
  {
#pragma omp for schedule(guided) nowait
    for (int i = 0; i < queue_size; ++i)
    {
      int vert = queue[i];
      int part = parts[vert];
      long out_degree = out_degree(g, vert);
      int* outs = out_vertices(g, vert);
      for (long j = 0; j < out_degree; ++j)
      {
        int out = outs[j];
        if (parts[out] == -1)
        {
          parts[out] = part;
          thread_queue[thread_queue_size++] = out;

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
#pragma omp atomic capture
    thread_start = next_size += thread_queue_size;
    
    thread_start -= thread_queue_size;
    for (int l = 0; l < thread_queue_size; ++l)
      queue_next[thread_start+l] = thread_queue[l];
    thread_queue_size = 0;

#pragma omp barrier

#pragma omp single
{
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;

    queue_size = next_size;
    next_size = 0;
}
  } // end while

#pragma omp for
  for (int i = 0; i < num_verts; ++i)
    if (parts[i] == -1)
      parts[i] = (int)xs1024star_next(&xs) % num_parts;
} // end parallel
  
#if OUTPUT_STEP
  evaluate_quality_step(g, "InitNonrandom", parts, num_parts);
#endif

  delete [] queue;
  delete [] queue_next;

  return parts;
}




// multi-source bfs with random start points
// constrain maximal size of part, randomly assign any remaining
int* init_nonrandom_constrained(pulp_graph_t& g, int num_parts, int* parts)
{
  int num_verts = g.n;

  int* queue = new int[num_verts*QUEUE_MULTIPLIER];
  int* queue_next = new int[num_verts*QUEUE_MULTIPLIER];
  int* part_sizes = new int[num_parts];
  int queue_size = num_parts;
  int next_size = 0;
  int max_part_size = num_verts / num_parts * 2;

#pragma omp parallel
{  
  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;
  int thread_start;

  xs1024star_t xs;
  xs1024star_seed((unsigned long)(seed + omp_get_thread_num()), &xs);

#pragma omp for
  for (int i = 0; i < num_verts; ++i)
    parts[i] = -1;

#pragma omp single
{
  for (int i = 0; i < num_parts; ++i)
  {
    int vert = (int)xs1024star_next(&xs) % num_verts;
    while (parts[vert] != -1) {vert = xs1024star_next(&xs) % num_verts;}
    parts[vert] = i;
    queue[i] = vert;
    part_sizes[i] = 1;
  }
}

  while (queue_size > 0)
  {
#pragma omp for schedule(guided) nowait
    for (int i = 0; i < queue_size; ++i)
    {
      int vert = queue[i];
      int part = parts[vert];
      long out_degree = out_degree(g, vert);
      int* outs = out_vertices(g, vert);
      for (long j = 0; j < out_degree; ++j)
      {
        int out = outs[j];
        if (parts[out] == -1)
        {
          if (part_sizes[part] < max_part_size)
            parts[out] = part;
          else
            parts[out] = rand() % num_parts;

      #pragma omp atomic
          ++part_sizes[parts[out]];

          thread_queue[thread_queue_size++] = out;
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
#pragma omp atomic capture
    thread_start = next_size += thread_queue_size;
    
    thread_start -= thread_queue_size;
    for (int l = 0; l < thread_queue_size; ++l)
      queue_next[thread_start+l] = thread_queue[l];
    thread_queue_size = 0;

#pragma omp barrier

#pragma omp single
{
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;

    queue_size = next_size;
    next_size = 0;
}
  } // end while

#pragma omp for
  for (int i = 0; i < num_verts; ++i)
    if (parts[i] == -1)
      parts[i] = xs1024star_next(&xs) % num_parts;

} // end parallel
  
#if OUTPUT_STEP
  evaluate_quality_step(g, "InitNonrandom2", parts, num_parts);
#endif

  delete [] queue;
  delete [] queue_next;
  delete [] part_sizes;

  return parts;
}

