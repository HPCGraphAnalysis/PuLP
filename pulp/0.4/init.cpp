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

#include <omp.h>
#include <cstdlib>
#include <cstdio>

#include "init.h"
#include "pulp.h"
#include "thread.h"
#include "graph.h"
#include "rand.h"

extern int seed;

// multi-source bfs with random start points
int* init_bfs(graph* g, int num_parts, int* parts)
{
  double elt = omp_get_wtime();
  printf("Begin init_nonrandom()\n");
  
  int num_verts = g->num_verts;
  int* queue = new int[num_verts*QUEUE_MULTIPLIER];
  int* queue_next = new int[num_verts*QUEUE_MULTIPLIER];
  int queue_size = num_parts;
  int queue_size_next = 0;

#pragma omp parallel
{  
  int thread_queue[ THREAD_QUEUE_SIZE ];
  int thread_queue_size = 0;

  xs1024star_t xs;
  xs1024star_seed((unsigned long)(seed + omp_get_thread_num()), &xs);

#pragma omp for
  for (int i = 0; i < num_verts; ++i)
    parts[i] = -1;

#pragma omp single
{
  for (int i = 0; i < num_parts; ++i) {
    int vert = (int)xs1024star_next(&xs) % num_verts;
    while (parts[vert] != -1) {vert = (int)xs1024star_next(&xs) % num_verts;}
    parts[vert] = i;
    queue[i] = vert;
  }
}

  while (queue_size > 0) {
    
#pragma omp for schedule(guided) nowait
    for (int i = 0; i < queue_size; ++i) {
      int vert = queue[i];
      int part = parts[vert];
      long degree = out_degree(g, vert);
      int* outs = out_vertices(g, vert);
      for (long j = 0; j < degree; ++j) {
        int out = outs[j];
        if (parts[out] == -1) {
          parts[out] = part;
          add_to_queue(thread_queue, thread_queue_size, 
                        queue_next, queue_size_next, out);
        }
      }
    }

    empty_queue(thread_queue, thread_queue_size, 
                queue_next, queue_size_next);
#pragma omp barrier

#pragma omp single
{
    int* temp = queue;
    queue = queue_next;
    queue_next = temp;

    queue_size = queue_size_next;
    queue_size_next = 0;
}
  } // end while

#pragma omp for
  for (int i = 0; i < num_verts; ++i)
    if (parts[i] < 0)
      parts[i] = (unsigned)xs1024star_next(&xs) % num_parts;
} // end parallel
  
#if OUTPUT_STEP
  char step[] = "InitBFS";
  evaluate_quality_step(step, g, num_parts, parts);
#endif

  delete [] queue;
  delete [] queue_next;

  printf("Done init_nonrandom(): %lf (s)\n", omp_get_wtime() - elt); 
  
  return parts;
}

