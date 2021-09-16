
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
  int* part_sizes = new int[num_parts];
  int queue_size = num_parts;
  int queue_size_next = 0;
  int max_part_size = num_verts / num_parts * 2;

  for (int i = 0; i < num_parts; ++i)
    part_sizes[i] = 0;

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
    int vert = (int)((unsigned)(xs1024star_next(&xs)) % (unsigned)num_verts);
    while (parts[vert] != -1) {vert = (int)xs1024star_next(&xs) % num_verts;}
    parts[vert] = i;
    queue[i] = vert;
    part_sizes[i] = 1;
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
        if (parts[out] == -1 && part_sizes[part] < max_part_size) {
          parts[out] = part;
          add_to_queue(thread_queue, thread_queue_size, 
                        queue_next, queue_size_next, out);
      #pragma omp atomic
          ++part_sizes[part];
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

