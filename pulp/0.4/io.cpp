
#include <omp.h>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdint>

#include "io.h"
#include "graph.h"
#include "util.h"

void write_parts(char* filename, int num_verts, int* parts)
{
  FILE* outfile = NULL;
  outfile = fopen(filename, "w");

  for (int i = 0; i < num_verts; ++i)
    fprintf(outfile, "%d\n", parts[i]);

  fclose(outfile);
}


void read_bin(char* filename,
 int& num_verts, int& num_edges,
 int*& srcs, int*& dsts)
{
  double elt = omp_get_wtime();
  printf("Begin read_bin()\n");
  
  num_verts = 0;
#pragma omp parallel
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  FILE *infp = fopen(filename, "rb");
  if(infp == NULL) {
    printf("%d - load_graph_edges() unable to open input file", tid);
    exit(0);
  }

  fseek(infp, 0L, SEEK_END);
  uint64_t file_size = ftell(infp);
  fseek(infp, 0L, SEEK_SET);

  uint64_t nedges_global = file_size/(2*sizeof(uint32_t));

#pragma omp single
{
  num_edges = (long)nedges_global;
  srcs = new int[num_edges];
  dsts = new int[num_edges];
}

  uint64_t read_offset_start = tid*2*sizeof(uint32_t)*(nedges_global/nthreads);
  uint64_t read_offset_end = (tid+1)*2*sizeof(uint32_t)*(nedges_global/nthreads);

  if (tid == nthreads - 1)
    read_offset_end = 2*sizeof(uint32_t)*nedges_global;

  uint64_t nedges = (read_offset_end - read_offset_start)/(2*sizeof(uint32_t));
  uint32_t* edges_read = (uint32_t*)malloc(2*nedges*sizeof(uint32_t));
  if (edges_read == NULL) {
    printf("%d - load_graph_edges(), unable to allocate buffer", tid);
    exit(0);
  }

  fseek(infp, read_offset_start, SEEK_SET);
  fread(edges_read, nedges, 2*sizeof(uint32_t), infp);
  fclose(infp);
  printf(".");

  uint64_t array_offset = (uint64_t)tid*(nedges_global/nthreads);
  uint64_t counter = 0;
  for (uint64_t i = 0; i < nedges; ++i) {
    int src = (int)edges_read[counter++];
    int dst = (int)edges_read[counter++];
    srcs[array_offset+i] = src;
    dsts[array_offset+i] = dst;
  }

  free(edges_read);
  printf(".");

#pragma omp barrier

#pragma omp for reduction(max:num_verts)
  for (uint64_t i = 0; i < nedges_global; ++i)
    if (srcs[i] > num_verts)
      num_verts = srcs[i];
#pragma omp for reduction(max:num_verts)
  for (uint64_t i = 0; i < nedges_global; ++i)
    if (dsts[i] > num_verts)
      num_verts = dsts[i]; 
           
} // end parallel

  num_edges *= 2;
  num_verts += 1;
  printf("Done read_bin(): %lf (s)\n", omp_get_wtime() - elt); 
}


void read_edge(char* filename,
  int& num_verts, int& num_edges,
  int*& srcs, int*& dsts)
{
  FILE* infile = fopen(filename, "r");
  char line[256];

  num_verts = 0;

  int count = 0;
  int cur_size = 1024*1024;
  srcs = (int*)malloc(cur_size*sizeof(int));
  dsts = (int*)malloc(cur_size*sizeof(int));

  while(fgets(line, 256, infile) != NULL) {
    if (line[0] == '%') continue;

    sscanf(line, "%d %d", &srcs[count], &dsts[count]);
    dsts[count+1] = srcs[count];
    srcs[count+1] = dsts[count];

    if (srcs[count] > num_verts)
      num_verts = srcs[count];
    if (dsts[count] > num_verts)
      num_verts = dsts[count];

    count += 2;
    if (count > cur_size) {
      cur_size *= 2;
      srcs = (int*)realloc(srcs, cur_size*sizeof(int));
      dsts = (int*)realloc(dsts, cur_size*sizeof(int));
    }
  }  
  num_edges = count;

  printf("Read: n: %d, m: %d\n", num_verts, num_edges);

  fclose(infile);

  return;
}

