#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include "graph.h"
#include "io.h"
#include "rand.h"
#include "util.h"


int create_csr_weighted(int num_verts, int num_edges,
  int* srcs, int* dsts, int* sd_weights,
  int*& out_adjlist, int*& out_offsets, int*& edge_weights)
{
  double elt = omp_get_wtime();
  printf("Begin create_csr_weighted()\n");

  out_adjlist = new int[num_edges];
  out_offsets = new int[num_verts+1];
  edge_weights = new int[num_edges];
  int* temp_counts = new int[num_verts];

#pragma omp parallel for
  for (int i = 0; i < num_edges; ++i)
    out_adjlist[i] = 0;
#pragma omp parallel for
  for (int i = 0; i < num_verts+1; ++i)
    out_offsets[i] = 0;
#pragma omp parallel for
  for (int i = 0; i < num_verts; ++i)
    temp_counts[i] = 0;

#pragma omp parallel for
  for (int i = 0; i < num_edges/2; ++i) {
#pragma omp atomic
    ++temp_counts[srcs[i]];
#pragma omp atomic
    ++temp_counts[dsts[i]];
  }
  parallel_prefixsums(temp_counts, out_offsets+1, num_verts);
  for (int i = 0; i < num_verts; ++i)
    assert(out_offsets[i+1] == out_offsets[i] + temp_counts[i]);
#pragma omp parallel for  
  for (int i = 0; i < num_verts; ++i)
    temp_counts[i] = out_offsets[i];
#pragma omp parallel for
  for (int i = 0; i < num_edges/2; ++i) {
    int index = -1;
    
    int src = srcs[i];
#pragma omp atomic capture
  { index = temp_counts[src]; temp_counts[src]++; }
    out_adjlist[index] = dsts[i];
    edge_weights[index] = sd_weights[i];
    
    int dst = dsts[i];
#pragma omp atomic capture
  { index = temp_counts[dst]; temp_counts[dst]++; }
    out_adjlist[index] = srcs[i];
    edge_weights[index] = sd_weights[i];
  }

  delete [] temp_counts;

  printf("Graph: n=%d, m=%d, davg=%d\n", 
    num_verts, num_edges/2, num_edges / num_verts / 2);
  printf("Done create_csr_weighted(): %lf (s)\n", omp_get_wtime() - elt); 
  
  return 0;
}

  
graph* create_graph(char* filename)
{
  double elt = omp_get_wtime();
  printf("Begin create_graph()\n");
  
  int* srcs = NULL;
  int* dsts = NULL;
  int* sd_weights = NULL;
  int num_verts = 0;
  int num_edges = 0;
  int* out_adjlist = NULL;
  int* out_offsets = NULL;
  int* edge_weights = NULL;
  int* vert_weights = NULL;
  pair* contracted_verts = NULL;

  read_bin(filename, num_verts, num_edges, srcs, dsts);
  
  edge_weights = new int[num_edges];
  vert_weights = new int[num_verts];
  contracted_verts = new pair[num_verts];
  sd_weights = new int[num_edges];
  
#pragma omp parallel for
  for (int i = 0; i < num_edges; ++i)
    sd_weights[i] = 1;
#pragma omp parallel for
  for (int i = 0; i < num_verts; ++i)
    vert_weights[i] = 1;
#pragma omp parallel for
  for (int i = 0; i < num_verts; ++i) {
    contracted_verts[i].u = i;
    contracted_verts[i].v = i;
  }
  
  create_csr_weighted(num_verts, num_edges, srcs, dsts, sd_weights,
    out_adjlist, out_offsets, edge_weights);
  delete [] srcs;
  delete [] dsts;
  delete [] sd_weights;

  graph* g = (graph*)malloc(sizeof(graph));
  g->num_verts = num_verts;
  g->num_edges = num_edges;
  g->out_adjlist = out_adjlist;
  g->out_offsets = out_offsets;
  g->vert_weights = vert_weights;
  g->edge_weights = edge_weights;
  g->vert_weights_sum = g->num_verts;
  g->edge_weights_sum = g->num_edges;
  g->contracted_verts = contracted_verts;
  
  printf("Done create_graph(): %lf (s)\n", omp_get_wtime() - elt);  
  
  return g;
}


int clear_graph(graph*& g)
{
  g->num_verts = 0;  
  g->num_edges = 0;
  delete [] g->out_adjlist;
  delete [] g->out_offsets;
  delete [] g->vert_weights;
  delete [] g->edge_weights;
  g->vert_weights_sum = 0;
  g->edge_weights_sum = 0;
  free(g);

  return 0;
}

int copy_graph(graph* g, graph* new_g)
{
  new_g->num_verts = g->num_verts;
  new_g->num_edges = g->num_edges;
  new_g->out_offsets = new int[g->num_verts+1];
  new_g->out_adjlist = new int[g->num_edges];
  new_g->vert_weights = new int[g->num_verts];
  new_g->edge_weights = new int[g->num_edges];
  new_g->vert_weights_sum = g->vert_weights_sum;
  new_g->edge_weights_sum = g->edge_weights_sum;

#pragma omp parallel for
  for (int i = 0; i < g->num_verts+1; ++i)
    new_g->out_offsets[i] = g->out_offsets[i];
#pragma omp parallel for
  for (long i = 0; i < g->num_edges; ++i)
    new_g->out_adjlist[i] = g->out_adjlist[i];
#pragma omp parallel for
  for (int i = 0; i < g->num_verts; ++i)
    new_g->vert_weights[i] = g->vert_weights[i];
#pragma omp parallel for
  for (long i = 0; i < g->num_edges; ++i)
    new_g->edge_weights[i] = g->edge_weights[i];
  
  
  return 0;
}
