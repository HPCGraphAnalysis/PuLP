#ifndef _GRAPH_H_
#define _GRAPH_H_

//typedef int32_t pulp_part_t;
//typedef int32_t pulp_vert_t;

typedef struct {
  int u;
  int v;
} pair;

typedef struct {
  int num_verts;
  int num_edges;
  int* out_adjlist;
  int* out_offsets;
  
  int* vert_weights;
  int* edge_weights;
  
  int vert_weights_sum;
  int edge_weights_sum;
  
  pair* contracted_verts;
} graph;

inline int out_degree(graph* g, int v) 
{ 
  return g->out_offsets[v+1] - g->out_offsets[v];
}

inline int* out_vertices(graph* g, int v) 
{ 
  return &g->out_adjlist[g->out_offsets[v]];
}

inline int* out_weights(graph* g, int v) 
{ 
  return &g->edge_weights[g->out_offsets[v]];
}

int create_csr_weighted(int num_verts, int num_edges,
  int* srcs, int* dsts, int* sd_weights,
  int*& out_adjlist, int*& out_offsets, int*& edge_weights);

graph* create_graph(char* filename);

int clear_graph(graph*& g);

int copy_graph(graph* g, graph* new_g);

#endif
