
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cassert>

#include "coarsen.h"
#include "pulp.h"
#include "thread.h"
#include "graph.h"
#include "fast_ts_map.h"

int get_contracted_edges_greedy(graph* g, 
  int*& contracted_edges, int& num_contracted_edges)
{
  double elt = omp_get_wtime();
  printf("Begin get_contracted_edges()\n");
  
  contracted_edges = new int[g->num_verts*2];
  num_contracted_edges = 0;
  
  bool* contracted = new bool[g->num_verts];
  for (int i = 0; i < g->num_verts; ++i)
    contracted[i] = false;
  
  for (int v = 0; v < g->num_verts; ++v) {
    if (contracted[v]) continue;
    
    int degree = out_degree(g, v);
    int* outs = out_vertices(g, v);
    for (int j = 0; j < degree; ++j) {
      int out = outs[j];
      if (contracted[out]) {
        continue;
      } else {
        contracted_edges[num_contracted_edges++] = v;
        contracted_edges[num_contracted_edges++] = out;
        contracted[v] = true;
        contracted[out] = true;
        break;
      }
    }
  }
  
  delete [] contracted;
  
  printf("Found %d edges to contract\n", num_contracted_edges / 2);
  printf("Done get_contracted_edges(): %lf (s)\n", omp_get_wtime() - elt); 
  
  return 0;
}

int get_contracted_edges_hem(graph* g, 
  int*& contracted_edges, int& num_contracted_edges)
{
  double elt = omp_get_wtime();
  printf("Begin get_contracted_edges()\n");
  
  contracted_edges = new int[g->num_verts*2];
  num_contracted_edges = 0;
  
  bool* contracted = new bool[g->num_verts];
  for (int i = 0; i < g->num_verts; ++i)
    contracted[i] = false;
  
  for (int v = 0; v < g->num_verts; ++v) {
    if (contracted[v]) continue;
    
    int max_weight = 0;
    int max_weight_vert = -1;
    int degree = out_degree(g, v);
    int* outs = out_vertices(g, v);
    int* weights = out_weights(g, v);
    for (int j = 0; j < degree; ++j) {
      int out = outs[j];
      if (contracted[out]) {
        continue;
      } else {
        if (weights[j] > max_weight) {
          max_weight = weights[j];
          max_weight_vert = out;
        }
      }
    }
    if (max_weight_vert != -1) {
      contracted_edges[num_contracted_edges++] = v;
      contracted_edges[num_contracted_edges++] = max_weight_vert;
      contracted[v] = true;
      contracted[max_weight_vert] = true;
    }
  }
  
  delete [] contracted;
  
  printf("Found %d edges to contract\n", num_contracted_edges / 2);
  printf("Done get_contracted_edges(): %lf (s)\n", omp_get_wtime() - elt); 
  
  return 0;
}

int get_contracted_edges_hec(graph* g, 
  int*& contracted_edges, int& num_contracted_edges)
{
  double elt = omp_get_wtime();
  printf("Begin get_contracted_edges()\n");
  
  contracted_edges = new int[g->num_verts*2];
  num_contracted_edges = 0;
  
  for (int v = 0; v < g->num_verts; ++v) {    
    int max_weight = 0;
    int max_weight_vert = -1;
    int degree = out_degree(g, v);
    int* outs = out_vertices(g, v);
    int* weights = out_weights(g, v);
    for (int j = 0; j < degree; ++j) {
      int out = outs[j];
      if (weights[j] >= max_weight) {
        max_weight = weights[j];
        max_weight_vert = out;
      }
    }
    if (max_weight_vert != -1) {
      contracted_edges[num_contracted_edges++] = v;
      contracted_edges[num_contracted_edges++] = max_weight_vert;
    }
  }
  
  printf("Found %d edges to contract\n", num_contracted_edges / 2);
  printf("Done get_contracted_edges(): %lf (s)\n", omp_get_wtime() - elt); 
  
  return 0;
}


int get_coarse_edges(graph* g,
  int* contracted_edges, int num_contracted_edges,
  int& num_verts_new, int& num_edges_new,
  int*& srcs_new, int*& dsts_new, int*& vert_weights_new, int*& sd_weights_new)
{
  double elt = omp_get_wtime();
  printf("Begin get_coarse_edges()\n");
  
  int* vid_map = new int[g->num_verts];
#pragma omp parallel for
  for (int i = 0; i < g->num_verts; ++i)
    vid_map[i] = -1;
  
  double elt2 = omp_get_wtime();
  
  num_verts_new = 0;
#pragma omp parallel for
  for (int i = 0; i < num_contracted_edges; i += 2) {
    int u = contracted_edges[i];
    int v = contracted_edges[i+1];
    if (vid_map[u] == -1 && vid_map[v] != -1)
      vid_map[u] = vid_map[v];
    else if (vid_map[v] == -1 && vid_map[u] != -1)
      vid_map[v] = vid_map[u];
    else {
      int new_vid = 0;
  #pragma omp atomic capture
      { new_vid = num_verts_new ; num_verts_new++; }
      
      vid_map[u] = new_vid;
      vid_map[v] = new_vid;
    }
  } 
  //assert(num_contracted_edges / 2 == num_verts_new);  
  printf("Time loop 1: %lf\n", omp_get_wtime() - elt2);
  elt2 = omp_get_wtime();

#pragma omp parallel for
  for (int i = 0; i < g->num_verts; ++i) {
    if (vid_map[i] == -1) {
      int new_vid = 0;
  #pragma omp atomic capture
      { new_vid = num_verts_new ; num_verts_new++; }
      
      vid_map[i] = new_vid;
    }
  }
  printf("Time loop 2: %lf\n", omp_get_wtime() - elt2);
  elt2 = omp_get_wtime();
  
  g->contracted_verts = new int[g->num_verts];
#pragma omp parallel for
  for (int i = 0; i < g->num_verts; ++i) {
    g->contracted_verts[i] = -1;
  }  
#pragma omp parallel for
  for (int i = 0; i < num_contracted_edges; i += 2) {
    int u = contracted_edges[i];
    int v = contracted_edges[i+1];
    int index1 = vid_map[u];
    int index2 = vid_map[v];
    //assert(index1 == index2);
    //assert(contracted_verts[index1].u == -1);
    //assert(contracted_verts[index1].v == -1);
    g->contracted_verts[u] = index1;
    g->contracted_verts[v] = index2;
  }
#pragma omp parallel for
  for (int i = 0; i < g->num_verts; ++i) {
    int index = vid_map[i];
    if (g->contracted_verts[i] == -1) {
      //assert(contracted_verts[index].v == -1);
      g->contracted_verts[i] = index;
    }
  }  
  printf("Time loop 3: %lf\n", omp_get_wtime() - elt2);
  elt2 = omp_get_wtime();
  // for (int i = 0; i < g->num_verts; ++i) { 
  //   int index = vid_map[i];
  //   assert(contracted_verts[index].u != -1);
  //   assert(contracted_verts[index].v != -1);
  // }
  
  fast_ts_map* map = new fast_ts_map;
  init_map(map, g->num_edges*4);
  
  vert_weights_new = new int[num_verts_new];
#pragma omp parallel for
  for (int i = 0; i < num_verts_new; ++i)
    vert_weights_new[i] = 0;
  
#pragma omp parallel for schedule(guided)
  for (int v = 0; v < g->num_verts; ++v) {
    int x = vid_map[v];

#pragma omp atomic
    vert_weights_new[x] += g->vert_weights[v];
    
    int degree = out_degree(g, v);
    int* outs = out_vertices(g, v);
    int* weights = out_weights(g, v);
    for (int j = 0; j < degree; ++j) {
      int y = vid_map[outs[j]];
      int w_xy = weights[j];
      
      if (x < y) test_set_value(map, (uint32_t)x, (uint32_t)y, w_xy);
      else if (x > y) test_set_value(map, (uint32_t)y, (uint32_t)x, w_xy);
    }
  }
  printf("Time loop 4: %lf\n", omp_get_wtime() - elt2);
  elt2 = omp_get_wtime();
  
  
  // vert_weights_new = new int[num_verts_new];
  // for (int i = 0; i < num_contracted_edges; i += 2) {
  //   int u = contracted_edges[i];
  //   int v = contracted_edges[i+1];
  //   int x = vid_map[u];
    
  //   int w_u = g->vert_weights[u];
  //   int w_v = g->vert_weights[v];
  //   vert_weights_new[x] = w_u + w_v;
    
  //   int degree = out_degree(g, u);
  //   int* outs = out_vertices(g, u);
  //   int* weights = out_weights(g, u);
  //   for (int j = 0; j < degree; ++j) {
  //     int y = vid_map[outs[j]];
  //     int w_xy = weights[j];
      
  //     if (x < y) test_set_value(map, (uint32_t)x, (uint32_t)y, w_xy);
  //     else if (x > y) test_set_value(map, (uint32_t)y, (uint32_t)x, w_xy);
  //   }
    
  //   degree = out_degree(g, v);
  //   outs = out_vertices(g, v);
  //   weights = out_weights(g, v);
  //   for (int j = 0; j < degree; ++j) {
  //     int y = vid_map[outs[j]];
  //     int w_xy = weights[j];
      
  //     if (x < y) test_set_value(map, (uint32_t)x, (uint32_t)y, w_xy);
  //     else if (x > y) test_set_value(map, (uint32_t)y, (uint32_t)x, w_xy);
  //   }
  // }
  // for (int v = 0; v < g->num_verts; ++v) {
  //   int x = vid_map[v];
  //   if (x < num_contracted_edges / 2) continue;
  //   vert_weights_new[x] = g->vert_weights[v];
    
  //   int degree = out_degree(g, v);
  //   int* outs = out_vertices(g, v);
  //   int* weights = out_weights(g, v);
  //   for (int j = 0; j < degree; ++j) {
  //     int y = vid_map[outs[j]];
  //     int w_xy = weights[j];
  //     //if (y < num_contracted_edges / 2) continue;
      
  //     if (x < y) test_set_value(map, (uint32_t)x, (uint32_t)y, w_xy);
  //     else if (x > y) test_set_value(map, (uint32_t)y, (uint32_t)x, w_xy);
  //   }
  // }
  num_edges_new = 0;

#pragma omp parallel for reduction(+:num_edges_new)
  for (int i = 0; i < (int)map->capacity; ++i) {
    if (map->arr[i].val == true) {
      ++num_edges_new;
    } 
  }
  
  srcs_new = new int[num_edges_new];
  dsts_new = new int[num_edges_new];
  sd_weights_new = new int[num_edges_new];
  num_edges_new = 0;
#pragma omp parallel 
{
  int thread_queue[THREAD_QUEUE_SIZE];
  int thread_queue_size = 0;
  
#pragma omp for
  for (int i = 0; i < (int)map->capacity; ++i) {
    if (map->arr[i].val == true) {
      uint64_t key = map->arr[i].key;
      int src = (int)((uint32_t)(key >> 32));
      int dst = (int)((uint32_t)(key & 0x00000000FFFFFFFF));
      
      add_to_queues(thread_queue, thread_queue_size, 
                   srcs_new, dsts_new, sd_weights_new, 
                   num_edges_new, 
                   src, dst, map->arr[i].count / 2);
    }
  }
  empty_queues(thread_queue, thread_queue_size, 
               srcs_new, dsts_new, sd_weights_new, 
               num_edges_new);
} // end parallel
  num_edges_new *= 2; // expected for csr creation
  printf("Time loop 5: %lf\n", omp_get_wtime() - elt2);
  elt2 = omp_get_wtime();
  
  delete map;
  
  printf("New graph will have %d verts and %d edges\n", 
    num_verts_new, num_edges_new / 2);
  printf("Done get_coarse_edges(): %lf (s)\n", omp_get_wtime() - elt); 
  
  return 0; 
}


graph* create_coarse_graph(int num_verts_new, int num_edges_new,
  int*& srcs_new, int*& dsts_new, int* vert_weights_new, int* sd_weights_new)
{
  double elt = omp_get_wtime();
  printf("Begin create_coarse_graph()\n");
  
  int* out_adjlist_new = NULL;
  int* out_offsets_new = NULL;
  int* edge_weights_new = NULL;
  
  create_csr_weighted(num_verts_new, num_edges_new,
    srcs_new, dsts_new, sd_weights_new,
    out_adjlist_new, out_offsets_new, edge_weights_new);
  delete [] srcs_new;
  delete [] dsts_new;
  delete [] sd_weights_new;

  graph* g_new = (graph*)malloc(sizeof(graph));
  g_new->num_verts = num_verts_new;
  g_new->num_edges = num_edges_new;
  g_new->out_adjlist = out_adjlist_new;
  g_new->out_offsets = out_offsets_new;
  g_new->vert_weights = vert_weights_new;
  g_new->edge_weights = edge_weights_new;
  g_new->vert_weights_sum = 0;
  g_new->edge_weights_sum = 0;
  g_new->contracted_verts = NULL;

  for (int i = 0; i < g_new->num_verts; ++i)
    g_new->vert_weights_sum += g_new->vert_weights[i];
  for (int i = 0; i < g_new->num_edges; ++i)
    g_new->edge_weights_sum += g_new->edge_weights[i];

  printf("Done create_coarse_graph(): %lf (s)\n", omp_get_wtime() - elt); 
  
  return g_new; 
}


int extrapolate_parts(graph* g_coarse, graph* g, 
  int* parts_coarse, int* parts)
{
  double elt = omp_get_wtime();
  printf("Begin extrapolate_parts()\n");
  
  for (int i = 0; i < g->num_verts; ++i)
    parts[i] = -1;
  
#pragma omp parallel for
  for (int i = 0; i < g->num_verts; ++i) {
    parts[i] = parts_coarse[g->contracted_verts[i]];
    
    // int u = g_coarse->contracted_verts[i].u;
    // int v = g_coarse->contracted_verts[i].v;
    // assert(u < g->num_verts);
    // assert(v < g->num_verts);
    // parts[u] = parts_coarse[i];
    // parts[v] = parts_coarse[i];
  }
  
  for (int i = 0; i < g->num_verts; ++i)
    assert(parts[i] >= 0);
  
  printf("Done extrapolate_parts(): %lf (s)\n", omp_get_wtime() - elt);
  
  return 0;
}