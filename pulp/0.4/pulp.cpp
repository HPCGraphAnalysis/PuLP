
#include <cstdlib>
#include <cstdio>

#include "pulp.h"
#include "graph.h"
#include "init.h"
#include "pulp_v.h"
#include "coarsen.h"

int pulp_run_coarsen(graph* g, int num_parts, int*& parts, double imb)
{
  graph** graphs = new graph*[MAX_COARSENING_LEVELS];
  graphs[0] = g;
  graph* g_new = g;
  
  int levels = 1;
  do {
    int* contracted_edges = NULL;
    int num_contract_edges = 0;    
    get_contracted_edges_hec(g_new, contracted_edges, num_contract_edges);
    
    int num_verts_new = 0;
    int num_edges_new = 0;
    int* srcs_new = NULL;
    int* dsts_new = NULL;
    int* vert_weights_new = NULL;
    int* sd_weights_new = NULL;
    get_coarse_edges(g_new, contracted_edges, num_contract_edges,
      num_verts_new, num_edges_new,
      srcs_new, dsts_new, vert_weights_new, sd_weights_new);
    
    g_new = create_coarse_graph(num_verts_new, num_edges_new,
      srcs_new, dsts_new, vert_weights_new, sd_weights_new);
    graphs[levels++] = g_new;
    
  } while (levels <= MAX_COARSENING_LEVELS &&
            g_new->num_verts > COARSE_VERTS_CUTOFF &&
            g_new->num_verts*2 < g_new->num_edges);
  
  --levels;
  int max_level = levels;
  graph* g_current = graphs[max_level];
  graph* g_next = NULL;
  int* parts_current = new int[g_current->num_verts];
  int* parts_next = NULL;
  init_bfs(g_current, num_parts, parts_current);
  while (levels > 0) {
    part_balance(g_current, num_parts, parts_current, imb);
    part_refine(g_current, num_parts, parts_current, imb);
    
    g_next = graphs[levels-1];
    parts_next = new int[g_next->num_verts];
    extrapolate_parts(g_current, g_next, parts_current, parts_next);
    evaluate_quality(g_current, num_parts, parts_current);
    evaluate_quality(g_next, num_parts, parts_next);
    //compare_cut(g_current, g_next, num_parts, parts_current, parts_next);
    delete [] parts_current;    
    parts_current = parts_next;
    g_current = g_next;
    
    --levels;
  }

  parts = parts_next;
  part_balance(g, num_parts, parts, imb);
  part_refine(g, num_parts, parts, imb);
  
  return 0;
}



void evaluate_quality(graph* g, int num_parts, int* parts)
{
  for (int i = 0; i < g->num_verts; ++i) {
    if (parts[i] < 0) {
      printf("invalid part: %d %d\n", i, parts[i]);
      exit(0);
    }
  }

  double comms_frac = 0.0;

  int num_verts = g->num_verts;
  int num_edges = g->num_edges;
  int num_comms = 0;
  bool** neighborhoods = new bool*[num_parts];
  bool** comms = new bool*[num_parts];
  int* part_sizes = new int[num_parts];
  int* num_comms_out = new int[num_parts];
  int* edge_cuts = new int[num_parts];
  int* edge_cuts_sizes = new int[num_parts];
  int* boundary_verts = new int[num_parts];
  bool** part_to_part = new bool*[num_parts];
  int* edges_per_part = new int[num_parts];

  for (int i = 0; i < num_parts; ++i) {
    part_sizes[i] = 0;
    num_comms_out[i] = 0;
    edge_cuts[i] = 0;
    edge_cuts_sizes[i] = 0;
    edges_per_part[i] = 0;
    boundary_verts[i] = 0;

    neighborhoods[i] = new bool[num_verts];
    comms[i] = new bool[num_verts];
    for (int j = 0; j < num_verts; ++j)
    {
      neighborhoods[i][j] = false;
      comms[i][j] = false;
    }

    part_to_part[i] = new bool[num_parts];
    for (int j = 0; j < num_parts; ++j)
      part_to_part[i][j] = false;
  }

  for (int v = 0; v < num_verts; ++v) {
    part_sizes[parts[v]] += g->vert_weights[v];

    int part = parts[v];
    neighborhoods[part][v] = true;
    bool boundary = false;

    int degree = out_degree(g, v);
    int* outs = out_vertices(g, v);
    int* weights = out_weights(g, v);
    for (int j = 0; j < degree; ++j) {
      int out = outs[j];
      neighborhoods[part][out] = true;

      int out_part = parts[out];
      if (out_part != part) {
        comms[part][out] = true;
        part_to_part[part][out_part] = true;
        edge_cuts_sizes[part] += weights[j];
        edge_cuts[part]++;

        boundary = true;
      }
      edges_per_part[part] += weights[j];
    }
    if (boundary)
      ++boundary_verts[part];
  }

  for (int i = 0; i < num_parts; ++i) {
    for (int j = 0; j < num_verts; ++j) {
      if (comms[i][j]) {
        ++num_comms_out[i];
        ++num_comms;
      }
    }
    for (int j = 0; j < num_parts; ++j)
      if (part_to_part[i][j])
        ++comms_frac;
  }

  int quality = 0;
  int edge_cut = 0;
  int edge_cut_size = 0;
  int max_vert_size = 0;
  int max_edge_size = 0.0;
  int max_comm_size = 0;
  int max_edge_cut = 0;
  int max_bound = 0;
  int num_bound = 0;
  for (int i = 0; i < num_parts; ++i)
  {
    printf("p: %d, v: %d, e: %u, com: %d, cut: %d, bound: %d\n", 
      i, part_sizes[i], 
      edges_per_part[i], num_comms_out[i], edge_cuts[i], boundary_verts[i]);
    
    quality += num_comms_out[i];
    edge_cut_size += edge_cuts_sizes[i];
    edge_cut += edge_cuts[i];
    num_bound += boundary_verts[i];

    if (edge_cuts[i] > max_edge_cut)
      max_edge_cut = edge_cuts[i];
    if (part_sizes[i] > max_vert_size)
      max_vert_size = part_sizes[i];
    if (edges_per_part[i] > max_edge_size)
      max_edge_size = edges_per_part[i];
    if (num_comms_out[i] > max_comm_size)
      max_comm_size = num_comms_out[i];
    if (boundary_verts[i] > max_bound)
      max_bound = boundary_verts[i];
  }

  comms_frac = comms_frac / (double)(num_parts*(num_parts-1));

  long avg_size_vert = g->vert_weights_sum / (long)num_parts;
  unsigned avg_size_edge = num_edges / (unsigned)num_parts;
  unsigned avg_comm_size = num_comms / (unsigned)num_parts;
  unsigned avg_edge_cut = edge_cut / (unsigned)num_parts;
  unsigned avg_bound = num_bound / (unsigned)num_parts;
  double max_overweight_v = (double)max_vert_size/(double)avg_size_vert;
  double max_overweight_e = (double)max_edge_size/(double)avg_size_edge;
  double max_overweight_cv = (double)max_comm_size/(double)avg_comm_size;
  double max_overweight_ec = (double)max_edge_cut/(double)avg_edge_cut;
  double max_overweight_b = (double)max_bound/(double)avg_bound;
  edge_cut /= 2;
  edge_cut_size /= 2;
  long unsigned comVol = (long unsigned)quality;
  double comVolRatio = (double)comVol / (double)(num_edges/2);
  long unsigned edgeCutSize = (long unsigned)edge_cut_size;
  long unsigned edgeCut = (long unsigned)edge_cut;
  double edgeCutRatio = (double)edgeCut / (double)(num_edges/2);
  double boundVertRatio = (double)num_bound / (double)(num_verts);

  printf("Edge Cut: %lu\n", edgeCutSize);
  printf("Edges Cut: %lu\n", edgeCut);
  printf("Max Cut: %u\n", max_edge_cut);
  printf("Comm Vol: %lu\n", comVol);
  printf("Num boundary verts: %u\n", num_bound);
  printf("Comm ratio: %9.3lf\n", comVolRatio);
  printf("Edge ratio: %9.3lf\n", edgeCutRatio);
  printf("Boundary ratio: %9.3lf\n", boundVertRatio);
  printf("Vert overweight: %9.3lf\n", max_overweight_v);
  printf("Edge overweight: %9.3lf\n", max_overweight_e);
  printf("Boundary overweight: %9.3lf\n", max_overweight_b);
  printf("CommVol overweight: %9.3lf, max: %u\n", max_overweight_cv, max_comm_size);
  printf("EdgeCut overweight: %9.3lf, max: %u\n", max_overweight_ec, max_edge_cut);
  
  for (int i = 0; i < num_parts; ++i) {
    delete [] neighborhoods[i];
    delete [] comms[i];
    delete [] part_to_part[i];
  }
  delete [] neighborhoods;
  delete [] comms;
  delete [] part_to_part;
  delete [] part_sizes;
  delete [] num_comms_out;
  delete [] edge_cuts;
  delete [] edge_cuts_sizes;
  delete [] boundary_verts;
  delete [] edges_per_part;
}

void evaluate_quality_step(char* step_name, 
  graph* g, int num_parts, int* parts)
{
  for (int i = 0; i < g->num_verts; ++i) {
    if (parts[i] < 0) {
      printf("invalid part: %d %d\n", i, parts[i]);
      exit(0);
    }
  }

  int num_verts = g->num_verts;
  bool** comms = new bool*[num_parts];
  int* part_sizes = new int[num_parts];
  int* num_comms_out = new int[num_parts];
  int* edge_cuts = new int[num_parts];
  int* edge_cuts_sizes = new int[num_parts];

  for (int i = 0; i < num_parts; ++i) {
    part_sizes[i] = 0;
    num_comms_out[i] = 0;
    edge_cuts[i] = 0;
    edge_cuts_sizes[i] = 0;

    comms[i] = new bool[num_verts];
    for (int j = 0; j < num_verts; ++j) {
      comms[i][j] = false;
    }
  }

  for (int v = 0; v < num_verts; ++v) {
    part_sizes[parts[v]] += g->vert_weights[v];

    int part = parts[v];
    int degree = out_degree(g, v);
    int* outs = out_vertices(g, v);
    int* weights = out_weights(g, v);
    for (int j = 0; j < degree; ++j) {
      int out = outs[j];

      int out_part = parts[out];
      if (out_part != part) {
        comms[part][out] = true;
        edge_cuts[part]++;
        edge_cuts_sizes[part] += weights[j];
      }
    }
  }

  for (int i = 0; i < num_parts; ++i) {
    for (int j = 0; j < num_verts; ++j) {
      if (comms[i][j]) {
        ++num_comms_out[i];
      }
    }
  }

  int comm_vol = 0;
  int edge_cut = 0;
  int edge_cut_size = 0;
  int max_vert_size = 0;
  for (int i = 0; i < num_parts; ++i) {
    comm_vol += num_comms_out[i];
    edge_cut += edge_cuts[i];
    edge_cut_size += edge_cuts_sizes[i];
    if (part_sizes[i] > max_vert_size)
      max_vert_size = part_sizes[i];
  }

  long avg_size_vert = g->vert_weights_sum / (long)num_parts;
  double max_overweight_v = (double)max_vert_size/(double)avg_size_vert;
  edge_cut /= 2;
  edge_cut_size /= 2;

  printf("%s, Cut: %d, Edges Cut: %d, Comm Vol: %d, Imbalance: %lf\n", 
    step_name, edge_cut_size, edge_cut, comm_vol, max_overweight_v);
  
  for (int i = 0; i < num_parts; ++i) {
    delete [] comms[i];
  }
  delete [] comms;
  delete [] part_sizes;
  delete [] num_comms_out;
  delete [] edge_cuts;
  delete [] edge_cuts_sizes;
}

// void compare_cut(graph* g, graph* g2, 
//   int num_parts, int* parts, int* parts2)
// {
//   int num_verts = g->num_verts;
//   int num_verts2 = g2->num_verts;
//   int* part_sizes = new int[num_parts];
//   int* part_sizes2 = new int[num_parts];
//   int* edge_cuts_sizes = new int[num_parts];
//   int* edge_cuts_sizes2 = new int[num_parts];

//   for (int i = 0; i < num_parts; ++i) {
//     part_sizes[i] = 0;
//     part_sizes2[i] = 0;
//     edge_cuts_sizes[i] = 0;
//     edge_cuts_sizes2[i] = 0;
//   }

//   for (int v = 0; v < num_verts; ++v)
//     part_sizes[parts[v]] += g->vert_weights[v];
//   for (int v = 0; v < num_verts2; ++v)
//     part_sizes2[parts2[v]] += g2->vert_weights[v];
  
//   for (int v = 0; v < num_verts; ++v) {
//     int part = parts[v];
//     printf("Coarse vertex %d in %d\n", v, part);
    
//     int sum_cut = 0;
//     int degree = out_degree(g, v);
//     int* outs = out_vertices(g, v);
//     int* weights = out_weights(g, v);
//     for (int j = 0; j < degree; ++j) {
//       int out = outs[j];

//       int out_part = parts[out];
//       if (out_part != part) {
//         printf("cut to %d (%d %d) with %d\n", 
//           out, g->contracted_verts[out].u, g->contracted_verts[out].v, 
//           weights[j]);
//         edge_cuts_sizes[part] += weights[j];
//         sum_cut += weights[j];
//       } else {
//         printf("no cut to %d (%d %d)\n", 
//           out, g->contracted_verts[out].u, g->contracted_verts[out].v);
//       }        
//     }
    
//     int x = g->contracted_verts[v].u;
//     int y = g->contracted_verts[v].v;
//     int part_x = parts2[x];
//     int part_y = parts2[y];
//     printf("Expands to %d %d in %d %d\n", x, y, part_x, part_y);
    
//     int sum_cut2 = 0;
//     degree = out_degree(g2, x);
//     outs = out_vertices(g2, x);
//     weights = out_weights(g2, x);
//     for (int j = 0; j < degree; ++j) {
//       int out = outs[j];

//       int out_part = parts2[out];
//       if (out_part != part) {
//         printf("%d cut to %d with %d\n", x, out, weights[j]);
//         edge_cuts_sizes[part] += weights[j];
//         sum_cut2 += weights[j];
//       } else {
//         printf("%d no cut to %d\n", x, out);
//       }        
//     }
//     degree = out_degree(g2, y);
//     outs = out_vertices(g2, y);
//     weights = out_weights(g2, y);
//     for (int j = 0; j < degree; ++j) {
//       int out = outs[j];

//       int out_part = parts2[out];
//       if (out_part != part) {
//         printf("%d cut to %d with %d\n", y, out, weights[j]);
//         edge_cuts_sizes[part] += weights[j];
//         sum_cut2 += weights[j];
//       } else {
//         printf("%d no cut to %d\n", y, out);
//       }
//     }
//     printf("sum cuts %d %d\n", sum_cut, sum_cut2);
//   }
// }

