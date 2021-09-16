
#ifndef _PULP_H_
#define _PULP_H_

#define VERBOSE 1
#define DEBUG 0
#define OUTPUT_STEP 1
#define OUTPUT_TIME 1
#define WRITE_OUTPUT 1
#define DO_EVAL 0
#define QUEUE_MULTIPLIER 2
#define MAX_COARSENING_LEVELS 20
#define COARSE_VERTS_CUTOFF 50000
#define IMPROVEMENT_RATIO 0.99
#define MAX_ITER 10

#include "graph.h"

int pulp_run_coarsen(graph* g, int num_parts, int*& parts, double imb);

void evaluate_quality(graph* g, int num_parts, int* parts);

void evaluate_quality_step(char* step_name, 
  graph* g, int num_parts, int* parts);

void compare_cut(graph* g, graph* g2, 
  int num_parts, int* parts, int* parts2);

#endif
