#ifndef _PULP_V_
#define _PULP_V_

#include "graph.h"

int part_balance(graph* g, int num_parts, int* parts, double imb_limit);

int part_refine(graph* g, int num_parts, int* parts, double imb_limit);

#endif
