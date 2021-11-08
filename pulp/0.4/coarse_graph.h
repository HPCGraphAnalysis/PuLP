#ifndef _COARSE_GRAPH_H_
#define _COARSE_GRAPH_H_

#include <stdint.h>

void print_array( int64_t* array, int64_t nodes );

void sort( int64_t* array , int64_t length );

int64_t interp_level( int64_t value, int64_t nodes );

int64_t interp_node( int64_t value, int64_t nodes );

void print_node_state( int64_t** adj, int64_t node_a, int64_t nodes);

int64_t get_k_root( int64_t **adj, int64_t node_b, int64_t k_level, int64_t nodes );

int64_t get_leaf( int64_t **adj, int64_t node_b, int64_t nodes );

int64_t get_k_node_num( int64_t** adj, int64_t k_level, int64_t nodes );

int get_k_nodes( int64_t** adj, int64_t* node_arr, int64_t k_level, int64_t nodes );

int merge( int64_t** adj, int64_t node_a, int64_t node_b, int64_t k_level, int64_t nodes );

int merge_many( int64_t** adj, int64_t** edges, int64_t k_level, int64_t nodes, int64_t edge_num );

int64_t get_adj_length( int64_t** adj, int64_t node_a, int64_t nodes, int64_t k_level );

int get_k_mapping( int64_t** adj, int64_t* map, int64_t nodes, int64_t k_level );

int get_k_neighbors( int64_t** adj, int64_t node_a, int64_t k_level, int64_t nodes, int64_t* neighbors );

int get_k_graph( int64_t** adj, int64_t** new_adj, int64_t k_level, int64_t nodes );

#endif
