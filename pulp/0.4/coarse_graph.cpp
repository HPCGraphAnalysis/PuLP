#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

void print_array( int64_t* array, int64_t nodes )
{ 
  for( int i = 0; i < nodes; i++ )
  {
    printf("%ld ",*(array + i) );
  }
  printf("\n");
}

/*
Takes in an array of int64_t and sorts it in ascending order.
*/
void sort( int64_t* array , int64_t length )
{
  for( int64_t i = 0; i < length; i++ )
  {
    for( int64_t j = i+1; j < length; j++ )
    {
      if( *(array + i) > *(array + j) )
      {
        int64_t temp = *(array + i);
        *(array + i) = *(array + j);
        *(array + j) = temp;
      }
    }
  }
}

/*
Takes in one of the two connections of a node and tells us what level that 
connection points to.
Runtime: O(1)
*/
int64_t interp_level( int64_t value, int64_t nodes )
{
  int64_t level = 0;
  level = value / nodes;
  return(level);
}

/*
Takes in one of the two connections of a node and tells us what node that 
connection points to.
Runtime: O(1)
*/
int64_t interp_node( int64_t value, int64_t nodes )
{
  int64_t node = 0;
  node = value % nodes;
  return(node);
} 

/*
Takes in a node and the adjacency matrix and prints out the nodes and 
and levels of coarsening.
*/
void print_node_state( int64_t** adj, int64_t node_a, int64_t nodes)
{
  printf("NODE STATE for node: %ld\n",node_a);
  printf(" --- values: from: %ld , to: %ld \n", *(*(adj + node_a)), *(*(adj + node_a) + 1) );
  int64_t from_node, from_level, to_node, to_level;
  from_node = interp_node(*(*(adj + node_a)), nodes);
  from_level = interp_level(*(*(adj + node_a)), nodes);
  to_node = interp_node(*(*(adj + node_a) + 1), nodes);
  to_level = interp_level(*(*(adj + node_a) + 1), nodes);
  printf(" --- from level: %ld --- from node: %ld \n",from_level, from_node);
  printf(" --- to level: %ld --- to node: %ld \n",to_level, to_node);
  printf("\n");
}

/*
Takes in the stored adjacency list, a node, the current level, 
and the number of nodes.
Returns the node that this edge connects to in the 'k'-th level.
Runtime: O(k) where k is the number of coarsenings.
*/
int64_t get_k_root( int64_t **adj, int64_t node_b, int64_t k_level, int64_t nodes )
{
  // set level to be low as possible.
  int64_t level = 0;
  int64_t node = node_b;
  int flag = 0;

  // If our node has not been coarsened return the node.
  while( flag == 0 )
  {
    level = interp_level( *(*(adj + node)), nodes );
    if( level > k_level || level == 0 )
    {
      flag = 1;
    }
    else
    {
      node = interp_node( *(*(adj + node)), nodes );
    }
  }

  return(node);
}

/*
Takes in the stored adjacency list, a node, and the number of nodes.
Returns the node that this edge connects to in the 'k'-th level.
Runtime: O(k) where k is the number of coarsenings.
*/
int64_t get_leaf( int64_t **adj, int64_t node_b, int64_t nodes )
{
  int64_t level = nodes + 1;
  int64_t node = node_b;

  if( *(*(adj + node) + 1) == 0 ) { return(node); }
  else
  { 
    while ( level > 0 ) 
    {
      node = interp_node( *(*(adj + node) + 1), nodes );
      level = interp_level( *(*(adj + node) + 1), nodes );
    }
  }

  return(node);
}

/*
Takes in the adjacency list and fills an array of nodes corresponding to the kth coarsening level.
Appends a '/0' to the end of the pointed to array.
Returns 0 on success.
Runtime: O(n) where n is the number of nodes.
*/
int64_t get_k_node_num( int64_t** adj, int64_t k_level, int64_t nodes )
{
  int64_t node_num = 0;
  for( int64_t i = 0; i < nodes; i++ )
  {
    int64_t level = interp_level(*(*(adj + i)), nodes);
    if( level > k_level || level == 0 )
    {
      node_num = node_num + 1;
    }
  }
  return(node_num);
}

/*
Takes in the adjacency list and fills an array of nodes corresponding to the kth coarsening level.
Appends a '/0' to the end of the pointed to array.
Returns 0 on success.
Runtime: O(n) where n is the number of nodes.
Note: This method induces a mapping for a k-level adjacency. That is the node i in the kth level
     coarsening is the same as node_arr[i]. This is used for graph reconstruction. 
*/
int get_k_nodes( int64_t** adj, int64_t* node_arr, int64_t k_level, int64_t nodes )
{
  int64_t index = 0;
  for( int64_t node_a = 0; node_a < nodes; node_a++ )
  {
    int64_t level = interp_level(*(*(adj + node_a)),nodes);

    if( level > k_level || level == 0 )
    {
      *(node_arr + index) = node_a;
      index += 1;
    }
  }
  return(0);
}

/*
Takes in an edge and performs a merge.
Returns 0 on success and -1 on error.
-- Allows for merging of any node to any other node. I.E. it checks if node 
   b has been subsumed by another node and then searches to connect that node to
   node a.
-- Here, node_b will be subsumed by node_a.
Runtime: O(k) where k is the number of coarsenings (this comes from get_node)
*/
int merge( int64_t** adj, int64_t node_a, int64_t node_b, int64_t k_level, int64_t nodes )
{
  int64_t node_b_root = node_b;
  if( **(adj + node_b) != 0 ) { node_b_root = get_k_root( adj, node_b, k_level, nodes ); }
  
  int64_t node_a_leaf = get_leaf( adj, node_a, nodes );
  int64_t node_a_root = get_k_root( adj, node_a, k_level, nodes );

  *(*(adj + node_a_leaf) + 1) = ( nodes*k_level ) + node_b_root;
  **(adj + node_b_root) = ( nodes*k_level ) + node_a_root;
  return(0);
}

/*
Takes in a list of edges and performs several merges.
Returns 0 on success.
Runtime: O(|e|k) where k is the level of coarsening, and |e| is the number of edges in our list.
*/
int merge_many( int64_t** adj, int64_t** edges, int64_t k_level, int64_t nodes, int64_t edge_num )
{
  int ret;
  int64_t node_a, node_b;

  for( int64_t i = 0; i < edge_num; i++ )
  {
    node_a = get_k_root( adj, *(*(edges + i)), k_level, nodes );
    node_b = *(*(edges + i) + 1);
    ret = merge( adj, node_a, node_b, k_level, nodes );
    if( ret == -1 )
    {
      printf("ERROR: in >> merge_many << \n");
      printf("-- failed to merge nodes %ld and %ld \n", node_a, node_b);
      return(-1);
    }
  }
  return(0);
}

/*
Takes in our adajacency list, a node, the number of nodes in the graph, and the kth level 
we wish to recover information for.
Determines how many adjacencies would be needed for an adjacency list of a node in the kth
level of coarsening.
Returns the number of adjacencies.
Notes: 
*/
int64_t get_adj_length( int64_t** adj, int64_t node_a, int64_t nodes, int64_t k_level )
{
  int64_t value = 0;
  int64_t flag = 0;
  int64_t curr_node = node_a;
  while( flag < 1 )
  {
    value = value + *(*(adj + curr_node) + 2);
    int64_t level = interp_level(*(*(adj + curr_node) + 1),nodes);
    if( level > k_level || level == 0 )
    {
      flag = 1;
    }
    else
    {
      curr_node = interp_node(*(*(adj + curr_node) + 1),nodes);
    }
  }
  return( value );
}

/* 
Takes in our graph adjacency, an array of nodes in the kth level "k_nodes", an array map with length "nodes", 
the number of nodes, and a level of coarsening k.
Fills the "map" array with mapping elements.
Returns: 0 on success
Runtime: O(kn)
*/
int get_k_mapping( int64_t** adj, int64_t* map, int64_t nodes, int64_t k_level )
{ 
  // get the roots of the nodes at the kth level
  for( int64_t i = 0; i < nodes; i++ )
  {
    *(map + i) = get_k_root( adj, i, k_level, nodes); 
  }

  // copy the array to a new array
  int64_t* temp_arr = ( int64_t* )malloc( nodes * sizeof( int64_t ) );
  for( int64_t i = 0; i < nodes; i++ )
  {
    *(temp_arr + i) = *(map + i);
  }

  // sort the copy and count the unique elements
  sort( temp_arr, nodes );
  int64_t curr_value = 0;
  int64_t length = 0;
  for( int64_t i = 0; i < nodes; i++ )
  { 
    if( *(temp_arr + i) > curr_value )
    {
      length = length + 1;
      curr_value = *(temp_arr + i);
    } 
  }

  // check the mapping for elements larger than the number of unique elements
  // subtract "(nodes-length)" from each of them
  for( int64_t i = 0; i < nodes; i++ )
  {
    if( *(map + i) > (length - 1) )
    {
      *(map + i) = *(map + i) - (nodes - length - 1);
    }
  }

  free( temp_arr );

  return(0);
}

/*
Takes in an adjacency list, a node, the number of nodes, and a level "k". 
Determines the neighbors of that node and populates an input array "neighbors"
-- Note this list may have duplicates. This is currently because it is the easiest way to preserve and recover connections.
-- Can never actually implement removing edges this way because it may require far more "pointers" than we have.
Returns 0 on success.
Runtime: O(dk) where d is the degree of the node and k is the level of coarsening. Note that obtaining all neighbors
         in the original graph will be O(nmk) where m is the number of nodes, m is the number of edges. This is not greeeat.
Notes: In general we will want a neighbors array that has at least one more element than required.
*/
int get_k_neighbors( int64_t** adj, int64_t node_a, int64_t k_level, int64_t nodes, int64_t* neighbors )
{
  int64_t value = 0;
  int64_t n_index = 0;
  int64_t level = 0;
  int64_t curr_node = node_a;
  while( value < 1 )
  {
    int64_t adj_nodes = * ( *(adj + curr_node) + 2 );
    for( int64_t i = 0; i < adj_nodes; i++ ) 
    {
      int64_t j = i + 3;
      *( neighbors + n_index ) = get_k_root( adj, *( *(adj + curr_node) + j ), k_level, nodes);
      level = interp_level( *(*(adj + curr_node ) + 1), nodes );
      n_index += 1; 

      if( *(*(adj + curr_node) + 1) == 0 || level > k_level ) 
        { 
          value = 1; 
        }
    }
    curr_node = interp_node( *(*( adj + curr_node ) + 1), nodes );
  }
  return(0);
}

/*
Takes in a graph adjacency, a level of coarsening, and a blank 2D array for a new adjacency.
Constructs a new adjacency based on the inputs.
Returns 0 on success.
Notes: This returns a loopy-multigraph version without weights.
*/
int get_k_graph( int64_t** adj, int64_t** new_adj, int64_t k_level, int64_t nodes )
{
  // get k-level nodes.
  int64_t node_num = get_k_node_num( adj, k_level, nodes );
  int64_t* k_nodes = ( int64_t* )malloc( node_num * sizeof(int64_t) );
  get_k_nodes( adj, k_nodes, k_level, nodes );

  // get map.
  int64_t* map = ( int64_t* )malloc( nodes * sizeof(int64_t) );
  get_k_mapping( adj, map, nodes, k_level );
  int64_t adj_length, node_a;

  // loop through our new nodes.
  for( int64_t i = 0; i < node_num; i++ )
  {
    node_a = *(k_nodes + i);
    adj_length = get_adj_length( adj, node_a, nodes, k_level );

    // reallocate memory length.
    int64_t* ret = realloc( *(new_adj + i), (adj_length + 3) * sizeof(int64_t) );
    *(new_adj + i) = ret; // for some reason the realloc doesnt work here and I have to do this?

    // populate the neighbor array for our current node.
    int64_t* neighbors = ( int64_t* )malloc( adj_length * sizeof(int64_t) );
    get_k_neighbors( adj, node_a, k_level, nodes, neighbors );

    for( int64_t j = 0; j < adj_length; j++ )
    {
      *(*(new_adj + i) + (j + 3)) = *(map + *(neighbors + j));
    }
    // Initialize degree and coarsening pointers (coarsening pointers are init to 0 here)
    *(*(new_adj + i) + 2) = adj_length;
    *(*(new_adj + i) + 1) = 0;
    *(*(new_adj + i) + 0) = 0;

    free( neighbors );
  } 

  free( k_nodes );
  return(0);
}

/*
Takes in a graph adjacency, a weight adjacency, the k_level of the graph, 
and the total number of nodes. 
Returns 0 on success.
Constructs a weighted graph adjacency
Notes: This returns a loopy-multigraph version without weights.
*/
