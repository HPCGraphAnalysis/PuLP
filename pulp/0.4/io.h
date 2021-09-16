#ifndef _IO_H_
#define _IO_H_

void write_parts(char* filename, int num_verts, int* parts);

void read_bin(char* filename,
 int& num_verts, int& num_edges,
 int*& srcs, int*& dsts);

void read_edge(char* filename,
  int& num_verts, int& num_edges,
  int*& srcs, int*& dsts);


#endif