using namespace std;

#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <vector>
#include <queue>
#include <getopt.h>
#include <string.h>
#include <omp.h>


typedef unsigned char uint8;


typedef struct graph {
  int n;
  long m;
  int* out_array;
  long* out_degree_list;
} graph;

#define out_degree(g, n) (g->out_degree_list[n+1] - g->out_degree_list[n])
#define out_vertices(g, n) &g->out_array[g->out_degree_list[n]]

void quicksort_inc(int* arr1, int left, int right) 
{
  int i = left;
  int j = right;
  int temp; 
  int pivot = arr1[(left + right) / 2];

  while (i <= j) 
  {
    while (arr1[i] < pivot) {i++;}
    while (arr1[j] > pivot) {j--;}
  
    if (i <= j) 
    {
      temp = arr1[i];
      arr1[i] = arr1[j];
      arr1[j] = temp;
      ++i;
      --j;
    }
  }

  if (j > left)
    quicksort_inc(arr1, left, j);
  if (i < right)
    quicksort_inc(arr1, i, right);
}


void read_adj(char* filename, int& n, long& m,
  int*& out_array, long*& out_degree_list)
{
  ifstream infile;
  string line;
  string val;
  infile.open(filename);

  getline(infile, line, ' ');
  n = atoi(line.c_str());
  getline(infile, line);
  m = atol(line.c_str())*2;
  printf("n: %d, m: %li\n", n, m);

  out_array = (int*)malloc(m*sizeof(int));
  out_degree_list = (long*)malloc((n+1)*sizeof(long));

#pragma omp parallel for
  for (long i = 0; i < m; ++i)
    out_array[i] = 0;

#pragma omp parallel for
  for (int i = 0; i < n+1; ++i)
    out_degree_list[i] = 0;

  long count = 0;
  int cur_vert = 0;

  while (getline(infile, line))
  {
    out_degree_list[cur_vert] = count;
    stringstream ss(line);
    while (getline(ss, val, ' '))
    {
      out_array[count++] = atoi(val.c_str())-1;
    }
    ++cur_vert;
  }
  out_degree_list[cur_vert] = count;

  infile.close();
}

void write_adj_vw(char* outfilename, graph* g)
{  
  ofstream outfile;
  string line;  
  outfile.open(outfilename);

  int* real_out_degrees = new int[g->n];
  long real_m = 0;

#pragma omp parallel for reduction(+:real_m)
  for (int i = 0; i < g->n; ++i) {
    int out_degree = out_degree(g, i);
    int* outs = out_vertices(g, i);

    quicksort_inc(outs, 0, out_degree);
    int prev_out = -1;
    int cur_index = 0;
    for (int j = 0; j < out_degree; ++j) {
      int out = outs[j];
      if (out != prev_out) {
        prev_out = out;
        outs[cur_index++] = out;
      }
    }
    if (cur_index == 0) printf("EOROR\n");
    real_out_degrees[i] = cur_index;
    real_m += cur_index;
  }

  outfile << g->n << " " << real_m << " 010 2" << endl;

  for (int i = 0; i < g->n; ++i) {
    int out_degree = out_degree(g, i);
    int* outs = out_vertices(g, i);

    outfile << "1 " << out_degree;
    for (int j = 0; j < out_degree; ++j)
      outfile << " " << outs[j]+1;
    outfile << endl;
  }

  outfile.close();
}


int main(int argc, char** argv)
{
  if (argc < 3)
  {
    printf("\nUsage: %s [infile] [outfile]\n", argv[0]);
    exit(0);
  }
  srand(time(0));

  int n;
  long m;

  double elt = omp_get_wtime();
  printf("reading in graph\n");

  int* out_array;
  long* out_degree_list;
  read_adj(argv[1], n, m, out_array, out_degree_list);
  struct graph g = {n, m, out_array, out_degree_list};

  elt = omp_get_wtime() - elt;
  printf("done: %9.6lf\n", elt);
  
  printf("writing\n");
  elt = omp_get_wtime();
  write_adj_vw(argv[2], &g);
  elt = omp_get_wtime() - elt;
  printf("done: %9.2lf\n", elt);

  free(out_array);
  free(out_degree_list);

  return 0;
}

