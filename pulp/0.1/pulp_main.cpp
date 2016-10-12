/*
//@HEADER
// *****************************************************************************
//
// PULP: Multi-Objective Multi-Constraint Partitioning Using Label Propagation
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//
// *****************************************************************************
//@HEADER
*/


using namespace std;

#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <getopt.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#include "pulp.h"


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
  m = atol(line.c_str()) * 2;
  printf("n: %d, m: %li ", n, m/2);

  out_array = new int[m];
  out_degree_list = new long[n+1];

#pragma omp parallel for
  for (int i = 0; i < n+1; ++i)
    out_degree_list[i] = 0;

  long count = 0;
  int cur_vert = 0;

  while (getline(infile, line))
  {
    out_degree_list[cur_vert++] = count;
    stringstream ss(line);
    while (getline(ss, val, ' '))
    {
      out_array[count++] = atoi(val.c_str())-1;
    }
  }
  out_degree_list[cur_vert] = count;
  assert(cur_vert == n);
  assert(count == m);

  infile.close();
}

void read_parts(char* filename, int num_verts, int* parts)
{
  ifstream infile;
  string line;
  infile.open(filename);

  for (int i = 0; i < num_verts; ++i)
  {
    getline(infile, line);
    parts[i] = atoi(line.c_str());
  }

  infile.close();
}

void write_parts(char* filename, int num_verts, int* parts)
{
  ofstream outfile;
  outfile.open(filename);

  for (int i = 0; i < num_verts; ++i)
    outfile << parts[i] << endl;

  outfile.close();
}


void print_usage_full(char** argv)
{
  printf("To run: %s [graphfile] [num parts] [options]\n\n", argv[0]);
  printf("Options:\n");
  printf("\t-v [#.#]:\n");
  printf("\t\tVertex balance constraint [default: 1.10 (10%%)]\n");
  printf("\t-e [#.#]:\n");
  printf("\t\tEdge balance constraint [default: off]\n");
  printf("\t-c:\n");
  printf("\t\tAttempt to minimize per-part cut\n");
  printf("\t-l:\n");
  printf("\t\tDo label propagation-based initialization\n");
  printf("\t-m [#]:\n");
  printf("\t\tGenerate multiple partitions [default: 1]\n");
  printf("\t-o [file]:\n");
  printf("\t\tOutput parts file [default: graphname.part.numparts]\n");
  printf("\t-i [file]:\n");
  printf("\t\tInput parts file [default: none]\n");
  exit(0);
}

/*
'##::::'##::::'###::::'####:'##::: ##:
 ###::'###:::'## ##:::. ##:: ###:: ##:
 ####'####::'##:. ##::: ##:: ####: ##:
 ## ### ##:'##:::. ##:: ##:: ## ## ##:
 ##. #: ##: #########:: ##:: ##. ####:
 ##:.:: ##: ##.... ##:: ##:: ##:. ###:
 ##:::: ##: ##:::: ##:'####: ##::. ##:
..:::::..::..:::::..::....::..::::..::
*/
int main(int argc, char** argv)
{
  setbuf(stdout, NULL);
  srand(time(0));
  if (argc < 3)
  {
    print_usage_full(argv);
    exit(0);
  }  

  int n = 0;
  long m = 0;
  int* out_array;
  long* out_degree_list;
  char* graph_name = strdup(argv[1]);
  char* num_parts_str = strdup(argv[2]);
  int num_parts = atoi(num_parts_str);
  int* parts;
  int num_partitions = 1;

  double vert_balance = 1.10;
  double vert_balance_lower = 0.25;
  double edge_balance = 1.50;
  char parts_out[1024]; parts_out[0] = '\0';
  char parts_in[1024]; parts_in[0] = '\0';

  bool do_bfs_init = true;
  bool do_lp_init = false;
  bool do_vert_balance = true;
  bool do_edge_balance = false;
  bool do_maxcut_balance = false;
  bool eval_quality = false;

  char c;
  while ((c = getopt (argc, argv, "v:e:i:o:clm:q")) != -1)
  {
    switch (c)
    {
      case 'v':
        vert_balance = strtod(optarg, NULL);
        break;
      case 'e':
        edge_balance = strtod(optarg, NULL);
        do_edge_balance = true;
        break;
      case 'i':
        strcat(parts_in, optarg);
        break;
      case 'o':
        strcat(parts_out, optarg);
        break;
      case 'c':
        do_maxcut_balance = true;
        break;
      case 'm':
        num_partitions = atoi(optarg);
        break;
      case 'l':
        do_lp_init = true;
        do_bfs_init = false;
        break;
      case 'q':
        eval_quality = true;
        break;
      case '?':
        if (optopt == 'v' || optopt == 'e' || optopt == 'i' || optopt == 'o' || optopt == 'm')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `\\x%x'.\n",
      optopt);
        print_usage_full(argv);
      default:
        abort();
    }
  }  

  double elt, elt2;
  double avg = 0.0;
  printf("Reading in graph %s ... ", graph_name);
  elt = timer();
  read_adj(graph_name, n, m, out_array, out_degree_list);
  pulp_graph_t g = {n, m, out_array, out_degree_list};
  elt = timer() - elt;
  printf("... Done: %9.6lf\n", elt);

  parts = new int[g.n];
  for (int i = 0; i < num_partitions; ++i)
  {  
    if (strlen(parts_in) != 0)
    {  
      printf("Reading in parts file %s ... ", parts_in);
      elt = timer();
      do_lp_init = false;
      read_parts(parts_in, g.n, parts);
      elt = timer() - elt;
      printf("Done: %9.6lf\n", elt);
    }
    else if (do_bfs_init)
      for (int j = 0; j < g.n; ++j) parts[j] = -1;
    else
      for (int i = 0; i < g.n; ++i) parts[i] = rand() % num_parts;

    pulp_part_control_t ppc = {vert_balance, edge_balance, 
      do_lp_init, do_bfs_init, do_edge_balance, do_maxcut_balance,
      false};
    
    printf("\nBeginning partitioning ... ");
    elt = timer();
    pulp_run(&g, &ppc, parts, num_parts);
    elt = timer() - elt;
    avg *= elt;
    printf("Partitioning Time: %9.6lf\n\n", elt);

#if OUTPUT_TIME
    if (!do_edge_balance && !do_maxcut_balance)
      printf("\n$$$ PULP, %s, %d, %d, %6.6lf\n", graph_name, num_parts, omp_get_max_threads(), elt);
    else if (do_edge_balance && !do_maxcut_balance)
      printf("\n$$$ PULP-M, %s, %d, %d, %6.6lf\n", graph_name, num_parts, omp_get_max_threads(), elt);
    else if (do_edge_balance && do_maxcut_balance)
      printf("\n$$$ PULP-MM, %s, %d, %d, %6.6lf\n", graph_name, num_parts, omp_get_max_threads(), elt);
#endif

#if WRITE_OUTPUT
    char temp_out[1024]; temp_out[0] = '\0';
    strcat(temp_out, parts_out);
    if (strlen(temp_out) == 0)
    {
      strcat(temp_out, graph_name);
      strcat(temp_out, ".parts.");
      strcat(temp_out, num_parts_str);
    }
    if (num_partitions > 1)
    {
      strcat(temp_out, ".");
      stringstream ss; 
      ss << i;
      strcat(temp_out, ss.str().c_str());
    }
    printf("writing parts file %s ... ", temp_out);
    elt = timer();
    write_parts(temp_out, g.n, parts);
    elt = timer() - elt;
    printf("Done: %9.6lf\n", elt);
#endif
    if (eval_quality)
      evaluate_quality(g, num_parts, parts);
  }

  delete [] parts;
  delete [] out_array;
  delete [] out_degree_list;

  return 0;
}
