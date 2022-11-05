/*
//@HEADER
// *****************************************************************************
//
//  XtraPuLP: Xtreme-Scale Graph Partitioning using Label Propagation
//              Copyright (2016) Sandia Corporation
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
// Questions?  Contact  George M. Slota   (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//                      Kamesh Madduri    (madduri@cse.psu.edu)
//
// *****************************************************************************
//@HEADER
*/

#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>
#include <string.h>
#include <sstream>

#include "xtrapulp.h"
#include "dist_graph.h"
#include "generate.h"
#include "comms.h"
#include "io_pp.h"
#include "pulp_util.h"
#include "util.h"

#define MAX_CONSTRAINTS 1024


extern int procid, nprocs;
extern int seed;
extern bool verbose, debug, verify;

void print_usage(char** argv)
{
  printf("To run: %s [graphfile] [num parts] [options]\n", argv[0]);
  printf("\t Use -h for list of options\n\n");
  exit(0);
}

void print_usage_full(char** argv)
{
  printf("To run: %s [graphfile] [num parts] [options]\n\n", argv[0]);
  printf("Options:\n");
  printf("\t-a\n");
  printf("\t\tGraph file is adjacency (METIS) format\n");
  printf("\t\tDefault is unsigned 32-bit binary edge list\n");
  printf("\t-t \"#.# #.# ...\"\n");
  printf("\t\tFor PuLP-W: Space delimited list of constraints\n");
  printf("\t-v [#.#]:\n");
  printf("\t\tFor PuLP-MM: Vertex balance constraint [default: 1.10 (10%%)]\n");
  printf("\t-e [#.#]:\n");
  printf("\t\tFor PuLP-MM: Edge balance constraint [default: off]\n");
  printf("\t-c:\n");
  printf("\t\tMinimize per-part cut in addition to edge cut\n");
  printf("\t-d:\n");
  printf("\t\tUse round-robin instead of vertex-block distribution\n");
  printf("\t\t\t (Might help with load imbalance, might hurt quality)\n");
  printf("\t-l:\n");
  printf("\t\tDo label propagation-based initialization\n");
  printf("\t-m [#]:\n");
  printf("\t\tGenerate multiple partitions [default: 1]\n");
  printf("\t-o [file]:\n");
  printf("\t\tOutput parts file [default: graphname.part.numparts]\n");
  printf("\t-i [file]:\n");
  printf("\t\tInput parts file [default: none]\n");
  printf("\t-s [seed]:\n");
  printf("\t\tSet seed integer [default: random int]\n");
  exit(0);
}

int parse_constraints(char* optarg, 
    double* constraints, int32_t& num_constraints)
{
  num_constraints = 0;
  double d = 0.0;
  std::stringstream ss(optarg);
  while (ss >> d) {
    constraints[num_constraints++] = d;
    if (num_constraints >= MAX_CONSTRAINTS)
      throw_err("Maximum constraints is 1024. Check input formatting.");
  }

  return 0;
}


int main(int argc, char **argv) 
{
  srand(time(0));
  setbuf(stdout, 0);

  verbose = false;
  debug = false;
  verify = false;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (argc < 3) 
  {
    if (procid == 0)
      print_usage_full(argv);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  char input_filename[1024]; input_filename[0] = '\0';
  char graphname[1024]; graphname[0] = '\0';  
  char* graph_name = strdup(argv[1]);
  char* num_parts_str = strdup(argv[2]);
  char parts_out[1024]; parts_out[0] = '\0';
  char parts_in[1024]; parts_in[0] = '\0';
  
  strcat(input_filename, argv[1]);
  int32_t num_parts = atoi(argv[2]);
  double vert_balance = 1.1;
  double edge_balance = 1.1;
  double constraints[MAX_CONSTRAINTS];
  int32_t num_constraints = 0;
  bool adj_format = false;

  uint64_t num_runs = 1;
  bool output_time = true;
  bool output_quality = false;

  bool gen_rmat = false;
  bool gen_rand = false;
  bool gen_hd = false;
  uint64_t gen_n = 0;
  uint64_t gen_m_per_n = 16;
  bool offset_vids = false;
  int pulp_seed = rand();

  bool do_bfs_init = true;
  bool do_lp_init = false;
  bool do_repart = false;
  bool do_edge_balance = false;
  bool do_maxcut_balance = false;

  char c;
  while ((c = getopt (argc, argv, "v:e:o:i:mn:s:p:dlqtc:a")) != -1) {
    switch (c) {
      case 'h':
        print_usage_full(argv);
        break;
      case 'a':
        adj_format = true;
        break;
      case 'c':
        parse_constraints(optarg, constraints, num_constraints);
        break;
      case 'v':
        vert_balance = strtod(optarg, NULL);
        break;
      case 'e':
        edge_balance = strtod(optarg, NULL);
        do_edge_balance = true;
        break;
      case 'm':
        do_maxcut_balance = true;
        break;
      case 'o':
        strcat(parts_out, optarg);
        break;
      case 'i':
        strcat(parts_in, optarg);
        do_repart = true;
        do_bfs_init = false;
        do_lp_init = false;
        break;
      case 'n':
        num_runs = strtoul(optarg, NULL, 10);
        break;
        break;
      case 's':
        pulp_seed = atoi(optarg);
        break;
      // Below is for testing only
      /*case 'r':
        gen_rmat = true;
        gen_n = strtoul(optarg, NULL, 10);
        break;
      case 'g':
        gen_rand = true;
        gen_n = strtoul(optarg, NULL, 10);
        break;
      case 's':
        gen_hd = true;
        gen_n = strtoul(optarg, NULL, 10);
        break;*/
      case 'p':
        gen_m_per_n = strtoul(optarg, NULL, 10);
        break;
      case 'd':
        offset_vids = true;
        break;
      case 'l':
        do_lp_init = true;
        do_bfs_init = false;
        break;
      case 'q':
        output_quality = true;
        break;
      case 't':
        output_time = true;
        break;
      default:
        throw_err("Input argument format error");
    }
  }

  graph_gen_data_t* ggi = (graph_gen_data_t*)malloc(sizeof(graph_gen_data_t));
  if (gen_rand)
  {
    std::stringstream ss;
    ss << "rand-" << gen_n << "-" << gen_m_per_n;
    strcat(graphname, ss.str().c_str());
    generate_rand_out_edges(ggi, gen_n, gen_m_per_n, offset_vids);
  }
  else if (gen_rmat)
  {
    std::stringstream ss;
    ss << "rmat-" << gen_n << "-" << gen_m_per_n;
    strcat(graphname, ss.str().c_str());
    generate_rmat_out_edges(ggi, gen_n, gen_m_per_n, offset_vids);
  }
  else if (gen_hd)
  {
    std::stringstream ss;
    ss << "hd-" << gen_n << "-" << gen_m_per_n;
    strcat(graphname, ss.str().c_str());
    generate_hd_out_edges(ggi, gen_n, gen_m_per_n, offset_vids);
  }
  else
  {
    double elt = omp_get_wtime();
    if (procid == 0) printf("Reading in graphfile %s\n", input_filename);
    strcat(graphname, input_filename);

    if (adj_format)
      read_graph(input_filename, ggi, offset_vids);
    else
      load_graph_edges_32(input_filename, ggi, offset_vids);

    elt = omp_get_wtime() - elt;
    if (procid == 0) printf("Reading Finished: %9.6lf (s)\n", elt);
  }


  dist_graph_t* g = (dist_graph_t*)malloc(sizeof(dist_graph_t));
  mpi_data_t* comm = (mpi_data_t*)malloc(sizeof(mpi_data_t));
  pulp_data_t* pulp = (pulp_data_t*)malloc(sizeof(pulp_data_t));
  queue_data_t* q = (queue_data_t*)malloc(sizeof(queue_data_t));
  init_comm_data(comm);  
  if (nprocs > 1)
  {
    if (ggi->num_vert_weights > 0) {
      exchange_edges_weighted(ggi, comm);
      create_graph_weighted(ggi, g);
    } else {
      exchange_edges(ggi, comm);
      create_graph(ggi, g);
    }
    //set_weights_graph(g);
    relabel_edges(g);
  }
  else
  {
    if (ggi->num_vert_weights > 0) {
      create_graph_serial_weighted(ggi, g);
    } else {
      create_graph_serial(ggi, g);
    }
    //set_weights_graph(g);
  }
  if (g->num_vert_weights > 0) {
    init_pulp_data_weighted(g, pulp, num_parts);
  } else {
    init_pulp_data(g, pulp, num_parts);
  }
  init_queue_data(g, q);
  get_ghost_degrees(g, comm, q);

  pulp_part_control_t* ppc = 
      (pulp_part_control_t*)malloc(sizeof(pulp_part_control_t));
  *ppc = {
      vert_balance, edge_balance, 
      constraints, (int)g->num_vert_weights, 
      do_lp_init, do_bfs_init, do_repart, 
      do_edge_balance, do_maxcut_balance,
      false, pulp_seed};

  double total_elt = 0.0;
  for (uint32_t i = 0; i < num_runs; ++i)
  {
    double elt = 0.0;
    if (parts_in[0] != '\0')
    {
      if (procid == 0) printf("Reading in parts file %s\n", parts_in);
      elt = omp_get_wtime();
      read_parts(parts_in, g, pulp, offset_vids);
      elt = omp_get_wtime() - elt;
      if (procid == 0) printf("Reading Finished: %9.6lf (s)\n", elt);
    }

    if (procid == 0) printf("Starting Partitioning\n");
    elt = omp_get_wtime();

    if (g->num_vert_weights > 0)
      xtrapulp_weighted(g, ppc, comm, pulp, q);
    else 
      xtrapulp(g, ppc, comm, pulp, q);

    total_elt += omp_get_wtime() - elt;
    elt = omp_get_wtime() - elt;
    if (procid == 0) printf("Partitioning Finished\n");
    if (procid == 0) printf("XtraPuLP Time: %9.6lf (s)\n", elt);

    if (output_quality)
    {
      if (g->num_vert_weights == 0)
        part_eval(g, pulp);
      else
        part_eval_weighted(g, pulp);
      // For testing
      //if (procid == 0)
      //  printf("&&& XtraPuLP, %s, %d, %2.3lf, %2.3lf, %li, %li\n", 
      //   graphname, num_parts, pulp.max_v, pulp.max_e, 
      //    pulp.cut_size, pulp.max_cut);
    }

    char temp_out[1024]; temp_out[0] = '\0';
    strcat(temp_out, parts_out);
    if (strlen(temp_out) == 0)
    {
      strcat(temp_out, graph_name);
      strcat(temp_out, ".parts.");
      strcat(temp_out, num_parts_str);
    }
    if (num_runs > 1)
    {
      std::stringstream ss; 
      ss << "." << i;
      strcat(temp_out, ss.str().c_str());
    }
    if (procid == 0) printf("Writing out parts file %s\n", temp_out);
    elt = omp_get_wtime();
    output_parts(temp_out, g, pulp->local_parts, offset_vids);
    if (procid == 0) printf("Done Writing: %9.6lf (s)\n", omp_get_wtime() - elt);
  }
  if (output_time && procid == 0 && num_runs > 1) 
  {
    printf("XtraPuLP Avg. Time: %9.6lf (s)\n", (total_elt / (double)num_runs) );
  }

  clear_graph(g);
  free(g);
  clear_comm_data(comm);
  free(comm);
  clear_queue_data(q);
  free(q);
  clear_pulp_data(pulp);
  free(pulp);
  free(ppc);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}

