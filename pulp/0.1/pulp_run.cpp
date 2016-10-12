/*
//@HEADER
// *****************************************************************************
//
// PuLP: Multi-Objective Multi-Constraint Partitioning Using Label Propagation
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

#include "init_nonrandom.cpp"
#include "label_prop.cpp"
#include "label_balance_verts.cpp"
#include "label_balance_edges.cpp"
#include "label_balance_edges_maxcut.cpp"

int* pulp_run(pulp_graph_t* g, pulp_part_control_t* pd, 
          pulp_part_t* parts, pulp_part_t* num_parts)
{
  double vert_balance = pd->vert_balance;
  double vert_balance_lower = 0.25;
  double edge_balance = pd->edge_balance;
  double do_label_prop - pd->do_label_prop;
  double do_nonrandom_init = pd->do_nonrandom_init;
  double verbose = pd->verbose_output;
  bool do_vert_balance = true;
  bool do_edge_balance = pd->do_edge_balance;
  bool do_maxcut_balance = pd->do_maxcut_balance;
  int balance_outer_iter = 1;
  int label_prop_iter = 3;
  int vert_outer_iter = 3;
  int vert_balance_iter = 5;
  int vert_refine_iter = 10;
  int edge_outer_iter = 3;
  int edge_balance_iter = 5;
  int edge_refine_iter = 10;

  double elt, elt2, elt3;
  elt = timer();

  if (do_label_prop)
  {
    if (verbose) printf("\tDoing label prop stage with %d parts\n", num_parts);
    elt2 = timer();

    label_prop(g, num_parts, parts,
      label_prop_iter, vert_balance_lower);

    elt2 = timer() - elt2;
    if (verbose) printf("done: %9.6lf(s)\n", elt2);
  }
  else if (do_nonrandom_init)
  {
    if (verbose) printf("\tDoing bfs init stage with %d parts\n", num_parts);
    elt2 = timer();
    init_nonrandom(g, num_parts, parts);
    elt2 = timer() - elt2;
    if (verbose) printf("done: %9.6lf(s)\n", elt2);
  }
  if (verbose) printf("\tBeginning vertex (and edge) refinement\n");
  for (int boi = 0; boi < balance_outer_iter; ++boi)
  {
    elt2 = timer();
    if (do_vert_balance)
    {
      if (verbose) printf("\t\tDoing vert balance and refinement stage\n");
      elt3 = timer(); 
      label_balance_verts(g, num_parts, parts,
        vert_outer_iter, vert_balance_iter, vert_refine_iter,
        vert_balance);
      elt3 = timer() - elt3;
      if (verbose) printf("done: %9.6lf(s)\n", elt);
    }

    if (do_edge_balance && !do_maxcut_balance)
    {
      if (verbose) printf("\t\tDoing edge balance and refinement stage\n");
      elt3 = timer();
      label_balance_edges(g, num_parts, parts,
        edge_outer_iter, edge_balance_iter, edge_refine_iter,
        vert_balance, edge_balance);
      elt3 = timer() - elt3;
      if (verbose) printf("done: %9.6lf(s)\n", elt3);
    }
    else if (do_edge_balance && do_maxcut_balance)
    {
      if (verbose) printf("\t\tDoing maxcut balance and refinement stage\n");
      elt3 = timer();
      label_balance_edges_maxcut(g, num_parts, parts,
        edge_outer_iter, edge_balance_iter, edge_refine_iter,
        vert_balance, edge_balance);
      elt3 = timer() - elt3;
      if (verbose) printf("done: %9.6lfs\n", elt3);
    }

    elt2 = timer() - elt2;
    if (verbose) printf("\tFinished outer loop iter %d: %9.6lf(s)\n", (boi+1), elt2);
  }

  elt = timer() - elt;
  if (verbose) printf("Partitioning finished: %9.6lf(s)\n", elt);

  return parts;
}

