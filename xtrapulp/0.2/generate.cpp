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
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

#include "generate.h"
#include "dist_graph.h"
#include "util.h"

extern int procid, nprocs;
extern int seed;
extern bool verbose, debug, verify;

int generate_rand_out_edges(graph_gen_data_t* ggi, 
  uint64_t num_verts, uint64_t edges_per_vert, bool offset_vids)
{
  if (debug) { printf("Task %d generate_rand_out_edges() start\n", procid); }
  
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  ggi->n = num_verts;
  ggi->m = num_verts * edges_per_vert;
  ggi->n_offset = (uint64_t)procid * (num_verts / (uint64_t)nprocs + 1);
  ggi->n_local = num_verts / (uint64_t)nprocs + 1;
  if (procid == nprocs - 1 && !offset_vids)
    ggi->n_local = ggi->n - ggi->n_offset;
  ggi->m_local_read = ggi->n_local * edges_per_vert;

  if (debug) { 
    printf("Task %d, n %li, m %li, n_offset %li, n_local %li, m_local_read %li\n", 
    procid, ggi->n, ggi->m, ggi->n_offset, ggi->n_local, ggi->m_local_read);
  }

  uint64_t* gen_edges = (uint64_t*)malloc(ggi->m_local_read*2*sizeof(uint64_t));
  if (gen_edges == NULL)
    throw_err("generate_rand_out_edges(), unable to allocate resources\n", procid);

  xs1024star_t xs;
  xs1024star_seed((uint64_t)seed, &xs);

  uint64_t counter = 0;
  for (uint64_t i = 0; i < ggi->n_local; ++i) {
    for (uint64_t j = 0; j < edges_per_vert; ++j)
    {
      uint64_t v1 = xs1024star_next(&xs) % ggi->n;
      uint64_t v2 = xs1024star_next(&xs) % ggi->n;
      while (v1 == v2)
        v2 = xs1024star_next(&xs) % ggi->n;

      gen_edges[counter++] = v1;
      gen_edges[counter++] = v2;
    }
  }

  assert(counter == ggi->m_local_read*2);

  ggi->gen_edges = gen_edges;

  if (offset_vids)
  {
#pragma omp parallel for
    for (uint64_t i = 0; i < ggi->m_local_read*2; ++i)
    {
      uint64_t task_id = ggi->gen_edges[i] / (uint64_t)nprocs;
      uint64_t task = ggi->gen_edges[i] % (uint64_t)nprocs;
      uint64_t task_offset = task * (ggi->n / (uint64_t)nprocs);
      uint64_t new_vid = task_offset + task_id;
      new_vid = (new_vid >= ggi->n) ? (ggi->n - 1) : new_vid;
      ggi->gen_edges[i] = new_vid;
    }
  }

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d generate_rand_out_edges() %9.6f (s)\n", procid, elt);
  }

  if (debug) { printf("Task %d generate_rand_out_edges() success\n", procid); }
  return 0;
}



int generate_rmat_out_edges(graph_gen_data_t* ggi, 
  uint64_t num_verts, uint64_t edges_per_vert, bool offset_vids)
{
  if (debug) { printf("Task %d generate_rmat_out_edges() start\n", procid); }

  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  ggi->n = num_verts;
  ggi->m = num_verts * edges_per_vert;
  ggi->n_offset = (uint64_t)procid * (num_verts / (uint64_t)nprocs + 1);
  ggi->n_local = num_verts / (uint64_t)nprocs + 1;
  if (procid == nprocs - 1 && !offset_vids)
    ggi->n_local = ggi->n - ggi->n_offset;
  ggi->m_local_read = ggi->n_local * edges_per_vert;

  if (debug) { 
    printf("Task %d, n %li, m %li, n_offset %li, n_local %li, m_local_read %li\n", 
    procid, ggi->n, ggi->m, ggi->n_offset, ggi->n_local, ggi->m_local_read);
  }

  uint64_t* gen_edges = (uint64_t*)malloc(ggi->m_local_read*2*sizeof(uint64_t));
  if (gen_edges == NULL)
    throw_err("generate_rmat_out_edges(), unable to allocate resources\n", procid);

  //uint64_t counter = 0;
  double A = 0.45;
  double B = 0.15;
  double C = 0.15;
  double D = 0.25;
  double V = 0.05;

  //uint64_t temp_n = ggi->n / 2;

#pragma omp parallel
{
  xs1024star_t xs;
  xs1024star_seed((uint64_t)(seed + omp_get_thread_num()), &xs);

#pragma omp for 
  for (uint64_t i = 0; i < ggi->m_local_read; ++i)
  {
    double a = A; double b = B; double c = C; double d = D;
    uint64_t u = 0; uint64_t v = 0;
    uint64_t step = ggi->n;

    //uint64_t count = 0;
    do
    {
      while (step > 1)
      {
        double p = xs1024star_next_real(&xs);
        if (p < a) {}
        else if ((a < p) && (p < a+b))
          v = v + step;
        else if ((a+b < p) && (p < a+b+c))
          u = u + step;
        else if ((a+b+c < p) && (p < a+b+c+d))
        {
          u = u + step;
          v = v + step;
        }
        step = step / 2;


        if (xs1024star_next_real(&xs) > 0.5)
          a += a * V * xs1024star_next_real(&xs);
        else
          a -= a * V * xs1024star_next_real(&xs);
        if (xs1024star_next_real(&xs) > 0.5)
          b += b * V * xs1024star_next_real(&xs);
        else
          b += b * V * xs1024star_next_real(&xs);
        if (xs1024star_next_real(&xs) > 0.5)
          c += c * V * xs1024star_next_real(&xs);
        else
          c -= c * V * xs1024star_next_real(&xs);
        if (xs1024star_next_real(&xs) > 0.5)
          d += d * V * xs1024star_next_real(&xs);
        else
          d -= d * V * xs1024star_next_real(&xs);

        a = fabs(a);
        b = fabs(b);
        c = fabs(c);
        d = fabs(d);

        double S = a + b + c + d;
        a = a/S;
        b = b/S;
        c = c/S;
        d = d/S;
      }
      if (u == v)
      {
        //printf("%u %u %u %1.3lf %1.3lf %1.3lf %1.3lf\n", u, v, ++count, a, b, c, d);
        a = A; b = B; c = C; d = D;
        u = 0; v = 0;
        step = ggi->n;
      }
    } while (u == v);

    assert(u / 2 < ggi->n);
    assert(v / 2< ggi->n);
    gen_edges[i*2] = u / 2;
    gen_edges[i*2+1] = v / 2;
    //printf("%u %u\n", u, v);
  }
} // end parallel

  ggi->gen_edges = gen_edges;


  if (offset_vids)
  {
#pragma omp parallel for
    for (uint64_t i = 0; i < ggi->m_local_read*2; ++i)
    {
      uint64_t task_id = ggi->gen_edges[i] / (uint64_t)nprocs;
      uint64_t task = ggi->gen_edges[i] % (uint64_t)nprocs;
      uint64_t task_offset = task * (ggi->n / (uint64_t)nprocs);
      uint64_t new_vid = task_offset + task_id;
      new_vid = (new_vid >= ggi->n) ? (ggi->n - 1) : new_vid;
      ggi->gen_edges[i] = new_vid;
    }
  }

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d generate_rmat_out_edges() %9.6f (s)\n", procid, elt);
  }

  if (debug) { printf("Task %d generate_rmat_out_edges() success\n", procid); }
  return 0;
}




int generate_hd_out_edges(graph_gen_data_t* ggi, 
  uint64_t num_verts, uint64_t edges_per_vert, bool offset_vids)
{
  if (debug) { printf("Task %d generate_mesh_out_edges() start\n", procid); }
  
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  ggi->n = num_verts;
  ggi->m = num_verts * edges_per_vert;
  ggi->n_offset = (uint64_t)procid * (num_verts / (uint64_t)nprocs + 1);
  ggi->n_local = num_verts / (uint64_t)nprocs + 1;
  if (procid == nprocs - 1 && !offset_vids)
    ggi->n_local = ggi->n - ggi->n_offset;
  ggi->m_local_read =ggi->n_local * edges_per_vert;

  if (debug) { 
    printf("Task %d, n %li, m %li, n_offset %li, n_local %li, m_local_read %li\n", 
    procid, ggi->n, ggi->m, ggi->n_offset, ggi->n_local, ggi->m_local_read);
  }

  uint64_t* gen_edges = (uint64_t*)malloc(ggi->m_local_read*2*sizeof(uint64_t));
  if (gen_edges == NULL)
    throw_err("generate_mesh_out_edges(), unable to allocate resources\n", procid);

  xs1024star_t xs;
  xs1024star_seed((uint64_t)(seed), &xs);


  uint64_t counter = 0;
  for (uint64_t i = ggi->n_offset; i < (ggi->n_offset+ggi->n_local); ++i) 
  {
    uint64_t cur;
    if (i < edges_per_vert)
      cur = ggi->n - (edges_per_vert - i);
    else
      cur = (i - edges_per_vert);
    for (uint64_t j = 0; j < edges_per_vert; ++j)
    {
      while (xs1024star_next_real(&xs) < 0.5 || cur == i)
        ++cur;

      gen_edges[counter++] = i;
      gen_edges[counter++] = cur % ggi->n;
    }
  }

  assert(counter == ggi->m_local_read*2);

  ggi->gen_edges = gen_edges;


  if (offset_vids)
  {
#pragma omp parallel for
    for (uint64_t i = 0; i < ggi->m_local_read*2; ++i)
    {
      uint64_t task_id = ggi->gen_edges[i] / (uint64_t)nprocs;
      uint64_t task = ggi->gen_edges[i] % (uint64_t)nprocs;
      uint64_t task_offset = task * (ggi->n / (uint64_t)nprocs);
      uint64_t new_vid = task_offset + task_id;
      new_vid = (new_vid >= ggi->n) ? (ggi->n - 1) : new_vid;
      ggi->gen_edges[i] = new_vid;
    }
  }

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d generate_mesh_out_edges() %9.6f (s)\n", procid, elt);
  }

  if (debug) { printf("Task %d generate_mesh_out_edges() success\n", procid); }
  return 0;
}
