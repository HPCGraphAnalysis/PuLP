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

int repart(dist_graph_t *g, int32_t* local_parts)
{
  MPI_Barrier(MPI_COMM_WORLD);
  double elt = timer();

  int32_t* sendcounts = (int32_t*)malloc(nprocs*sizeof(int32_t));
  assert(sendcounts != NULL);  
  int32_t* recvcounts = (int32_t*)malloc(nprocs*sizeof(int32_t));
  assert(recvcounts != NULL);
  for (uint32_t i = 0; i < nprocs; ++i)
  {
    sendcounts[i] = 0;
    recvcounts[i] = 0;
  }
  for (uint32_t i = 0; i < g->n_local; ++i)
  {
    int32_t rank = local_parts[i];
    ++sendcounts[rank];
  }

  MPI_Alltoall(sendcounts, 1, MPI_INT32_T, 
    recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);
   
  int32_t* sdispls = (int32_t*)malloc(nprocs*sizeof(int32_t));
  assert(sdispls != NULL);  
  int32_t* sdispls_cpy = (int32_t*)malloc(nprocs*sizeof(int32_t));
  assert(sdispls_cpy != NULL);
  int32_t* rdispls = (int32_t*)malloc(nprocs*sizeof(int32_t));
  assert(rdispls != NULL);   

  sdispls[0] = 0;
  sdispls_cpy[0] = 0;
  rdispls[0] = 0;
  for (uint32_t i = 0; i < nprocs-1; ++i)
  {
    sdispls[i+1] = sdispls[i] + sendcounts[i];
    sdispls_cpy[i+1] = sdispls[i+1];
    rdispls[i+1] = rdispls[i] + recvcounts[i];
  }
  int32_t total_send_deg = sdispls[nprocs-1] + sendcounts[nprocs-1];
  int32_t total_recv_deg = rdispls[nprocs-1] + recvcounts[nprocs-1];
  printf("%d totals %d %d %li\n", procid, total_send_deg, total_recv_deg, g->n_local);
  assert(total_send_deg == g->n_local);

  uint32_t* sendbuf_vids;
  uint32_t* sendbuf_deg_out;
  uint32_t* sendbuf_deg_in;
  uint32_t* recvbuf_vids;
  uint32_t* recvbuf_deg_out;
  uint32_t* recvbuf_deg_in;
  sendbuf_vids = (uint32_t*)malloc(total_send_deg*sizeof(uint32_t));
  assert(sendbuf_vids != NULL);   
  sendbuf_deg_out = (uint32_t*)malloc(total_send_deg*sizeof(uint32_t));
  assert(sendbuf_deg_out != NULL);   
  sendbuf_deg_in = (uint32_t*)malloc(total_send_deg*sizeof(uint32_t));
  assert(sendbuf_deg_in != NULL);    
  recvbuf_vids = (uint32_t*)malloc(total_recv_deg*sizeof(uint32_t));
  assert(recvbuf_vids != NULL);  
  recvbuf_deg_out = (uint32_t*)malloc(total_recv_deg*sizeof(uint32_t));
  assert(recvbuf_deg_out != NULL);   
  recvbuf_deg_in = (uint32_t*)malloc(total_recv_deg*sizeof(uint32_t));
  assert(recvbuf_deg_in != NULL);
  for (uint32_t i = 0; i < g->n_local; ++i)
  {    
    uint32_t rank = local_parts[i];
    uint32_t snd_index = sdispls_cpy[rank]++;
    sendbuf_vids[snd_index] = g->local_unmap[i];
    sendbuf_deg_out[snd_index] = (uint32_t)out_degree(g, i);
    sendbuf_deg_in[snd_index] = (uint32_t)in_degree(g, i);
  }

  MPI_Alltoallv(sendbuf_vids, sendcounts, sdispls, MPI_UINT32_T, 
    recvbuf_vids, recvcounts, rdispls, MPI_UINT32_T, MPI_COMM_WORLD);
  MPI_Alltoallv(sendbuf_deg_out, sendcounts, sdispls, MPI_UINT32_T, 
    recvbuf_deg_out, recvcounts, rdispls, MPI_UINT32_T, MPI_COMM_WORLD);
  MPI_Alltoallv(sendbuf_deg_in, sendcounts, sdispls, MPI_UINT32_T, 
    recvbuf_deg_in, recvcounts, rdispls, MPI_UINT32_T, MPI_COMM_WORLD);
  free(sendbuf_vids);
  free(sendbuf_deg_out);
  free(sendbuf_deg_in);

  for (uint32_t i = 0; i < nprocs; ++i)
  {
    sendcounts[i] = 0;
    recvcounts[i] = 0;
  }
  for (uint32_t i = 0; i < g->n_local; ++i)
  {
    int32_t rank = local_parts[i];
    sendcounts[rank] += (int32_t)out_degree(g, i);
  }

  MPI_Alltoall(sendcounts, 1, MPI_INT32_T, 
    recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

  sdispls[0] = 0;
  sdispls_cpy[0] = 0;
  rdispls[0] = 0;
  for (uint32_t i = 0; i < nprocs-1; ++i)
  {
    sdispls[i+1] = sdispls[i] + sendcounts[i];
    sdispls_cpy[i+1] = sdispls[i+1];
    rdispls[i+1] = rdispls[i] + recvcounts[i];
  }
  uint32_t total_send_out = sdispls[nprocs-1] + sendcounts[nprocs-1];
  uint32_t total_recv_out = rdispls[nprocs-1] + recvcounts[nprocs-1];
  assert(total_send_out == g->m_local_out);

  uint32_t* sendbuf_e_out; 
  uint32_t* recvbuf_e_out; 
  sendbuf_e_out = (uint32_t*)malloc(total_send_out*sizeof(uint32_t));
  assert(sendbuf_e_out != NULL);
  recvbuf_e_out = (uint32_t*)malloc(total_recv_out*sizeof(uint32_t));
  assert(recvbuf_e_out != NULL);

  uint32_t counter = 0;
  for (uint32_t i = 0; i < g->n_local; ++i)
  {
    uint32_t out_degree = out_degree(g, i);
    uint32_t* outs = out_vertices(g, i);
    uint32_t rank = local_parts[i];
    uint32_t snd_index = sdispls_cpy[rank];
    sdispls_cpy[rank] += out_degree;
    for (uint32_t j = 0; j < out_degree; ++j)
    {
      assert(outs[j] < g->n_total);
      uint32_t out;
      if (outs[j] < g->n_local)
        out = g->local_unmap[outs[j]];
      else
        out = g->ghost_unmap[outs[j]-g->n_local];
      sendbuf_e_out[snd_index++] = out;
      counter++;
      assert(out < g->n);
    }
  }
  assert(counter == g->m_local_out);

  MPI_Alltoallv(sendbuf_e_out, sendcounts, sdispls, MPI_UINT32_T, 
    recvbuf_e_out, recvcounts, rdispls, MPI_UINT32_T, MPI_COMM_WORLD);
  free(sendbuf_e_out);
  free(g->out_edges);
  free(g->out_degree_list);
  g->out_edges = recvbuf_e_out;
  g->m_local_out = (int64_t)total_recv_out;
  g->out_degree_list = (uint32_t*)malloc((total_recv_deg+1)*sizeof(uint32_t));
  g->out_degree_list[0] = 0;
  for (uint32_t i = 0; i < total_recv_deg; ++i)
    g->out_degree_list[i+1] = g->out_degree_list[i] + recvbuf_deg_out[i];
  printf("BLAH %d %u %u\n", procid, g->out_degree_list[total_recv_deg], g->m_local_out);
  assert(g->out_degree_list[total_recv_deg] == g->m_local_out);
  free(recvbuf_deg_out);



  for (uint32_t i = 0; i < nprocs; ++i)
    sendcounts[i] = 0;
  for (uint32_t i = 0; i < g->n_local; ++i)
  {
    int32_t rank = (uint32_t)local_parts[i];
    sendcounts[rank] += (int32_t)in_degree(g, i);
  }

  MPI_Alltoall(sendcounts, 1, MPI_INT32_T, 
    recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

  sdispls[0] = 0;
  sdispls_cpy[0] = 0;
  rdispls[0] = 0;
  for (uint32_t i = 0; i < nprocs-1; ++i)
  {
    sdispls[i+1] = sdispls[i] + sendcounts[i];
    sdispls_cpy[i+1] = sdispls[i+1];
    rdispls[i+1] = rdispls[i] + recvcounts[i];
  }
  uint32_t total_send_in = sdispls[nprocs-1] + sendcounts[nprocs-1];
  uint32_t total_recv_in = rdispls[nprocs-1] + recvcounts[nprocs-1];
  assert(total_send_in == g->m_local_in);

  uint32_t* sendbuf_e_in; 
  uint32_t* recvbuf_e_in; 
  sendbuf_e_in = (uint32_t*)malloc(total_send_in*sizeof(uint32_t));
  assert(sendbuf_e_in != NULL);
  recvbuf_e_in = (uint32_t*)malloc(total_recv_in*sizeof(uint32_t));
  assert(recvbuf_e_in != NULL);

  counter = 0;
  for (uint32_t i = 0; i < g->n_local; ++i)
  {
    uint32_t in_degree = in_degree(g, i);
    uint32_t* ins = in_vertices(g, i);
    uint32_t rank = local_parts[i];
    uint32_t snd_index = sdispls_cpy[rank];
    sdispls_cpy[rank] += in_degree;
    for (uint32_t j = 0; j < in_degree; ++j)
    {
      assert(ins[j] < g->n_total);
      uint32_t in;
      if (ins[j] < g->n_local)
        in = g->local_unmap[ins[j]];
      else
        in = g->ghost_unmap[ins[j]-g->n_local];
      sendbuf_e_in[snd_index++] = in;
      counter++;
      assert(in < g->n);
    }
  }
  assert(counter == g->m_local_in);

  MPI_Alltoallv(sendbuf_e_in, sendcounts, sdispls, MPI_UINT32_T, 
    recvbuf_e_in, recvcounts, rdispls, MPI_UINT32_T, MPI_COMM_WORLD);
  free(sendbuf_e_in);
  free(g->in_edges);
  free(g->in_degree_list);
  g->in_edges = recvbuf_e_in;
  g->m_local_in = (int64_t)total_recv_in;
  g->in_degree_list = (uint32_t*)malloc((total_recv_deg+1)*sizeof(uint32_t));
  g->in_degree_list[0] = 0;
  for (uint32_t i = 0; i < total_recv_deg; ++i)
    g->in_degree_list[i+1] = g->in_degree_list[i] + recvbuf_deg_in[i];
  assert(g->in_degree_list[total_recv_deg] == g->m_local_in);
  free(recvbuf_deg_in);


  free(g->local_unmap);
  g->local_unmap = (uint32_t*)malloc(total_recv_deg*sizeof(uint32_t));
  for (uint32_t i = 0; i < total_recv_deg; ++i)
    g->local_unmap[i] = recvbuf_vids[i];
  free(recvbuf_vids);

  g->n_local = total_recv_deg;

  elt = timer() - elt;

  printf("%d done repart %li %li %li, %9.6lf (s)\n", procid, g->n_local, g->m_local_in, g->m_local_out, elt);

}


int get_ghost_tasks(dist_graph_t *g)
{
  double elt = timer();

  int64_t n_local_max = g->n_local;
  int32_t cur_size;
  MPI_Allreduce(MPI_IN_PLACE, &n_local_max, 1,
    MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  uint32_t* buf = (uint32_t*)malloc(n_local_max*sizeof(uint32_t));
  uint32_t* tmp_buf;

  for (uint32_t p = 0; p < nprocs; ++p)  
  {
    if (p == procid)
    {
      printf("%d my time to shine\n", procid);
      tmp_buf = g->local_unmap;
      cur_size = (int32_t)g->n_local;
    }
    else
      tmp_buf = buf;

    MPI_Bcast(&cur_size, 1, MPI_INT32_T, p, MPI_COMM_WORLD);
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(tmp_buf, cur_size, MPI_UINT32_T, p, MPI_COMM_WORLD);
    //MPI_Barrier(MPI_COMM_WORLD);

    if (p != procid)
    {
      //printf("%d putting this shit to biz\n", procid);
#pragma omp parallel for
      for (uint32_t i = 0; i < (uint32_t)cur_size; ++i)
      {
#if NO_HASH
        uint32_t val = g->mapper[buf[i]];
#else 
        uint32_t val = get_value(&g->map, buf[i]);
#endif
        if (val != NULL_KEY)
          g->ghost_tasks[val-g->n_local] = p;
      }
    }
  }

  for (uint32_t i = 0; i < (uint32_t)g->n_ghost; ++i)
    if (g->ghost_tasks[i] >= nprocs)
      printf("EROR %d -- %u, %u %u\n", procid, i, g->ghost_unmap[i], g->ghost_tasks[i]);


  elt = timer() - elt;

  printf("%d done getting ghost tasks %9.6lf (s)\n", procid, elt);  
}
