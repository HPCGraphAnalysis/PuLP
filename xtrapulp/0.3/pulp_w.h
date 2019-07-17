#ifndef _PULP_W_H_
#define _PULP_W_H_

int pulp_w(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
            pulp_data_t *pulp,            
            uint64_t outer_iter, 
            uint64_t balance_iter, uint64_t refine_iter, 
            uint64_t weight_index, double* constraints);

#endif
