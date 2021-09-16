#ifndef __THREAD_H__
#define __THREAD_H__

#define THREAD_QUEUE_SIZE 768

inline void add_to_queue(int* thread_queue, int& thread_queue_size, 
                         int* queue_next, int& queue_size_next, int vert);
inline void empty_queue(int* thread_queue, int& thread_queue_size, 
                        int* queue_next, int& queue_size_next);
inline void add_to_queues(int* thread_queue, int& thread_queue_size, 
                          int* queue1, int* queue2, int* queue3,
                          int& queue_size,
                          int val1, int val2, int val3);
inline void empty_queues(int* thread_queue, int& thread_queue_size, 
                          int* queue1, int* queue2, int* queue3,
                          int& queue_size);


inline void add_to_queue(int* thread_queue, int& thread_queue_size, 
                         int* queue_next, int& queue_size_next, int vert)
{
  thread_queue[thread_queue_size++] = vert;

  if (thread_queue_size == THREAD_QUEUE_SIZE)
    empty_queue(thread_queue, thread_queue_size, 
                queue_next, queue_size_next);
}

inline void empty_queue(int* thread_queue, int& thread_queue_size, 
                        int* queue_next, int& queue_size_next)
{
  int start_offset;

#pragma omp atomic capture
  start_offset = queue_size_next += thread_queue_size;

  start_offset -= thread_queue_size;
  for (int i = 0; i < thread_queue_size; ++i)
    queue_next[start_offset + i] = thread_queue[i];
  thread_queue_size = 0;
}

inline void add_to_queues(int* thread_queue, int& thread_queue_size, 
                          int* queue1, int* queue2, int* queue3,
                          int& queue_size,
                          int val1, int val2, int val3)
{
  thread_queue[thread_queue_size++] = val1;
  thread_queue[thread_queue_size++] = val2;
  thread_queue[thread_queue_size++] = val3;

  if (thread_queue_size == THREAD_QUEUE_SIZE)
    empty_queues(thread_queue, thread_queue_size, 
                  queue1, queue2, queue3,
                  queue_size);
}

inline void empty_queues(int* thread_queue, int& thread_queue_size, 
                          int* queue1, int* queue2, int* queue3,
                          int& queue_size)
{
  int start_offset;
  int num_new_edges = (thread_queue_size / 3);

#pragma omp atomic capture
  start_offset = queue_size += num_new_edges;

  start_offset -= num_new_edges;
  for (int i = 0; i < num_new_edges; ++i) {
    queue1[start_offset + i] = thread_queue[3*i];
    queue2[start_offset + i] = thread_queue[3*i+1];
    queue3[start_offset + i] = thread_queue[3*i+2];
  }
  thread_queue_size = 0;
}

#endif