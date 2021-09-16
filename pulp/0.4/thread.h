#ifndef __THREAD_H__
#define __THREAD_H__

#define THREAD_QUEUE_SIZE 1024

inline void add_to_queue(int* thread_queue, int& thread_queue_size, 
                         int* queue_next, int& queue_size_next, int vert);
inline void empty_queue(int* thread_queue, int& thread_queue_size, 
                        int* queue_next, int& queue_size_next);


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


#endif