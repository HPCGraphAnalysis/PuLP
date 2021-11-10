
#include <omp.h>
#include <sys/time.h>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include "util.h"
#include "rand.h"
#include "graph.h"

void parallel_prefixsums(int* in_array, int* out_array, int size)
{
  int* thread_sums;

#pragma omp parallel
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();
#pragma omp single
{
  thread_sums = new int[nthreads+1];
  thread_sums[0] = 0;
}

  int my_sum = 0;
#pragma omp for schedule(static)
  for(int i = 0; i < size; ++i) {
    my_sum += in_array[i];
    out_array[i] = my_sum;
  }

  thread_sums[tid+1] = my_sum;
#pragma omp barrier

  int my_offset = 0;
  for(int i = 0; i < (tid+1); ++i)
    my_offset += thread_sums[i];

#pragma omp for schedule(static)
  for(int i = 0; i < size; ++i)
    out_array[i] += my_offset;
}

  delete [] thread_sums;
}

int binary_search(double* array, double value, int max_index)
{
  bool found = false;
  int index = 0;
  int bound_low = 0;
  int bound_high = max_index;
  while (!found)
  {
    //rintf("high %d low %d index %d cur %lf value %lf\n", 
    //  bound_high, bound_low, index, array[index], value);

    index = (bound_high + bound_low) / 2;
    if (array[index] <= value && array[index+1] > value)
    {
      return index;
    }
    else if (array[index] <= value)
      bound_low = index;
    else if (array[index] > value)
      bound_high = index;
  }

  return index;
}

void quicksort(int* arr1, int left, int right) 
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
    quicksort(arr1, left, j);
  if (i < right)
    quicksort(arr1, i, right);
}

void quicksort(double* arr1, int left, int right) 
{
  int i = left;
  int j = right;
  double temp;
  double pivot = arr1[(left + right) / 2];

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
    quicksort(arr1, left, j);
  if (i < right)
    quicksort(arr1, i, right);
}


void quicksort(int* arr1, int* arr2, int left, int right) 
{
  int i = left;
  int j = right;
  int temp; int temp2;
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
      temp2 = arr2[i];
      arr2[i] = arr2[j];
      arr2[j] = temp2;
      ++i;
      --j;
    }
  }

  if (j > left)
    quicksort(arr1, arr2, left, j);
  if (i < right)
    quicksort(arr1, arr2, i, right);
}
