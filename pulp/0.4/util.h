#ifndef _UTIL_H_
#define _UTIL_H_

#include <unordered_map>
#include "graph.h"

void parallel_prefixsums(int* in_array, int* out_array, int size);

int binary_search(double* array, double value, int max_index);

void quicksort(int* arr1, int left, int right);

void quicksort(double* arr1, int left, int right);

void quicksort(int* arr1, int* arr2, int left, int right);

#endif
