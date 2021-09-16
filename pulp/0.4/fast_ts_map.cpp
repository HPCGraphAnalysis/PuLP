
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "fast_ts_map.h"

void init_map(fast_ts_map* map, uint64_t init_size)
{
  map->arr = (entry*)malloc(init_size*sizeof(entry));
  if (map->arr == NULL) {
    printf("init_map(), unable to allocate resources\n");
    exit(0);
  }

  map->capacity = init_size;
 
#pragma omp parallel for
  for (uint64_t i = 0; i < map->capacity; ++i) {
    map->arr[i].key = NULL_KEY;
    map->arr[i].val = false;
    map->arr[i].count = 0;
  }
}

void clear_map(fast_ts_map* map)
{
  free(map->arr);
  map->capacity = 0;
}
