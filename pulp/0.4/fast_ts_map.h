#ifndef _FAST_TS_MAP_H_
#define _FAST_TS_MAP_H_

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define NULL_KEY 0

struct entry {
  uint64_t key;
  bool val;
  int count;
} ;

struct fast_ts_map {
  entry* arr;
  uint64_t capacity;
} ;

void init_map(fast_ts_map* map, uint64_t init_size);

void clear_map(fast_ts_map* map);

inline uint64_t hash64(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    return x;
}

inline bool test_set_value(fast_ts_map* map, 
  uint32_t src, uint32_t dst, int32_t weight)
{
  uint64_t key = (((uint64_t)src << 32) | (uint64_t)dst);
  uint64_t init_idx = hash64(key) % map->capacity;

  for (uint64_t idx = init_idx;; idx = (idx+1) % map->capacity)
  {
    bool test = false;
    //test = __sync_fetch_and_or(&map->arr[idx].val, true);
#pragma omp atomic capture
    { test = map->arr[idx].val; map->arr[idx].val = true; }

    // race condition handling below
    // - other thread which won slot might have not yet set key
    if (test == false) { // this thread got empty slot
      map->arr[idx].key = key;
#pragma omp atomic
      map->arr[idx].count += weight;
      
      return false;
    }
    else if (test == true) {// key already exists in table
      // wait for key to get set if it isn't yet
      while (map->arr[idx].key == NULL_KEY) {
        printf("."); // can comment this out and do other trivial work
      }

      if (map->arr[idx].key == key) {// this key already exists in table
#pragma omp atomic
        map->arr[idx].count += weight;
        
        //printf("updating weight %d %d %d %d\n", src, dst, weight, map->arr[idx].count);
        
        return true;
      }
    } // else slot is taken by another key, loop and increment
  }

  return false;
}

#endif
