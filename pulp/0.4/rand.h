#ifndef _RAND_H_
#define _RAND_H_

#include <cstdint>

struct xs1024star_t {
  uint64_t s[16];
  int64_t p;
} ;

uint64_t xs1024star_next(xs1024star_t* xs);

double xs1024star_next_real(xs1024star_t* xs);

void xs1024star_seed(uint64_t seed, xs1024star_t* xs);

void xs1024star_seed(xs1024star_t* xs);

#endif
