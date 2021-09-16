
#include <cstdlib>
#include <cstdint>

#include "rand.h"

extern int seed;

uint64_t xs1024star_next(xs1024star_t* xs) 
{
   const uint64_t s0 = xs->s[xs->p];
   uint64_t s1 = xs->s[xs->p = (xs->p + 1) & 15];
   s1 ^= s1 << 31;
   xs->s[xs->p] = s1 ^ s0 ^ (s1 >> 11) ^ (s0 >> 30);
   return xs->s[xs->p] * uint64_t(1181783497276652981U);
}

double xs1024star_next_real(xs1024star_t* xs) 
{
   const uint64_t s0 = xs->s[xs->p];
   uint64_t s1 = xs->s[xs->p = (xs->p + 1) & 15];
   s1 ^= s1 << 31;
   xs->s[xs->p] = s1 ^ s0 ^ (s1 >> 11) ^ (s0 >> 30);
   double ret = (double)(xs->s[xs->p] * uint64_t(1181783497276652981U));   
   return ret /= (double)uint64_t(18446744073709551615U);
}

void xs1024star_seed(uint64_t seed, xs1024star_t* xs) 
{
  for (uint64_t i = 0; i < 16; ++i)
  {
    uint64_t z = (seed += uint64_t(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * uint64_t(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * uint64_t(0x94D049BB133111EB);
    xs->s[i] = z ^ (z >> 31);
  }
  xs->p = 0;
}

void xs1024star_seed(xs1024star_t* xs) 
{
  if (seed == 0)
    seed = (unsigned long)rand();
  
  for (uint64_t i = 0; i < 16; ++i)
  {
    uint64_t z = (seed += uint64_t(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * uint64_t(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * uint64_t(0x94D049BB133111EB);
    xs->s[i] = z ^ (z >> 31);
  }
  xs->p = 0;
}
