/*
//@HEADER
// *****************************************************************************
//
// PuLP: Multi-Objective Multi-Constraint Partitioning Using Label Propagation
//              Copyright (2014) Sandia Corporation
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
// Questions?  Contact  George M. Slota (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//
// *****************************************************************************
//@HEADER
*/

struct xs1024star_t {
  uint64_t s[16];
  int64_t p;
} ;


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
