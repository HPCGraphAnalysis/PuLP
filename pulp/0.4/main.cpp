
#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cctype>
#include <getopt.h>

#include "pulp.h"
#include "graph.h"
#include "io.h"

bool debug = false;
bool verbose = false;
int seed = 0;

void print_usage_full(char** argv)
{
  printf("To run: %s [graphfile] [num parts] [options]\n\n", argv[0]);
  printf("Options:\n");
  printf("\t-b [#.#]:\n");
  printf("\t\tImbalance constraint [default: 1.05 (5%%)]\n");
  printf("\t-o [file]:\n");
  printf("\t\tOutput parts file [default: graphname.part.numparts]\n");
  printf("\t-s [seed]:\n");
  printf("\t\tSet seed integer [default: random int]\n");
  exit(0);
}

/*
'##::::'##::::'###::::'####:'##::: ##:
 ###::'###:::'## ##:::. ##:: ###:: ##:
 ####'####::'##:. ##::: ##:: ####: ##:
 ## ### ##:'##:::. ##:: ##:: ## ## ##:
 ##. #: ##: #########:: ##:: ##. ####:
 ##:.:: ##: ##.... ##:: ##:: ##:. ###:
 ##:::: ##: ##:::: ##:'####: ##::. ##:
..:::::..::..:::::..::....::..::::..::
*/
int main(int argc, char** argv)
{
  setbuf(stdout, NULL);
  srand(time(0));
  if (argc < 3)
  {
    print_usage_full(argv);
    exit(0);
  }  

  graph* g = create_graph(argv[1]); 
  int num_parts = atoi(argv[2]);
  int* parts = NULL;
  
  double imb = 1.1;
  char parts_out[1024]; parts_out[0] = '\0';
  seed = rand();  

  char c;
  while ((c = getopt (argc, argv, "b:o:s:")) != -1)
  {
    switch (c)
    {
      case 'b':
        imb = strtod(optarg, NULL);
        break;
      case 'o':
        strcat(parts_out, optarg);
        break;
      case 's':
        seed = atoi(optarg);
        break;
      case '?':
        if (optopt == 'b' || optopt == 'o' || optopt == 's')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `\\x%x'.\n",
      optopt);
        print_usage_full(argv);
      default:
        abort();
    }
  }  

  double elt = omp_get_wtime();
  
  pulp_run_coarsen(g, num_parts, parts, imb);
  evaluate_quality(g, num_parts, parts);
  
  printf("Done: %lf\n", omp_get_wtime() - elt);

  return 0;
}
