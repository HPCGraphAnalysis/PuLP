
***********************************************
*:'##::::'##::::'##:::::::::::::::::::::::::: *
*:. ##::'##::::: ##:::::::::::::::::::::::::: *
*::. ##'##:::'########::'#######:::'#######:: *
*:::. ###::::... ##::::'##.... ##:'##... ##:: *
*::: ## ##:::::: ##:::: ##::::..:: ##::: ##:: *
*:: ##:. ##::::: ##:::: ##:::::::: ##::: ##:: *
*: ##:::. ##:::: ##:::: ##::::::::  ########: *
*:..:::::..:::::..:::::..::::::::::........:: *************
*:::::::::::'########:::::::::::::'##:::::::'########:::: *
*::::::::::: ##.... ##:::::::::::: ##::::::: ##.... ##::: *
*::::::::::: ##:::: ##:::::::::::: ##::::::: ##:::: ##::: *
*::::::::::: ########::'##::::'##: ##::::::: ########:::: *
*::::::::::: ##.....::: ##:::: ##: ##::::::: ##.....::::: *
*::::::::::: ##:::::::: ##:::: ##: ##::::::: ##:::::::::: *
*::::::::::: ##::::::::. #######:: ########: ##:::::::::: *
*:::::::::::..::::::::::.......:::........::..::::::::::: *
*********************** Version 0.1 ***********************
********************************************************************************

XtraPuLP: Multi-Objective Multi-Constraint Partitioning using Label Propagation
                    Copyright (2016) Sandia Corporation

Questions?  Contact George M. Slota    (gmslota@sandia.gov)
                    Siva Rajamanickam  (srajama@sandia.gov)

********************************************************************************
Version info:

v0.21 -- 23 April 2017
--Bug fixes
    *pulp->avg_size wasn't being set correctly in weighted case
    *missing parantheses affected refinement calculation in pulp_v routines
    *other minor fixes and changes 

v0.2 -- 25 July 2016
--Initial release - version number corresponds to similar functionality as PuLP


********************************************************************************
To make:

1.) Set MPICXX in Makefile to your c++ compiler, adjust CXXFLAGS if necessary
-OpenMP 3.1 support is required for parallel execution
-No other dependencies needed

2.) $ make 
-This will make xtrapulp executable and library

3.) $ make libxtrapulp
-This will just make libxtrapulp.a static library for use with xtrapulp.h


********************************************************************************
To run:

$ mpirun -n [#] ./xtrapulp [graphfile] [num parts] [options]

-[graphfile] is of binary directed edge list format with 32 or 64 bit unsigned integers as vertex labels; vertex labels are assumed to begin at 0. To switch between 32 and 64 bit, alter load_graph_edges_32() to load_graph_edges_64() in main.cpp. 

A binary file containing a list of unsigned 0-indexed 32 or 64 bit integers like this:
v0 v1 v1 v2 v2 v3 v3 v4

would indicate a graph with the following directed edges:
v0 -> v1
v1 -> v2
v2 -> v3
v3 -> v0

[num parts] is the number of desired partitions (>=2)

Options:
  -v [#.#]:
      Vertex balance constraint [default: 1.10 (10%)]
  -e [#.#]:
      Edge balance constraint [default: none; 1.50 (50%) when -c option is used]
  -c:
      Attempt to minimize per-part cut in addition to total edgecut
  -d:
    Use round-robin instead of vertex-based distribution
       (Might help with load imbalance with many MPI tasks)
  -m [#]:
    Generate multiple partitions [default: 1]
  -o [file]:
      Output parts file [default: graphname.part.numparts(.#)]
  -i [file]:
      Input parts file [default: none]
  -q:
      Evaluate generated partition quality

[Input/Output Files] are text files that have n lines. Each line contains a single integer [0...(num parts-1)] that corresponds to the part assignment of the vertex identifier of that line number. I.e., a '5' on line 7 indicates that vertex 7 is assigned to part 5.


********************************************************************************
Examples:

1.) Generate 16 parts of Live Journal network with default vertex balance constraint, only minimizing edge cut, and using the default BFS-based initialization; perform quality evaluation

$ mpirun -n [#] ./xtrapulp LiveJournal.adj 16 -q


2.) Generate 16 parts of Live Journal network with tighter vertex balance constraint (3%), only minimizing edge cut, and using label propagation-based initialization

$ mpirun -n [#] ./xtrapulp LiveJournal.adj 16 -v 1.03 -l


3.) Generate 128 parts of Live Journal network with tighter vertex balance constraint and looser edge balance constraint while minimizing edge cut and using BSF-based initialization

$ mpirun -n [#] ./xtrapulp LiveJournal.adj 128 -v 1.03 -e 2.5


4.) Generate 128 parts of Live Journal network with tighter vertex balance constraint and looser edge balance constraint while minimizing both edge cut and max per-part edge cut, write output part file to temp.parts

$ mpirun -n [#] ./xtrapulp LiveJournal.adj 128 -v 1.03 -e 2.5 -c -o temp.parts


5.) Generate 16 parts of Live Journal network with tighter vertex balance constraint and looser edge balance constraint while minimizing both edge cut and max per-part edge cut, reading in an initial partition, and writing output part file to temp.parts

$ mpirun -n [#] ./xtrapulp LiveJournal.adj 128 -v 1.03 -e 2.5 -c -i temp.parts -o temp.parts.new


6.) Generate 5 different partitionings of Live Journal network each having 16 parts with tighter vertex balance constraint and looser edge balance constraint while doing label propagation-based initialization

$ mpirun -n [#] ./xtrapulp LiveJournal.adj 128 -v 1.03 -e 2.5 -o temp.parts -m 5 -l
Note: outputs will be written to temp.parts.0, temp.parts.1, ..., temp.parts.4


********************************************************************************
Notes:

1.) The use case for XtraPuLP is small-world graphs with skewed degree distributions. Other partitioners are more well-suited for handling regular mesh-like graphs. XtraPuLP can partition such graphs quickly, but the quality won't be as high as when using other tools.
2.) Be careful when determining vertex and edge balance constraints, as you might create a problem for which there is no solution; XtraPuLP will take longer to execute and the output won't be as nice as you're expecting


********************************************************************************
Known issues:

1.) Minimal error checking for inputs, exact format listed above for graphfile required; be careful when passing filepaths since there's no error handling

2.) No options for adjusting iteration counts for various algorithm loops, however, these can be changed in the source files
