MPICXX = mpicxx
CXXFLAGS = -fopenmp -O3 -Wall
LINKFLAGS = -fopenmp -O3 -Wall
TARGET = xtrapulp
LIBTARGET = libxtrapulp.a
TOCOMPILE = util.o generate.o pulp_util.o pulp_data.o fast_map.o dist_graph.o comms.o io_pp.o main.o
FORLIBPULP = util.o generate.o pulp_util.o pulp_data.o fast_map.o dist_graph.o comms.o io_pp.o pulp_init.o pulp_vec.o pulp_ve.o pulp_v.o xtrapulp.o


all: libxtrapulp $(TOCOMPILE)
	$(MPICXX) $(LINKFLAGS) -o $(TARGET) $(TOCOMPILE) $(LIBTARGET)

libxtrapulp: $(FORLIBPULP)
	ar rvs $(LIBTARGET) *.o

.cpp.o:
	$(MPICXX) $(CXXFLAGS) -c $*.cpp

clean:
	rm -f *.o *.a $(TARGET)

