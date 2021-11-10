CXX = g++
CXXFLAGS = -I /usr/include/eigen3/ -fopenmp -O0 -g -Wall -std=c++11
LINKFLAGS = -fopenmp -O0- -g  -Wall -std=c++11
TARGET = pulp
TOCOMPILE = util.o rand.o fast_ts_map.o graph.o io.o coarsen.o pulp_v.o init.o init_eigen.o pulp.o main.o


all: $(TOCOMPILE)
	$(CXX) $(LINKFLAGS) -o $(TARGET) $(TOCOMPILE)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $*.cpp

clean:
	rm -f $(TOCOMPILE) $(TARGET)

