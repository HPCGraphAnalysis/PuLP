CXX=g++
CXXFLAGS=-fopenmp -O3

all: pulp libpulp

pulp.o:
	$(CXX) $(CXXFLAGS) -c pulp.cpp

libpulp: pulp.o
	ar rvs libpulp.a pulp.o

pulp: pulp.o
	$(CXX) $(CXXFLAGS) -o pulp pulp_main.cpp pulp.o

clean:
	rm -f pulp *.o *.a
